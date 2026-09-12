"""Source adapter: targeted read-only fetch of a complete CI job log, and
excerpt-to-source mapping.

The fetch is deliberately narrow: one known repository / run / attempt / job,
limited timeout and one retry, result cached under cache/. Credentials are
taken (in this order) from $DEMO_A_GITHUB_TOKEN, $GITHUB_TOKEN, $GH_TOKEN and
are only attached to the request header; they are never printed or written
anywhere.

Failure kinds are distinguished so callers can tell them apart and not infer
a root cause from a status code alone. The first two are deliberately split:
"no credential was sent" is NOT the same as "a credential was sent but the
server rejected it", and a 401/403 alone does not prove which happened.

  fetched               — the bytes were downloaded successfully
  cached                — a previously verified cached copy was returned
  no_credential_sent    — 401/403 with NO Authorization header attached
                          (credentials were never sent)
  credential_rejected   — 401 with an Authorization header attached; the token
                          was rejected as invalid / expired / revoked
  scope_denied          — 403 with a token attached; scope / visibility / access
                          insufficient (only asserted when the 403 body supports it)
  rate_limited          — 403 with X-RateLimit-Remaining=0 or a secondary
                          rate-limit / abuse message
  not_found             — 404 (job/run/attempt no longer exists)
  gone                  — 410 (log expired and purged by GitHub)
  network_error         — connection-level failure
  timeout               — request timed out

Source identity is a separate concern from content integrity. Downloading the
job-log endpoint returns bytes but does NOT, by itself, prove the job belongs
to the declared repo / run / attempt. `fetch_job_log` therefore (when asked)
cross-checks the job's run_id / run_attempt via the read-only jobs endpoint
and records `identity_verified`. A recorded sha256 only proves the bytes are
consistent with what was stored — it does NOT prove provenance.

A manual_import path is also exposed as a backup: callers can paste an
already-fetched log into the cache directly. Because a manual import carries
only the caller's claim about provenance, it is always registered with
`identity_verified=False` (sha256 proves content consistency only).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from materials import HarnessError

DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 1

# GitHub Actions log lines are prefixed with a timestamp, e.g.
# 2026-09-10T13:32:45.1234567Z <content>
_TIMESTAMP_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z ")


def _token_from_env() -> tuple[str | None, str | None]:
    """Return (token, source) where source names which env var provided it.

    Never log the token value; only the source name. Lookup order is explicit
    so users can tell which variable is in effect.
    """
    for var in ("DEMO_A_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(var)
        if value:
            return value, var
    return None, None


def _cache_paths(cache_dir: Path, repo: str, run_id: int, attempt: int, job_id: int) -> tuple[Path, Path]:
    slug = repo.replace("/", "_")
    stem = f"{slug}_run{run_id}_attempt{attempt}_job{job_id}"
    return cache_dir / f"{stem}.log", cache_dir / f"{stem}.meta.json"


def _load_cache(meta_path: Path, log_path: Path) -> dict | None:
    if not (meta_path.is_file() and log_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "fetched":
        return None
    current = hashlib.sha256(log_path.read_bytes()).hexdigest()
    if current != meta.get("sha256"):
        return None
    meta["status"] = "cached"
    meta["cached"] = True
    return meta


def _classify_http_error(
    code: int,
    authenticated: bool,
    body_text: str | None,
    headers: dict | None,
) -> tuple[str, str]:
    """Distinguish auth/scope/rate/not-found/gone without relying on code alone.

    Returns (status_kind, message). Reads GitHub's documented JSON error
    body when available (it carries a "message" string), and the
    X-RateLimit-* headers for rate-limit classification.
    """
    body_msg = None
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                body_msg = parsed.get("message")
        except (ValueError, TypeError):
            body_msg = None

    if code == 404:
        return "not_found", f"GitHub API returned HTTP 404 ({body_msg or 'resource not found'})"
    if code == 410:
        return "gone", f"GitHub API returned HTTP 410 ({body_msg or 'log expired'})"

    if code in (401, 403):
        # Rate limit: prefer it over generic 403 when GitHub says so.
        if headers and headers.get("X-RateLimit-Remaining") == "0":
            return "rate_limited", (
                f"GitHub API rate limit exhausted ({body_msg or 'try again later'})"
            )
        # Secondary rate limit messages contain "abuse" or "secondary".
        if body_msg and any(tag in body_msg.lower() for tag in ("secondary rate", "abuse")):
            return "rate_limited", f"GitHub API secondary rate limit hit ({body_msg})"

        if not authenticated:
            # NO Authorization header was attached. Whatever the server says,
            # the harness never sent a credential, so this must be reported as
            # "no credential sent" — NOT as "credential rejected".
            return "no_credential_sent", (
                f"GitHub API HTTP {code} with NO Authorization header attached; "
                f"server says: {body_msg or '(no body message)'}"
            )

        if code == 401:
            # Authorization header WAS attached and the server replied 401:
            # the token was rejected (bad / expired / revoked).
            return "credential_rejected", (
                f"GitHub API HTTP 401 with Authorization header attached; the token "
                f"was REJECTED ({body_msg or 'Bad credentials / token expired / revoked'})"
            )

        # 403 with a token attached is most likely a scope / visibility issue.
        scope_hint = "token accepted but scope/visibility insufficient"
        if body_msg:
            scope_hint = f"{scope_hint} ({body_msg})"
        return "scope_denied", scope_hint

    return "network_error", f"GitHub API returned HTTP {code} ({body_msg or 'unexpected'})"


def _verify_job_identity(
    repo: str,
    job_id: int,
    expected_run_id: int,
    expected_attempt: int,
    token: str | None,
    timeout: float,
) -> dict:
    """Cross-check that a job actually belongs to the declared run/attempt.

    Calls the read-only jobs endpoint ``GET /repos/{repo}/actions/jobs/{job_id}``
    and compares its ``run_id`` / ``run_attempt`` against the values the caller
    claims. This is what turns "we registered a hash for repo/run/attempt/job"
    into "we verified the log belongs to that run".

    Returns a dict with ``verified`` (bool) and either the observed values or
    the failure reason. Never raises — a failure to verify means ``verified``
    is False, leaving identity unverified rather than silently asserted.
    """
    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "demo_a_foundation-harness",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        return {
            "verified": False,
            "reason": "http_error",
            "http_status": exc.code,
            "authenticated": bool(token),
            "message": f"jobs endpoint returned HTTP {exc.code}",
        }
    except (TimeoutError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", None)
        return {
            "verified": False,
            "reason": "network_error" if not isinstance(reason, TimeoutError) else "timeout",
            "message": f"jobs endpoint unreachable: {reason or exc}",
        }

    observed_run_id = data.get("run_id")
    observed_attempt = data.get("run_attempt")
    verified = observed_run_id == expected_run_id and observed_attempt == expected_attempt
    result = {
        "verified": verified,
        "expected_run_id": expected_run_id,
        "expected_attempt": expected_attempt,
        "observed_run_id": observed_run_id,
        "observed_attempt": observed_attempt,
    }
    if not verified:
        result["reason"] = "run_attempt_mismatch"
        result["message"] = (
            f"job {job_id} belongs to run {observed_run_id} attempt {observed_attempt}, "
            f"not the claimed run {expected_run_id} attempt {expected_attempt}"
        )
    return result


def fetch_job_log(
    repo: str,
    run_id: int,
    attempt: int,
    job_id: int,
    cache_dir: str | Path = "cache",
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    token: str | None = None,
    token_source: str | None = None,
    verify_identity: bool = True,
) -> dict:
    """Fetch one job log (read-only), or report why it could not be fetched.

    token / token_source are optional overrides; if absent, the function
    looks up DEMO_A_GITHUB_TOKEN / GITHUB_TOKEN / GH_TOKEN itself.

    ``verify_identity=True`` (default) additionally cross-checks the job's
    run_id / run_attempt via the jobs endpoint before writing the cache. When
    verification fails (or is unavailable), the log is still cached but the
    meta records ``identity_verified=False`` and the reason.

    Returns a dict with at least: status, message. On success also:
    path, sha256, line_count, fetched_at, http_status, identity_verified,
    identity_check.
    """
    cache_dir = Path(cache_dir)
    log_path, meta_path = _cache_paths(cache_dir, repo, run_id, attempt, job_id)
    cached = _load_cache(meta_path, log_path)
    if cached is not None:
        return cached

    url = f"https://api.github.com/repos/{repo}/actions/jobs/{job_id}/logs"
    if token is None:
        token, token_source = _token_from_env()
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "demo_a_foundation-harness",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    identity_check = (
        _verify_job_identity(repo, job_id, run_id, attempt, token, timeout)
        if verify_identity
        else {"verified": None, "reason": "skipped", "message": "identity verification disabled"}
    )
    identity_verified = identity_check.get("verified") is True

    attempts = retries + 1
    last_error: dict | None = None
    for round_no in range(1, attempts + 1):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                status_code = resp.status
            if status_code not in (200, 302):
                # urlopen follows redirects; reaching here with 302 is unexpected.
                last_error = {
                    "status": "not_found" if status_code == 404 else "permission_denied",
                    "http_status": status_code,
                    "message": f"unexpected HTTP status {status_code}",
                }
                break
            cache_dir.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(body)
            meta = {
                "status": "fetched",
                "cached": False,
                "repo": repo,
                "run_id": run_id,
                "run_attempt": attempt,
                "job_id": job_id,
                "url": url,
                "http_status": status_code,
                "authenticated": bool(token),
                "token_source": token_source,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sha256": hashlib.sha256(body).hexdigest(),
                "line_count": len(body.decode("utf-8", errors="replace").splitlines()),
                "path": str(log_path),
                "import_source": "fetch",
                "identity_verified": identity_verified,
                "identity_check": identity_check,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return meta
        except urllib.error.HTTPError as exc:
            body_text = None
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body_text = None
            status_kind, message = _classify_http_error(
                exc.code, bool(token), body_text, getattr(exc, "headers", None)
            )
            last_error = {
                "status": status_kind,
                "http_status": exc.code,
                "message": message,
                "authenticated": bool(token),
                "token_source": token_source,
            }
            break  # HTTP errors are deterministic; retrying will not help.
        except TimeoutError:
            last_error = {"status": "timeout", "message": f"request timed out after {timeout}s"}
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError):
                last_error = {"status": "timeout", "message": f"request timed out after {timeout}s"}
            else:
                last_error = {
                    "status": "network_error",
                    "message": f"network error: {reason or exc}",
                }
        if round_no < attempts:
            time.sleep(1.0)

    result = dict(last_error or {"status": "network_error", "message": "unknown failure"})
    result.update(
        {
            "repo": repo,
            "run_id": run_id,
            "run_attempt": attempt,
            "job_id": job_id,
            "url": url,
            "authenticated": bool(token),
            "token_source": token_source,
            "cached": False,
        }
    )
    return result


def manual_import(
    repo: str,
    run_id: int,
    attempt: int,
    job_id: int,
    raw_text: str | bytes,
    cache_dir: str | Path = "cache",
    source_label: str = "manual_import",
    extra: dict | None = None,
) -> dict:
    """Manually import an already-fetched log into the cache as a backup path.

    Used when fetch_job_log cannot reach GitHub but the caller has obtained
    the log through some other means. The imported file is treated the same
    as a fetched one: cache meta records repo/run/attempt/job, sha256,
    line_count and import_source="manual_import".

    A manual import carries only the caller's claim about which run/attempt the
    log came from, so it is ALWAYS registered with ``identity_verified=False``
    and an explicit ``identity_check`` stating that the sha256 proves content
    consistency only, not provenance.

    `source_label` lets the caller record how the log was obtained
    (e.g. "manual_import:browser_save").
    """
    if isinstance(raw_text, str):
        raw_bytes = raw_text.encode("utf-8")
    else:
        raw_bytes = raw_text
    cache_dir = Path(cache_dir)
    log_path, meta_path = _cache_paths(cache_dir, repo, run_id, attempt, job_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(raw_bytes)
    sha = hashlib.sha256(raw_bytes).hexdigest()
    line_count = len(raw_bytes.decode("utf-8", errors="replace").splitlines())
    meta = {
        "status": "fetched",
        "cached": False,
        "repo": repo,
        "run_id": run_id,
        "run_attempt": attempt,
        "job_id": job_id,
        "url": None,
        "http_status": None,
        "authenticated": None,
        "token_source": None,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": sha,
        "line_count": line_count,
        "path": str(log_path),
        "import_source": "manual_import",
        "source_label": source_label,
        "extra": extra or {},
        "identity_verified": False,
        "identity_check": {
            "verified": False,
            "reason": "manual_import_unverified",
            "message": "sha256 proves content consistency only; provenance is the caller's unverified claim",
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def import_collector_artifact(
    artifact_dir: str | Path, provenance_path: str | Path, repo: str,
    run_id: int, attempt: int, job_id: int, collector_run_id: int,
    head_sha: str, workflow_name: str, job_name: str, conclusion: str,
    cache_dir: str | Path = "cache",
) -> dict:
    """Validate a collector artifact before writing normal verified cache meta."""
    artifact_dir, provenance_path = Path(artifact_dir), Path(provenance_path)
    files = {name: artifact_dir / name for name in (
        "raw_job.log", "run_metadata.json", "job_metadata.json",
        "identity_gate.json", "SHA256SUMS.txt")}
    files["provenance"] = provenance_path
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise HarnessError("collector_artifact_invalid", "collector artifact is incomplete",
                           {"reason": "missing_required_file", "missing": missing})
    try:
        run = json.loads(files["run_metadata.json"].read_text(encoding="utf-8"))
        job = json.loads(files["job_metadata.json"].read_text(encoding="utf-8"))
        gate = json.loads(files["identity_gate.json"].read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HarnessError("collector_artifact_invalid", "collector metadata is not valid JSON",
                           {"reason": "invalid_json", "detail": str(exc)}) from exc
    raw = files["raw_job.log"].read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    sums = files["SHA256SUMS.txt"].read_text(encoding="utf-8").splitlines()
    sum_sha = next((line.split()[0] for line in sums
                    if len(line.split()) >= 2 and line.split()[-1] == "raw_job.log"), None)
    source, collector = provenance.get("source") or {}, provenance.get("collector") or {}
    artifact = provenance.get("artifact") or {}
    observed = {
        "repository": (run.get("repository") or {}).get("full_name"),
        "run_id": run.get("id"), "run_attempt": run.get("run_attempt"),
        "job_id": job.get("id"), "job_run_id": job.get("run_id"),
        "job_run_attempt": job.get("run_attempt"), "head_sha": run.get("head_sha"),
        "job_head_sha": job.get("head_sha"), "workflow_name": run.get("name"),
        "job_workflow_name": job.get("workflow_name"), "job_name": job.get("name"),
        "conclusion": job.get("conclusion"), "collector_run_id": collector.get("run_id"),
        "artifact_directory": artifact.get("directory"), "log_sha256": actual_sha,
    }
    checks = {
        "repository": observed["repository"] == repo == gate.get("repository"),
        "run_id": observed["run_id"] == observed["job_run_id"] == source.get("run_id") == gate.get("run_id") == run_id,
        "run_attempt": observed["run_attempt"] == observed["job_run_attempt"] == source.get("run_attempt") == gate.get("run_attempt") == attempt,
        "job_id": observed["job_id"] == source.get("job_id") == gate.get("job_id") == job_id,
        "head_sha": observed["head_sha"] == observed["job_head_sha"] == source.get("head_sha") == gate.get("head_sha") == head_sha,
        "workflow_name": observed["workflow_name"] == observed["job_workflow_name"] == gate.get("workflow_name") == workflow_name,
        "job_name": observed["job_name"] == source.get("job_name") == gate.get("job_name") == job_name,
        "conclusion": observed["conclusion"] == source.get("conclusion") == gate.get("conclusion") == conclusion,
        "collector_run_id": observed["collector_run_id"] == collector_run_id,
        "artifact_directory": observed["artifact_directory"] == artifact_dir.name,
        "digest": actual_sha == sum_sha == gate.get("log_sha256") == artifact.get("raw_job_log_sha256"),
        "gate": gate.get("identity_verified") is True and all((gate.get("checks") or {}).values()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise HarnessError("collector_artifact_invalid", "collector artifact validation failed",
                           {"reason": "digest_conflict" if "digest" in failed else "identity_conflict",
                            "failed_checks": failed, "observed": observed})
    cache_dir = Path(cache_dir)
    log_path, meta_path = _cache_paths(cache_dir, repo, run_id, attempt, job_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(raw)
    meta = {
        "status": "fetched", "cached": False, "repo": repo, "run_id": run_id,
        "run_attempt": attempt, "job_id": job_id, "job_name": job_name,
        "head_sha": head_sha, "workflow_name": workflow_name, "conclusion": conclusion,
        "url": None, "http_status": None, "authenticated": True, "token_source": None,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": actual_sha,
        "line_count": len(raw.decode("utf-8", errors="replace").splitlines()),
        "path": str(log_path), "import_source": "verified_collector_artifact",
        "collector_run_id": collector_run_id, "collector_artifact_id": collector.get("artifact_id"),
        "collector_artifact_name": collector.get("artifact_name"),
        "artifact_provenance_path": str(provenance_path.resolve()), "identity_verified": True,
        "identity_check": {"verified": True, "reason": "collector_artifact_closure",
                           "expected_run_id": run_id, "expected_attempt": attempt,
                           "observed_run_id": observed["run_id"],
                           "observed_attempt": observed["run_attempt"], "checks": checks},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")
    return meta


def identity_status(meta: dict) -> str:
    """Classify a cached log's identity as VERIFIED / MISMATCH / UNVERIFIED.

    VERIFIED   — the cache meta records a successful jobs-endpoint cross-check
                 (identity_verified=True).
    MISMATCH   — the cross-check RAN and positively contradicted the claimed
                 run/attempt (reason=run_attempt_mismatch): a confirmed
                 wrong association, not merely an unknown one.
    UNVERIFIED — everything else: manual import, verification skipped or
                 failed for network/HTTP reasons, or an unknown meta shape.

    The distinction matters: MISMATCH material must never be promoted to
    formal evidence, while UNVERIFIED material may stay cached but also must
    not enter the diagnostic context until it is verified.
    """
    check = meta.get("identity_check") or {}
    if meta.get("identity_verified") is True:
        return "VERIFIED"
    if check.get("verified") is False and check.get("reason") == "run_attempt_mismatch":
        return "MISMATCH"
    return "UNVERIFIED"


def register_log_material(
    repo: str,
    run_id: int,
    attempt: int,
    job_id: int,
    cache_dir: str | Path = "cache",
    material_id: str = "job_log",
    summary: str = "Complete raw CI job log (fetched read-only, identity verified)",
) -> dict:
    """Turn a cached job log into a case-manifest material entry.

    This is the single gate between the log cache and formal diagnostic use.
    The cache key already binds repo/run/attempt/job (see _cache_paths), and
    _load_cache re-verifies the sha256 before the meta is trusted. Only a
    VERIFIED cache may be registered:

      VERIFIED    -> returns a manifest-ready material entry carrying
                     identity_verified=True / identity_status="VERIFIED".
      MISMATCH    -> raises HarnessError("identity_mismatch"): the log is
                     positively known to belong to a different run/attempt.
      UNVERIFIED  -> raises HarnessError("identity_unverified"): manual
                     import, verification skipped/failed, or no intact cache
                     at all — a local file merely existing is NOT proof of
                     provenance and is never treated as verified material.
    """
    cache_dir = Path(cache_dir)
    log_path, meta_path = _cache_paths(cache_dir, repo, run_id, attempt, job_id)
    meta = _load_cache(meta_path, log_path)
    if meta is None:
        raise HarnessError(
            "identity_unverified",
            f"no intact cached log for {repo} run {run_id} attempt {attempt} job {job_id}; "
            "a local file alone does not count as verified material",
            {"repo": repo, "run_id": run_id, "attempt": attempt, "job_id": job_id},
        )
    status = identity_status(meta)
    if status == "MISMATCH":
        check = meta.get("identity_check") or {}
        raise HarnessError(
            "identity_mismatch",
            (
                f"cached log for job {job_id} is positively known to belong to "
                f"run {check.get('observed_run_id')} attempt {check.get('observed_attempt')}, "
                f"NOT the claimed run {run_id} attempt {attempt}; it must not be "
                f"registered, retrieved, or diagnosed"
            ),
            {
                "repo": repo,
                "run_id": run_id,
                "attempt": attempt,
                "job_id": job_id,
                "observed_run_id": check.get("observed_run_id"),
                "observed_attempt": check.get("observed_attempt"),
                "reason": check.get("reason"),
            },
        )
    if status != "VERIFIED":
        check = meta.get("identity_check") or {}
        raise HarnessError(
            "identity_unverified",
            (
                f"cached log for {repo} run {run_id} attempt {attempt} job {job_id} "
                f"has no successful identity cross-check "
                f"(reason={check.get('reason', meta.get('identity_check', {}).get('reason'))}); "
                f"it stays cached but cannot enter the formal diagnostic context"
            ),
            {
                "repo": repo,
                "run_id": run_id,
                "attempt": attempt,
                "job_id": job_id,
                "reason": (meta.get("identity_check") or {}).get("reason"),
            },
        )
    return {
        "material_id": material_id,
        "kind": "ci_job_log",
        "path": str(log_path.resolve()),
        "sha256": meta["sha256"],
        "line_numbering": "source",
        "summary": summary,
        "provenance": (
            f"fetch_job_log cache for {repo} run {run_id} attempt {attempt} job {job_id}; "
            "identity cross-checked against the jobs endpoint"
        ),
        "limitations": [],
        "identity_verified": True,
        "identity_status": "VERIFIED",
        "identity_check": {
            "verified": True,
            "expected_run_id": run_id,
            "expected_attempt": attempt,
            "observed_run_id": (meta.get("identity_check") or {}).get("observed_run_id"),
            "observed_attempt": (meta.get("identity_check") or {}).get("observed_attempt"),
        },
        # Verifiable binding between this manifest entry and the cache meta it
        # was derived from. load_case RE-CHECKS this binding against the live
        # cache, so a hand-edited manifest cannot claim VERIFIED without the
        # cache actually backing it (the meta digest pins the recorded
        # verification result; the log digest pins the content it applies to).
        "identity_binding": {
            "repo": repo,
            "run_id": run_id,
            "run_attempt": attempt,
            "job_id": job_id,
            "cache_dir": str(cache_dir.resolve()),
            "log_sha256": meta["sha256"],
            "meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        },
    }


def _strip_timestamp(line: str) -> str:
    match = _TIMESTAMP_PREFIX.match(line)
    return line[match.end():] if match else line


def match_excerpt(excerpt_lines: list[str], source_lines: list[str]) -> dict:
    """Locate an excerpt inside a complete source log.

    GitHub log lines carry a timestamp prefix; matching compares the content
    after that prefix. Returns one of:
      mapped     — exactly one consecutive match; source_lines gives the range
      ambiguous  — multiple matches; no position is chosen arbitrarily
      unmatched  — the excerpt does not appear in the source
    """
    if not excerpt_lines:
        raise HarnessError("invalid_query", "excerpt_lines must not be empty")
    stripped_source = [_strip_timestamp(line) for line in source_lines]
    targets = [line.rstrip() for line in excerpt_lines]
    n = len(targets)

    matches: list[dict] = []
    for start in range(0, len(stripped_source) - n + 1):
        window = [line.rstrip() for line in stripped_source[start : start + n]]
        if window == targets:
            matches.append({"start": start + 1, "end": start + n})

    if not matches:
        return {"status": "unmatched", "matches": []}
    if len(matches) == 1:
        return {"status": "mapped", "matches": matches}
    return {"status": "ambiguous", "matches": matches}

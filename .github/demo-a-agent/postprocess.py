#!/usr/bin/env python3
"""Demo A agent postprocess -- analyze (Job A) and publish (Job B).

Subcommands:
  analyze  -- gates + agent diagnosis; writes analysis.json (always exit 0)
  publish  -- deduplicated Issue publication; writes publication.json

Security model (P2 contract, enforced here and in the workflow YAML):
- Reacts only to workflow_run events of "Demo A Controlled CI" on
  Xiu11111/rustsbi; every event payload value is treated as a CLAIM and is
  re-checked against the GitHub API before use.
- Never checks out / executes / restores anything from the source run:
  no checkout of workflow_run.head_sha (only github.sha), no execution of
  source artifacts, no restore of source caches.
- The frozen agent bundle is sha256-verified against bundle_manifest.json
  (BUNDLE_SHA_MATCH) before any bundle module is imported.
- CI job log content is untrusted data: it is only searched for one fixed
  scenario substring and embedded into JSON documents / Issue bodies through
  the GitHub API. It is never interpolated into a shell command.
- Secret values are never logged, stored, or measured beyond a boolean
  non-empty check. "Secret name exists" (MODEL_SECRET_NONEMPTY gate) and
  "provider call works" (model_secret_value_validity, first real provider
  call) are two separate, independent gates.
- Job A holds the model secret but has no issues:write; Job B holds
  issues:write but never receives the model secret and never imports the
  frozen agent bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent

REPO = "Xiu11111/rustsbi"
SOURCE_WORKFLOW_NAME = "Demo A Controlled CI"
SOURCE_WORKFLOW_PATH = ".github/workflows/demo-a-ci.yml"
CONTROLLED_JOB_NAME = "controlled-git-ref-check"
SCENARIO_LABEL = "git-ref-failure"
SCENARIO_SUBSTRING = "DEMO_A_SCENARIO=git-ref-failure"
FAILURE_TYPE = "git_ref_not_found"
API_BASE = "https://api.github.com"

STABLE_MARKER_TEMPLATE = (
    "<!-- demo-a-key: repo={repo}; workflow={workflow}; job={job}; "
    "scenario={scenario}; failure_type={failure_type} -->"
)
RUN_EVIDENCE_MARKER_TEMPLATE = (
    "<!-- demo-a-run-evidence: run_id={run_id}; run_attempt={run_attempt} -->"
)

ISSUE_TITLE = "[Demo A] Controlled failure diagnosis: git-ref-failure"

GATE_ORDER = [
    "REPOSITORY_MATCH",
    "BUNDLE_SHA_MATCH",
    "WORKFLOW_NAME_MATCH",
    "CONCLUSION_RECHECK",
    "WORKFLOW_PATH_MATCH",
    "RUN_ID_MATCH",
    "RUN_ATTEMPT_MATCH",
    "UNIQUE_JOB_MATCH",
    "JOB_ID_MATCH",
    "HEAD_SHA_MATCH",
    "LOG_DOWNLOADED",
    "LOG_SHA256_COMPUTED",
    "IDENTITY_BINDING_VERIFIED",
    "CONTROLLED_SCENARIO_MATCH",
    "CASE_STRICT_LOAD",
    "MODEL_SECRET_NONEMPTY",
]

ANALYSIS_SCHEMA = "demo_a_agent_postprocess/analysis/1"
PUBLICATION_SCHEMA = "demo_a_agent_postprocess/publication/1"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_issue_marker() -> str:
    return STABLE_MARKER_TEMPLATE.format(
        repo=REPO,
        workflow=SOURCE_WORKFLOW_NAME,
        job=CONTROLLED_JOB_NAME,
        scenario=SCENARIO_LABEL,
        failure_type=FAILURE_TYPE,
    )


def run_evidence_marker(run_id: int, run_attempt: int) -> str:
    return RUN_EVIDENCE_MARKER_TEMPLATE.format(run_id=int(run_id), run_attempt=int(run_attempt))


# ---------------------------------------------------------------------------
# Injectable seams (kept as module functions so offline tests can stub them
# without touching the frozen bundle modules).
# ---------------------------------------------------------------------------


def fetch_log(run_id: int, attempt: int, job_id: int, cache_dir: Path, token: str | None) -> dict:
    from sources import fetch_job_log

    return fetch_job_log(
        repo=REPO,
        run_id=run_id,
        attempt=attempt,
        job_id=job_id,
        cache_dir=str(cache_dir),
        token=token,
        token_source="env:GITHUB_TOKEN",
        verify_identity=True,
    )


def register_entry(run_id: int, attempt: int, job_id: int, cache_dir: Path) -> dict:
    from sources import register_log_material

    return register_log_material(
        repo=REPO,
        run_id=run_id,
        attempt=attempt,
        job_id=job_id,
        cache_dir=str(cache_dir),
        material_id="ci_job_log",
        summary="Complete raw controlled CI job log",
    )


def load_case_strict(manifest_path: Path):
    from materials import load_case

    return load_case(manifest_path, config={}, strict=True)


def build_provider(runtime, secrets):
    from provider import LLMProvider

    return LLMProvider(runtime=runtime, secrets=secrets)


def run_agent_case(case, runtime, provider):
    from agent import run_agent, run_result_to_dict

    result = run_agent(case, runtime=runtime, provider=provider)
    return result, run_result_to_dict(result)


# ---------------------------------------------------------------------------
# GitHub REST client (stdlib only). Values from responses are treated as
# data; nothing from them ever reaches a shell.
# ---------------------------------------------------------------------------


class ApiError(Exception):
    def __init__(self, kind: str, message: str, http_status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.http_status = http_status


class GitHubClient:
    def __init__(self, token: str | None = None, timeout: float = 30.0, base_url: str = API_BASE):
        self.token = token
        self.timeout = timeout
        self.base = base_url.rstrip("/")

    def request(self, method: str, path: str, body: dict | None = None):
        url = f"{self.base}{path}"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "demo-a-agent-postprocess",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            raise ApiError("http_error", f"github api http error {exc.code}", http_status=exc.code) from exc
        except TimeoutError as exc:
            raise ApiError("timeout", f"github api request timed out after {self.timeout}s") from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(reason, TimeoutError):
                raise ApiError("timeout", "github api request timed out") from exc
            raise ApiError("network_error", f"github api network error: {reason or exc}") from exc
        parsed = None
        if raw:
            try:
                parsed = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                parsed = raw.decode("utf-8", errors="replace")
        return status, parsed

    def get(self, path: str):
        return self.request("GET", path)[1]

    def post(self, path: str, body: dict):
        return self.request("POST", path, body)[1]

    def get_run(self, run_id: int) -> dict:
        return self.get(f"/repos/{REPO}/actions/runs/{int(run_id)}")

    def _paginate(self, first_path: str, max_pages: int) -> list:
        items: list = []
        path = first_path
        for _ in range(max_pages):
            page = self.get(path)
            if not isinstance(page, list):
                break
            items.extend(page)
            if len(page) < 100:
                break
            separator = "&" if "?" in path else "?"
            path = f"{path}{separator}page={len(items) // 100 + 1}"
        return items

    def list_jobs(self, run_id: int, run_attempt: int, max_pages: int = 10) -> list:
        path = f"/repos/{REPO}/actions/runs/{int(run_id)}/attempts/{int(run_attempt)}/jobs?per_page=100"
        jobs = self._paginate(path, max_pages)
        return [j for j in jobs if isinstance(j, dict)]

    def list_issues(self, state: str = "all", max_pages: int = 20) -> list:
        path = f"/repos/{REPO}/issues?state={state}&per_page=100"
        issues = self._paginate(path, max_pages)
        # pull requests are returned by the issues endpoint as well; the
        # publication must only ever consider real issues.
        return [i for i in issues if isinstance(i, dict) and "pull_request" not in i]

    def create_issue(self, title: str, body: str) -> dict:
        return self.post(f"/repos/{REPO}/issues", {"title": title, "body": body})

    def get_issue(self, number: int) -> dict:
        return self.get(f"/repos/{REPO}/issues/{int(number)}")

    def add_comment(self, number: int, body: str) -> dict:
        return self.post(f"/repos/{REPO}/issues/{int(number)}/comments", {"body": body})

    def list_issue_comments(self, number: int, max_pages: int = 20) -> list:
        path = f"/repos/{REPO}/issues/{int(number)}/comments?per_page=100"
        comments = self._paginate(path, max_pages)
        return [c for c in comments if isinstance(c, dict)]


def find_issue_with_marker(client: GitHubClient, marker: str) -> dict | None:
    """Paginated open+closed search; exact substring match on the full
    stable marker; pull-request objects already filtered by the client."""
    for issue in client.list_issues(state="all"):
        body = issue.get("body") or ""
        if marker in body:
            return issue
    return None


def run_evidence_present(client: GitHubClient, issue: dict, ev_marker: str) -> bool:
    if ev_marker in (issue.get("body") or ""):
        return True
    for comment in client.list_issue_comments(issue.get("number", 0)):
        if ev_marker in (comment.get("body") or ""):
            return True
    return False


# ---------------------------------------------------------------------------
# Gate engine
# ---------------------------------------------------------------------------


class GateEngine:
    def __init__(self):
        self.gates: dict[str, dict] = {
            name: {"status": "SKIPPED", "detail": ""} for name in GATE_ORDER
        }

    def passed(self, name: str, detail: str) -> None:
        self.gates[name] = {"status": "PASS", "detail": detail}

    def failed(self, name: str, detail: str) -> None:
        self.gates[name] = {"status": "FAIL", "detail": detail}

    def all_passed_through(self, name: str) -> bool:
        idx = GATE_ORDER.index(name)
        return all(self.gates[g]["status"] == "PASS" for g in GATE_ORDER[: idx + 1])


def check_bundle_shas(bundle_dir: Path) -> tuple[bool, str]:
    """Verify every frozen module against bundle_manifest.json."""
    manifest_path = bundle_dir / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"bundle_manifest.json unreadable: {type(exc).__name__}"
    mismatches = []
    checked = 0
    for entry in manifest.get("files", []):
        rel = entry.get("relative_path", "")
        expected = entry.get("bundled_sha256", "")
        file_path = bundle_dir / rel
        try:
            actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        except OSError:
            mismatches.append(f"{rel}: missing")
            continue
        checked += 1
        if actual != expected:
            mismatches.append(f"{rel}: sha256 mismatch")
    if mismatches:
        return False, "; ".join(mismatches)
    return True, f"{checked}/{checked} frozen module sha256 match bundle_manifest.json"


def classify_provider_failure(stop_reason: str) -> tuple[str, int | None]:
    """Classify a provider_error stop_reason. run_agent captures the
    ProviderError into a string, losing details, so the HTTP status is
    recovered from the message text ('chat completions returned HTTP NNN')."""
    text = stop_reason or ""
    match = re.search(r"HTTP (\d{3})", text)
    if match:
        return "http", int(match.group(1))
    if "timed out" in text:
        return "timeout", None
    if "request failed" in text:
        return "network", None
    return "unknown", None


def _harness_failure_result(exc) -> str:
    code = getattr(exc, "error_type", "")
    if code == "identity_mismatch":
        return "IDENTITY_MISMATCH"
    if code == "identity_unverified":
        return "IDENTITY_UNVERIFIED"
    return "EVIDENCE_INSUFFICIENT"


# ---------------------------------------------------------------------------
# Derived runtime case (shape authority: the formal case manifest template)
# ---------------------------------------------------------------------------


def _plain_material(material_id: str, kind: str, path: Path, summary: str, provenance: str, limitations: list) -> dict:
    data = path.read_bytes()
    return {
        "material_id": material_id,
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "line_numbering": "source",
        "summary": summary,
        "provenance": provenance,
        "limitations": limitations,
    }


def build_derived_case(
    run: dict,
    job: dict,
    log_entry: dict,
    run_id: int,
    run_attempt: int,
    job_id: int,
    head_sha: str,
    cache_dir: Path,
    postprocess_run_id: int,
) -> tuple[Path, dict]:
    """Write the derived runtime case next to this run's log cache and
    return (manifest_path, case_dict). The ci_job_log material entry is the
    exact dict returned by register_log_material (identity-bound)."""
    case_dir = cache_dir / f"run_{run_id}_attempt_{run_attempt}" / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    run_meta_path = case_dir / "run_metadata.json"
    run_meta_path.write_text(json.dumps(run, indent=2), encoding="utf-8")

    job_meta_path = case_dir / "job_metadata.json"
    job_meta_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    repo_root = Path(os.environ.get("GITHUB_WORKSPACE") or Path.cwd())
    workflow_src = repo_root / SOURCE_WORKFLOW_PATH
    workflow_dst = case_dir / "workflow_demo_a_ci.yml"
    workflow_dst.write_bytes(workflow_src.read_bytes())

    postprocess_run_url = f"https://github.com/{REPO}/actions/runs/{postprocess_run_id}"

    case = {
        "schema": "demo_a_foundation/case-manifest/1",
        "case_id": f"demo_a_online_run_{run_id}_attempt_{run_attempt}",
        "title": (
            f"Derived runtime case: controlled {SCENARIO_LABEL} in Demo A online "
            f"test repository (run {run_id} attempt {run_attempt})"
        ),
        "source": {
            "repository": REPO,
            "commit": head_sha,
            "workflow_path": SOURCE_WORKFLOW_PATH,
            "run_id": run_id,
            "run_attempt": run_attempt,
            "event": run.get("event"),
            "run_conclusion": run.get("conclusion"),
            "job_id": job_id,
            "job_name": CONTROLLED_JOB_NAME,
            "collector_run_id": postprocess_run_id,
        },
        "materials": [
            log_entry,
            _plain_material(
                "run_metadata",
                "run_metadata_json",
                run_meta_path,
                "GitHub Actions source run metadata (re-checked via API during postprocess)",
                f"GitHub API run object re-checked during postprocess run {postprocess_run_id} ({postprocess_run_url})",
                [],
            ),
            _plain_material(
                "job_metadata",
                "job_metadata_json",
                job_meta_path,
                "GitHub Actions source job metadata (jobs endpoint, attempts path)",
                f"GitHub API job object re-checked during postprocess run {postprocess_run_id} ({postprocess_run_url})",
                [],
            ),
            _plain_material(
                "workflow",
                "workflow_yaml",
                workflow_dst,
                "Workflow definition of the controlled CI (postprocess checkout copy)",
                (
                    f"copied from the postprocess checkout at github.sha "
                    f"(postprocess run {postprocess_run_id}), NOT from the source run head "
                    f"commit; demo-a-ci.yml is unchanged in this cycle, so the contents "
                    f"are expected to be identical to the source run's version"
                ),
                [
                    "workflow file comes from the postprocess ref (github.sha), not from "
                    "the source head commit; the source run's head_sha is recorded in "
                    "source.commit and run_metadata"
                ],
            ),
        ],
    }

    manifest_path = case_dir / "case_manifest.json"
    manifest_path.write_text(json.dumps(case, indent=2), encoding="utf-8")
    return manifest_path, case


# ---------------------------------------------------------------------------
# analyze (Job A)
# ---------------------------------------------------------------------------


def _int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _sha_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", value or ""))


def cmd_analyze(args) -> int:
    out_dir = Path(args.output)
    cache_dir = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    gates = GateEngine()
    result = "GATE_REJECTED"
    should_publish = False
    validity = "NOT_CALLED"
    provider_calls_total = 0
    provider_attempts: list[dict] = []
    agent_payload: dict | None = None
    issue_spec: dict | None = None
    case_info: dict | None = None
    evidence: dict = {}

    postprocess_run_id = _int_env("GITHUB_RUN_ID")
    postprocess_run_attempt = _int_env("GITHUB_RUN_ATTEMPT", 1)
    postprocess_sha = os.environ.get("GITHUB_SHA", "")

    claimed = {
        "repository": os.environ.get("CLAIMED_SOURCE_REPOSITORY", ""),
        "workflow_name": os.environ.get("CLAIMED_SOURCE_WORKFLOW_NAME", ""),
        "run_id": _int_env("CLAIMED_SOURCE_RUN_ID"),
        "run_attempt": _int_env("CLAIMED_SOURCE_RUN_ATTEMPT", 1),
        "head_sha": os.environ.get("CLAIMED_SOURCE_HEAD_SHA", ""),
        "conclusion": os.environ.get("CLAIMED_SOURCE_CONCLUSION", ""),
    }

    source_conclusion_handled = False
    try:
        # Explicit source-conclusion routing happens BEFORE any gate: the
        # postprocess only ever works on failure runs of the controlled CI.
        if claimed["conclusion"] == "success":
            result = "NO_ACTION_SUCCESS"
            source_conclusion_handled = True
        elif claimed["conclusion"] != "failure":
            result = "NO_ACTION_UNHANDLED_CONCLUSION"
            source_conclusion_handled = True

        if not source_conclusion_handled:
            run_id = claimed["run_id"]
            run_attempt = claimed["run_attempt"]

            # G1 REPOSITORY_MATCH
            gh_repo = os.environ.get("GITHUB_REPOSITORY", "")
            if gh_repo == REPO and claimed["repository"] == REPO:
                gates.passed("REPOSITORY_MATCH", f"github.repository={gh_repo}; event claim matches")
            else:
                gates.failed(
                    "REPOSITORY_MATCH",
                    f"github.repository={gh_repo!r}; claimed={claimed['repository']!r}; expected={REPO!r}",
                )
                result = "GATE_REJECTED"
                raise _GateStop()

            # G2 BUNDLE_SHA_MATCH
            bundle_ok, bundle_detail = check_bundle_shas(BUNDLE_DIR)
            if bundle_ok:
                gates.passed("BUNDLE_SHA_MATCH", bundle_detail)
            else:
                gates.failed("BUNDLE_SHA_MATCH", bundle_detail)
                result = "GATE_REJECTED"
                raise _GateStop()

            # G3 WORKFLOW_NAME_MATCH
            if claimed["workflow_name"] == SOURCE_WORKFLOW_NAME:
                gates.passed("WORKFLOW_NAME_MATCH", f"claimed workflow name matches {SOURCE_WORKFLOW_NAME!r}")
            else:
                gates.failed(
                    "WORKFLOW_NAME_MATCH",
                    f"claimed={claimed['workflow_name']!r}; expected={SOURCE_WORKFLOW_NAME!r}",
                )
                result = "GATE_REJECTED"
                raise _GateStop()

            client = GitHubClient(token=os.environ.get("GITHUB_TOKEN"))

            # G4 CONCLUSION_RECHECK (API re-check of the source run)
            try:
                run = client.get_run(run_id)
            except ApiError as exc:
                gates.failed("CONCLUSION_RECHECK", f"github api error ({exc.kind}) re-checking run {run_id}")
                result = "GATE_REJECTED"
                raise _GateStop()
            api_conclusion = run.get("conclusion") or ""
            if api_conclusion == "failure":
                gates.passed("CONCLUSION_RECHECK", f"api run {run_id} conclusion='failure' (re-checked)")
            else:
                gates.failed("CONCLUSION_RECHECK", f"api run {run_id} conclusion={api_conclusion!r}, expected 'failure'")
                result = "GATE_REJECTED"
                raise _GateStop()

            # G5 WORKFLOW_PATH_MATCH
            api_path = run.get("path") or ""
            if api_path == SOURCE_WORKFLOW_PATH:
                gates.passed("WORKFLOW_PATH_MATCH", f"api run path={api_path}")
            else:
                gates.failed("WORKFLOW_PATH_MATCH", f"api run path={api_path!r}; expected={SOURCE_WORKFLOW_PATH!r}")
                result = "GATE_REJECTED"
                raise _GateStop()

            # G6 RUN_ID_MATCH
            if int(run.get("id", 0) or 0) == run_id:
                gates.passed("RUN_ID_MATCH", f"api run id matches claimed run_id {run_id}")
            else:
                gates.failed("RUN_ID_MATCH", f"api run id={run.get('id')}; claimed={run_id}")
                result = "IDENTITY_MISMATCH"
                raise _GateStop()

            # G7 RUN_ATTEMPT_MATCH
            api_attempt = int(run.get("run_attempt", -1) or -1)
            if api_attempt == run_attempt:
                gates.passed("RUN_ATTEMPT_MATCH", f"api run_attempt matches claimed attempt {run_attempt}")
            else:
                gates.failed("RUN_ATTEMPT_MATCH", f"api run_attempt={api_attempt}; claimed={run_attempt}")
                result = "IDENTITY_MISMATCH"
                raise _GateStop()

            # G8 UNIQUE_JOB_MATCH (jobs endpoint is authoritative; the
            # workflow_run event carries no job field at all)
            try:
                jobs = client.list_jobs(run_id, run_attempt)
            except ApiError as exc:
                gates.failed("UNIQUE_JOB_MATCH", f"github api error ({exc.kind}) listing jobs")
                result = "GATE_REJECTED"
                raise _GateStop()
            matched = [j for j in jobs if j.get("name") == CONTROLLED_JOB_NAME]
            if len(jobs) >= 1 and len(matched) == 1:
                gates.passed(
                    "UNIQUE_JOB_MATCH",
                    f"jobs listed={len(jobs)}; jobs named {CONTROLLED_JOB_NAME!r}=1 (expected exactly 1)",
                )
            else:
                gates.failed(
                    "UNIQUE_JOB_MATCH",
                    f"jobs listed={len(jobs)}; jobs named {CONTROLLED_JOB_NAME!r}={len(matched)} (expected exactly 1)",
                )
                result = "EVIDENCE_INSUFFICIENT"
                raise _GateStop()

            # G9 JOB_ID_MATCH
            job = matched[0]
            job_id = int(job.get("id", 0) or 0)
            if job_id > 0:
                gates.passed(
                    "JOB_ID_MATCH",
                    f"authoritative job_id={job_id} taken from the jobs API "
                    "(event.workflow_run carries no job field; fetch_job_log re-cross-checks "
                    "the job<->run binding via the jobs endpoint)",
                )
            else:
                gates.failed("JOB_ID_MATCH", f"controlled job has invalid id {job.get('id')}")
                result = "EVIDENCE_INSUFFICIENT"
                raise _GateStop()

            # G10 HEAD_SHA_MATCH (claim vs API truth; head_sha is never
            # checked out -- only recorded as evidence)
            api_head_sha = run.get("head_sha") or ""
            if _sha_hex(api_head_sha) and api_head_sha == claimed["head_sha"]:
                gates.passed("HEAD_SHA_MATCH", f"api head_sha={api_head_sha} matches claim")
            else:
                gates.failed("HEAD_SHA_MATCH", f"api head_sha={api_head_sha!r}; claimed={claimed['head_sha']!r}")
                result = "IDENTITY_MISMATCH"
                raise _GateStop()

            # G11 LOG_DOWNLOADED (identity cross-check against the jobs
            # endpoint runs inside fetch_job_log; result is never raised,
            # only reported)
            token = os.environ.get("GITHUB_TOKEN")
            fetch_result = fetch_log(run_id, run_attempt, job_id, cache_dir, token)
            fetch_status = fetch_result.get("status")
            if fetch_status in ("fetched", "cached") and fetch_result.get("path"):
                gates.passed(
                    "LOG_DOWNLOADED",
                    f"fetch status={fetch_status}; identity_check.verified="
                    f"{bool(fetch_result.get('identity_verified'))}; cached={bool(fetch_result.get('cached'))}",
                )
            else:
                gates.failed(
                    "LOG_DOWNLOADED",
                    f"fetch status={fetch_status!r}; message={fetch_result.get('message', '')[:200]}",
                )
                result = "EVIDENCE_INSUFFICIENT"
                raise _GateStop()

            # G12 LOG_SHA256_COMPUTED
            log_sha256 = fetch_result.get("sha256") or ""
            if re.fullmatch(r"[0-9a-f]{64}", log_sha256):
                gates.passed("LOG_SHA256_COMPUTED", f"log sha256={log_sha256}")
            else:
                gates.failed("LOG_SHA256_COMPUTED", "cache meta carries no valid sha256")
                result = "EVIDENCE_INSUFFICIENT"
                raise _GateStop()

            # G13 IDENTITY_BINDING_VERIFIED (single gate between the cache
            # and formal diagnostic use; refuses MISMATCH and UNVERIFIED)
            try:
                log_entry = register_entry(run_id, run_attempt, job_id, cache_dir)
            except Exception as exc:  # HarnessError from the frozen module
                gates.failed(
                    "IDENTITY_BINDING_VERIFIED",
                    f"register_log_material refused: {getattr(exc, 'error_type', type(exc).__name__)}",
                )
                result = _harness_failure_result(exc)
                raise _GateStop()
            binding = log_entry.get("identity_binding") or {}
            gates.passed(
                "IDENTITY_BINDING_VERIFIED",
                f"identity_status={log_entry.get('identity_status')}; log_sha256={binding.get('log_sha256')}",
            )

            # G14 CONTROLLED_SCENARIO_MATCH (content check on a
            # provenance-verified log; substring only, never executed)
            log_text = Path(fetch_result["path"]).read_text(encoding="utf-8", errors="replace")
            if SCENARIO_SUBSTRING in log_text:
                gates.passed("CONTROLLED_SCENARIO_MATCH", f"substring {SCENARIO_SUBSTRING!r} found in verified log")
            else:
                gates.failed("CONTROLLED_SCENARIO_MATCH", f"substring {SCENARIO_SUBSTRING!r} not found in log")
                result = "NO_CONTROLLED_SCENARIO"
                raise _GateStop()

            # G15 CASE_STRICT_LOAD (derived case generated for THIS run,
            # then load_case(strict=True) re-verifies every material,
            # including the identity binding, against the live cache)
            postprocess_run_url = f"https://github.com/{REPO}/actions/runs/{postprocess_run_id}"
            try:
                manifest_path, case_dict = build_derived_case(
                    run=run,
                    job=job,
                    log_entry=log_entry,
                    run_id=run_id,
                    run_attempt=run_attempt,
                    job_id=job_id,
                    head_sha=api_head_sha,
                    cache_dir=cache_dir,
                    postprocess_run_id=postprocess_run_id,
                )
                case = load_case_strict(manifest_path)
            except Exception as exc:
                gates.failed(
                    "CASE_STRICT_LOAD",
                    f"derived case failed strict load: {getattr(exc, 'error_type', type(exc).__name__)}",
                )
                result = _harness_failure_result(exc)
                raise _GateStop()
            gates.passed(
                "CASE_STRICT_LOAD",
                f"case {case_dict['case_id']} loaded with strict=True (4 materials verified)",
            )
            case_info = {
                "case_id": case_dict["case_id"],
                "manifest_path": str(manifest_path),
                "cache_dir": str(cache_dir),
            }

            # G16 MODEL_SECRET_NONEMPTY (name existence only; the VALUE is
            # never read into a record, and validity is only decided by the
            # first real provider call below)
            from config import load_runtime_config, load_secrets

            secrets = load_secrets(None, os.environ)
            secret_value = secrets.get("DEMO_A_LLM_API_KEY", "")
            if secret_value and secret_value.strip():
                gates.passed(
                    "MODEL_SECRET_NONEMPTY",
                    "DEMO_A_LLM_API_KEY present and non-empty (value not read or recorded)",
                )
            else:
                gates.failed("MODEL_SECRET_NONEMPTY", "DEMO_A_LLM_API_KEY missing or empty")
                result = "PROVIDER_CREDENTIAL_MISSING"
                raise _GateStop()

            # All 16 gates PASS -> and only now -> the agent runs.
            runtime = load_runtime_config(None, os.environ)
            try:
                provider = build_provider(runtime, secrets)
            except Exception as exc:
                result = "PROVIDER_CREDENTIAL_MISSING"
                validity = "NOT_CALLED"
                raise _AgentStop(f"provider construction failed: {getattr(exc, 'error_type', type(exc).__name__)}")

            final_status = None
            retried = False
            agent_error = ""
            for _attempt_no in (1, 2):
                result_obj, payload = run_agent_case(case, runtime, provider)
                calls = int((result_obj.request_sizing or {}).get("provider_calls", 0))
                provider_calls_total += calls
                provider_attempts.append(
                    {
                        "attempt": _attempt_no,
                        "status": payload["status"],
                        "stop_reason": payload["stop_reason"],
                        "provider_calls": calls,
                    }
                )
                agent_payload = payload
                final_status = payload["status"]
                if final_status != "provider_error":
                    validity = "PASS"
                    break
                kind, code = classify_provider_failure(payload["stop_reason"])
                if kind == "http" and code in (401, 403):
                    # Credential itself rejected: never blind-retry.
                    validity = "FAIL"
                    break
                if retried:
                    validity = "INCONCLUSIVE_NETWORK"
                    break
                retryable = kind in ("timeout", "network") or code == 429 or (code is not None and 500 <= code <= 599)
                if not retryable:
                    validity = "INCONCLUSIVE_NETWORK"
                    break
                retried = True

            if final_status == "completed":
                result = "AGENT_COMPLETED"
            elif final_status in ("max_steps", "no_progress"):
                result = "AGENT_INCOMPLETE"
            elif final_status == "request_budget_exceeded":
                result = "AGENT_REQUEST_BUDGET_EXCEEDED"
            else:
                result = "AGENT_PROVIDER_ERROR"
                agent_error = (agent_payload or {}).get("stop_reason", "")

            final_text = (agent_payload or {}).get("final_text", "") or ""
            should_publish = bool(final_text.strip()) and result != "AGENT_PROVIDER_ERROR"

            if should_publish:
                run_url = f"https://github.com/{REPO}/actions/runs/{run_id}/attempts/{run_attempt}"
                job_url = f"https://github.com/{REPO}/actions/runs/{run_id}/attempts/{run_attempt}/job/{job_id}"
                marker = stable_issue_marker()
                ev_marker = run_evidence_marker(run_id, run_attempt)
                generated_at = utc_now()
                body = (
                    f"{marker}\n"
                    "\n"
                    "## Demo A controlled failure diagnosis\n"
                    "\n"
                    f"- repository: {REPO}\n"
                    f"- workflow: {SOURCE_WORKFLOW_NAME} ({SOURCE_WORKFLOW_PATH})\n"
                    f"- job: {CONTROLLED_JOB_NAME}\n"
                    f"- scenario: {SCENARIO_LABEL}\n"
                    f"- failure_type: {FAILURE_TYPE}\n"
                    f"- source run: {run_url}\n"
                    f"- run_id / run_attempt: {run_id} / {run_attempt}\n"
                    f"- head_sha: `{api_head_sha}`\n"
                    f"- log_sha256: `{log_sha256}`\n"
                    f"- case_id: {case_dict['case_id']}\n"
                    f"- generated_at: {generated_at}\n"
                    "- diagnosed_by: Demo A agent postprocess (sha256-frozen bundle)\n"
                    "\n"
                    "## Agent final output\n"
                    "\n"
                    f"{final_text}\n"
                    "\n"
                    f"{ev_marker}\n"
                    f"- run_url: {run_url}\n"
                    f"- run_id: {run_id}\n"
                    f"- run_attempt: {run_attempt}\n"
                    f"- head_sha: {api_head_sha}\n"
                    f"- log_sha256: {log_sha256}\n"
                    f"- generated_at: {generated_at}\n"
                )
                issue_spec = {
                    "title": ISSUE_TITLE,
                    "body": body,
                    "stable_marker": marker,
                    "run_evidence_marker": ev_marker,
                }
                evidence = {
                    "run_url": run_url,
                    "job_url": job_url,
                    "head_sha": api_head_sha,
                    "log_sha256": log_sha256,
                    "job_id": job_id,
                    "postprocess_run_url": postprocess_run_url,
                }

    except _GateStop:
        pass
    except _AgentStop as exc:
        agent_payload = agent_payload or {"status": "provider_error", "stop_reason": str(exc)}
    except Exception as exc:  # unexpected: still record an analysis artifact
        result = "GATE_REJECTED"
        gates.failed("REPOSITORY_MATCH", gates.gates["REPOSITORY_MATCH"]["detail"])
        agent_payload = agent_payload or {
            "status": "internal_error",
            "stop_reason": f"{type(exc).__name__}: {str(exc)[:300]}",
        }

    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "generated_at": utc_now(),
        "postprocess": {
            "run_id": postprocess_run_id,
            "run_attempt": postprocess_run_attempt,
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "sha": postprocess_sha,
        },
        "source_claim": claimed,
        "gates": gates.gates,
        "result": result,
        "should_publish": should_publish,
        "issue_created": "PENDING_PUBLISH_JOB" if should_publish else "NO",
        "provider_calls": provider_calls_total,
        "model_secret_value_validity": validity,
        "provider_attempts": provider_attempts,
        "agent_result": agent_payload,
        "case": case_info,
        "issue": issue_spec,
        "evidence": evidence,
    }
    out_path = out_dir / "analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"analysis written: {out_path}; result={result}; should_publish={should_publish}")
    # analyze always exits 0 so the publish job and artifact upload still run.
    return 0


class _GateStop(Exception):
    """Internal control flow: stop the gate chain after a failure."""


class _AgentStop(Exception):
    """Internal control flow: stop after agent-phase setup failure."""


# ---------------------------------------------------------------------------
# publish (Job B)
# ---------------------------------------------------------------------------


def _issue_body_ok(issue: dict, marker: str, ev_marker: str, run_url: str) -> bool:
    body = issue.get("body") or ""
    return (
        bool(issue.get("number"))
        and bool(issue.get("html_url"))
        and marker in body
        and ev_marker in body
        and run_url in body
    )


def cmd_publish(args) -> int:
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    publication = {
        "schema": PUBLICATION_SCHEMA,
        "generated_at": utc_now(),
        "action": "none",
        "published": False,
        "verified": False,
        "new_issue_count": 0,
        "issue_number": None,
        "issue_url": None,
        "reason": "",
        "stable_marker": None,
    }

    try:
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        publication["reason"] = f"analysis artifact missing or invalid: {type(exc).__name__}"
        _write_publication(out_dir, publication)
        return 0

    publication["analysis_result"] = analysis.get("result")
    if not analysis.get("should_publish"):
        publication["reason"] = f"should_publish=false (result={analysis.get('result')})"
        _write_publication(out_dir, publication)
        return 0

    issue_spec = analysis.get("issue") or {}
    marker = issue_spec.get("stable_marker") or ""
    ev_marker = issue_spec.get("run_evidence_marker") or ""
    title = issue_spec.get("title") or ""
    body = issue_spec.get("body") or ""
    if not (marker and ev_marker and title and body):
        publication["reason"] = "issue specification incomplete in analysis artifact"
        _write_publication(out_dir, publication)
        return 1
    publication["stable_marker"] = marker

    run_url = ""
    for line in body.splitlines():
        if line.startswith("- source run: "):
            run_url = line.split("- source run: ", 1)[1].strip()
            break

    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN"))

    try:
        existing = find_issue_with_marker(client, marker)
    except ApiError as exc:
        publication["reason"] = f"issue search failed ({exc.kind})"
        _write_publication(out_dir, publication)
        return 1

    if existing is not None:
        number = int(existing.get("number", 0) or 0)
        try:
            issue_now = client.get_issue(number)
        except ApiError as exc:
            publication["reason"] = f"re-check of existing issue #{number} failed ({exc.kind})"
            _write_publication(out_dir, publication)
            return 1
        if marker not in (issue_now.get("body") or ""):
            publication["reason"] = f"existing issue #{number} no longer carries the stable marker"
            _write_publication(out_dir, publication)
            return 1
        try:
            already = run_evidence_present(client, issue_now, ev_marker)
        except ApiError as exc:
            publication["reason"] = f"comment listing for issue #{number} failed ({exc.kind})"
            _write_publication(out_dir, publication)
            return 1
        if already:
            publication.update(
                {
                    "action": "reused_noop",
                    "published": True,
                    "verified": True,
                    "new_issue_count": 0,
                    "issue_number": number,
                    "issue_url": issue_now.get("html_url"),
                    "reason": "same stable key and same run evidence already present (serialized by workflow concurrency)",
                }
            )
            _write_publication(out_dir, publication)
            return 0
        comment_body = _evidence_comment(issue_spec)
        try:
            client.add_comment(number, comment_body)
            comments = client.list_issue_comments(number)
        except ApiError as exc:
            publication["reason"] = f"evidence comment on issue #{number} failed ({exc.kind})"
            _write_publication(out_dir, publication)
            return 1
        comment_ok = any(ev_marker in (c.get("body") or "") for c in comments)
        verified = comment_ok and marker in (issue_now.get("body") or "")
        publication.update(
            {
                "action": "reused_comment",
                "published": True,
                "verified": bool(verified),
                "new_issue_count": 0,
                "issue_number": number,
                "issue_url": issue_now.get("html_url"),
                "reason": "" if verified else "post-write verification failed",
            }
        )
        _write_publication(out_dir, publication)
        return 0 if verified else 1

    try:
        created = client.create_issue(title, body)
    except ApiError as exc:
        if exc.kind in ("timeout", "network_error"):
            # The POST may have landed: re-check before any retry decision.
            try:
                existing = find_issue_with_marker(client, marker)
            except ApiError:
                existing = None
            if existing is not None:
                number = int(existing.get("number", 0) or 0)
                try:
                    issue_now = client.get_issue(number)
                except ApiError:
                    issue_now = None
                if issue_now is not None and _issue_body_ok(issue_now, marker, ev_marker, run_url):
                    publication.update(
                        {
                            "action": "created_after_recheck",
                            "published": True,
                            "verified": True,
                            "new_issue_count": 1,
                            "issue_number": number,
                            "issue_url": issue_now.get("html_url"),
                            "reason": "issue confirmed after ambiguous POST (re-check)",
                        }
                    )
                    _write_publication(out_dir, publication)
                    return 0
                # ambiguity unresolved: one retry
            try:
                created = client.create_issue(title, body)
            except ApiError as exc2:
                publication["reason"] = f"issue creation failed after re-check ({exc2.kind})"
                _write_publication(out_dir, publication)
                return 1
        else:
            publication["reason"] = f"issue creation refused (http {exc.http_status})"
            _write_publication(out_dir, publication)
            return 1

    number = int((created or {}).get("number", 0) or 0)
    if number <= 0:
        publication["reason"] = "issue creation returned no number"
        _write_publication(out_dir, publication)
        return 1
    try:
        issue_now = client.get_issue(number)
    except ApiError as exc:
        publication["reason"] = f"post-write verification GET failed ({exc.kind})"
        publication.update({"action": "created", "published": True, "new_issue_count": 1, "issue_number": number})
        _write_publication(out_dir, publication)
        return 1
    verified = _issue_body_ok(issue_now, marker, ev_marker, run_url)
    publication.update(
        {
            "action": "created",
            "published": True,
            "verified": bool(verified),
            "new_issue_count": 1,
            "issue_number": number,
            "issue_url": issue_now.get("html_url"),
            "reason": "" if verified else "post-write verification failed (number/html_url/marker/run_url)",
        }
    )
    _write_publication(out_dir, publication)
    return 0 if verified else 1


def _evidence_comment(issue_spec: dict) -> str:
    marker = issue_spec["run_evidence_marker"]
    body = issue_spec["body"]
    lines = []
    capture = False
    for line in body.splitlines():
        if line.strip() == marker:
            capture = True
            continue
        if capture:
            if line.startswith("- "):
                lines.append(line)
            elif lines:
                break
    return marker + "\n" + "\n".join(lines) + "\n"


def _write_publication(out_dir: Path, publication: dict) -> None:
    out_path = out_dir / "publication.json"
    out_path.write_text(json.dumps(publication, indent=2), encoding="utf-8")
    print(f"publication written: {out_path}; action={publication.get('action')}; published={publication.get('published')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demo A agent postprocess (analyze/publish)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="gated analysis of a controlled failure run")
    p_analyze.add_argument("--output", default="output/postprocess", help="directory for analysis.json")
    p_analyze.add_argument("--cache", default="cache/demo_a_postprocess", help="log cache directory")

    p_publish = sub.add_parser("publish", help="deduplicated issue publication")
    p_publish.add_argument("--analysis", default="output/postprocess/analysis.json", help="path to analysis.json")
    p_publish.add_argument("--output", default="output/postprocess", help="directory for publication.json")

    args = parser.parse_args(argv)
    if args.command == "analyze":
        return cmd_analyze(args)
    return cmd_publish(args)


if __name__ == "__main__":
    sys.exit(main())

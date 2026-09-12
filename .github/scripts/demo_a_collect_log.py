#!/usr/bin/env python3
"""Collect one authenticated Actions job log and fail closed on identity drift."""

from __future__ import annotations
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"
EXPECTED_JOB = "controlled-git-ref-check"
EVIDENCE_DIR = Path("online_evidence")

def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value

def api_request(path: str, token: str) -> bytes:
    request = urllib.request.Request(
        f"{API_ROOT}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "rustsbi-demo-a-log-collector",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()

def get_json(path: str, token: str) -> dict[str, Any]:
    return json.loads(api_request(path, token))

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def false_gate() -> dict[str, Any]:
    return {
        "repository": "", "run_id": 0, "run_attempt": 0, "job_id": 0,
        "job_name": "", "head_sha": "", "workflow_name": "",
        "conclusion": "", "log_sha256": "", "identity_verified": False,
        "checks": {
            "repository_match": False, "run_id_match": False,
            "run_attempt_match": False, "job_id_match": False,
            "job_name_match": False, "head_sha_match": False,
            "workflow_name_match": False, "conclusion_match": False,
        },
    }

def main() -> int:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    gate = false_gate()
    try:
        token = required_env("GH_TOKEN")
        repository = required_env("SOURCE_REPOSITORY")
        run_id = int(required_env("SOURCE_RUN_ID"))
        run_attempt = int(required_env("SOURCE_RUN_ATTEMPT"))
        event_head_sha = required_env("SOURCE_HEAD_SHA")
        event_workflow_name = required_env("SOURCE_WORKFLOW_NAME")
        event_conclusion = required_env("SOURCE_CONCLUSION")
        run = get_json(f"/repos/{repository}/actions/runs/{run_id}", token)
        jobs = get_json(
            f"/repos/{repository}/actions/runs/{run_id}/attempts/{run_attempt}/jobs?per_page=100",
            token,
        ).get("jobs", [])
        matches = [job for job in jobs if job.get("name") == EXPECTED_JOB]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {EXPECTED_JOB!r} job, found {len(matches)}")
        job = matches[0]
        checks = {
            "repository_match": (run.get("repository") or {}).get("full_name") == repository,
            "run_id_match": run.get("id") == run_id and job.get("run_id") == run_id,
            "run_attempt_match": run.get("run_attempt") == run_attempt and job.get("run_attempt") == run_attempt,
            "job_id_match": isinstance(job.get("id"), int) and job["id"] > 0,
            "job_name_match": job.get("name") == EXPECTED_JOB,
            "head_sha_match": run.get("head_sha") == event_head_sha and job.get("head_sha") == event_head_sha,
            "workflow_name_match": run.get("name") == event_workflow_name == "Demo A Controlled CI",
            "conclusion_match": run.get("conclusion") == event_conclusion == "failure",
        }
        if not all(checks.values()):
            raise RuntimeError(f"identity mismatch: checks={checks}")
        job_id = int(job["id"])
        raw_log = api_request(f"/repos/{repository}/actions/jobs/{job_id}/logs", token)
        if not raw_log:
            raise RuntimeError("downloaded job log is empty")
        digest = hashlib.sha256(raw_log).hexdigest()
        (EVIDENCE_DIR / "raw_job.log").write_bytes(raw_log)
        write_json(EVIDENCE_DIR / "run_metadata.json", run)
        write_json(EVIDENCE_DIR / "job_metadata.json", job)
        gate.update({
            "repository": repository, "run_id": run_id, "run_attempt": run_attempt,
            "job_id": job_id, "job_name": job["name"], "head_sha": event_head_sha,
            "workflow_name": run["name"], "conclusion": run["conclusion"],
            "log_sha256": digest, "identity_verified": True, "checks": checks,
        })
        write_json(EVIDENCE_DIR / "identity_gate.json", gate)
        (EVIDENCE_DIR / "SHA256SUMS.txt").write_text(
            f"{digest}  raw_job.log\n", encoding="utf-8"
        )
        print(f"identity verified for run {run_id}, attempt {run_attempt}, job {job_id}")
        return 0
    except (KeyError, ValueError, RuntimeError, urllib.error.URLError, json.JSONDecodeError) as error:
        write_json(EVIDENCE_DIR / "identity_gate.json", gate)
        write_json(EVIDENCE_DIR / "collector_error.json", {
            "error_type": type(error).__name__,
            "message": str(error),
            "identity_verified": False,
        })
        print(f"collector failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

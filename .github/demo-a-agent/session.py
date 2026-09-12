"""save_session / load_session: minimal single-task session persistence.

A session is a small versioned JSON file written atomically (temp file +
rename). It stores the case identity, the evidence references and digests the
session was built on, the executed tools with compact observations, the
current working-set selections, evicted-selection references and the budget.
Body text is never copied into the session — it stays in the evidence store
and is re-read through the references on restore.

On restore, sources and content are verified: materials that went missing or
changed since the save are explicitly invalidated (missing / content_changed
/ version_conflict), selections that depend on them are invalidated too, and
the remaining reliable materials continue to work.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from materials import Case, HarnessError, load_case
from retrieval import parse_evidence_id

SESSION_SCHEMA = "demo_a_foundation/session/1"

_PERSISTENT_KEYS = ("schema", "case_id", "budget_chars", "working_set", "index", "operations")


def save_session(state: dict, case: Case, path: str | Path) -> dict:
    """Atomically write the session for the given working-context state."""
    payload = {
        "schema": SESSION_SCHEMA,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "case": {
            "manifest": str(case.manifest_path),
            "case_id": case.case_id,
        },
        "materials": [
            {"material_id": m.material_id, "sha256": m.sha256_registered}
            for m in case.materials.values()
        ],
        "budget_chars": state.get("budget_chars"),
        "working_set": state.get("working_set", []),
        "index": state.get("index", {"materials": [], "selections": []}),
        "operations": state.get("operations", []),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    os.replace(tmp, target)
    return payload


def load_session(path: str | Path, config: dict | None = None) -> dict:
    """Load a session and verify its sources and content consistency.

    Returns {"case", "state", "verification"} where verification lists each
    recorded material as verified / missing / content_changed / version_conflict
    and each working-set selection as ok or invalidated with a reason.
    """
    session_path = Path(path)
    if not session_path.is_file():
        raise HarnessError(
            "session_not_found",
            f"session file not found: {session_path}",
            {"session_path": str(session_path)},
        )
    try:
        payload = json.loads(session_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(
            "session_invalid_json",
            f"session file is not valid JSON: {session_path} ({exc})",
            {"session_path": str(session_path)},
        ) from exc
    if payload.get("schema") != SESSION_SCHEMA:
        raise HarnessError(
            "session_version_conflict",
            f"unsupported session schema {payload.get('schema')!r}, expected {SESSION_SCHEMA!r}",
            {"session_path": str(session_path), "found": payload.get("schema")},
        )

    manifest = payload.get("case", {}).get("manifest")
    if not manifest:
        raise HarnessError(
            "session_invalid",
            "session is missing case.manifest",
            {"session_path": str(session_path)},
        )

    # Partial load: keep the reliable materials, list the invalid ones.
    case = load_case(manifest, config=config, strict=False)

    material_status: dict[str, dict] = {}
    invalidated_materials_map: dict[str, dict] = {}
    for recorded in payload.get("materials", []):
        material_id = recorded["material_id"]
        problem = next(
            (p for p in case.load_problems if p.get("material_id") == material_id), None
        )
        if problem:
            kind = "missing" if problem["problem"] == "material_missing" else "content_changed"
            entry = {"status": kind, "detail": problem}
            material_status[material_id] = entry
            invalidated_materials_map[material_id] = entry
        elif material_id not in case.materials:
            entry = {"status": "missing", "detail": "not in manifest"}
            material_status[material_id] = entry
            invalidated_materials_map[material_id] = entry
        elif case.materials[material_id].sha256_registered != recorded["sha256"]:
            # The manifest was re-registered with different content since the
            # session was saved: the evidence version moved under us.
            entry = {
                "status": "version_conflict",
                "detail": {
                    "sha256_in_session": recorded["sha256"],
                    "sha256_in_manifest": case.materials[material_id].sha256_registered,
                },
            }
            material_status[material_id] = entry
            invalidated_materials_map[material_id] = entry
        else:
            material_status[material_id] = {"status": "verified"}

    # Surface invalidation reasons through case.invalidated_materials so that
    # later reads of these ids fail with a meaningful error (content_changed,
    # missing, version_conflict) instead of the generic unknown_material.
    case.invalidated_materials = dict(invalidated_materials_map)

    # Validate working-set selections against the verified materials.
    invalid_material_ids = {
        mid for mid, info in material_status.items() if info["status"] != "verified"
    }
    working_set: list[dict] = []
    invalidated: list[dict] = []
    for sel in payload.get("working_set", []):
        try:
            material_id, anchor, _ = parse_evidence_id(
                f"{sel['material_id']}#L{sel['start']}"
            )
        except (KeyError, HarnessError):
            invalidated.append({"selection": sel, "reason": "malformed_selection"})
            continue
        if material_id in invalid_material_ids:
            invalidated.append(
                {
                    "selection": sel,
                    "reason": f"material_{material_status[material_id]['status']}",
                }
            )
            continue
        material = case.materials[material_id]
        line_count = len(material.read_verified())
        if not (1 <= sel["start"] <= sel["end"] <= line_count):
            invalidated.append({"selection": sel, "reason": "invalid_range"})
            continue
        working_set.append(sel)

    state = {
        "schema": payload.get("schema"),
        "case_id": payload.get("case_id") or case.case_id,
        "budget_chars": payload.get("budget_chars"),
        "working_set": working_set,
        "index": payload.get("index", {"materials": [], "selections": []}),
        "operations": payload.get("operations", []),
    }
    if payload.get("case_id") and payload["case_id"] != case.case_id:
        raise HarnessError(
            "session_case_mismatch",
            f"session belongs to case {payload['case_id']!r} but the manifest "
            f"describes {case.case_id!r}",
            {"session_case": payload["case_id"], "manifest_case": case.case_id},
        )

    return {
        "case": case,
        "state": state,
        "verification": {
            "session_path": str(session_path),
            "saved_at": payload.get("saved_at"),
            "case_id": case.case_id,
            "manifest": manifest,
            "materials": material_status,
            "selections_invalidated": invalidated,
            "working_set_kept": [f"{s['material_id']}#L{s['start']}-L{s['end']}" for s in working_set],
        },
    }

"""build_context / update_context: a budgeted working context over evidence.

Two clearly separated artifacts come out of this module:

- context_text — the text a future model would receive. It is HARD-limited by
  budget_chars (characters, not tokens) and always contains the complete
  picture it claims to show: identity, title, snippets, short references and
  separators are all counted. On success or partial success,
  len(context_text) <= budget_chars. When even the minimum necessary identity
  does not fit, the status is "insufficient" and context_text is empty.

- index / trim_report — harness-local metadata. The index keeps the complete
  source references (material, path, digest, line range, status) for every
  selection ever added, including evicted ones, so anything can be read back
  later. It is never spliced into context_text automatically.

update_context adds new tool observations to the working set: overlapping
snippets are merged, duplicates are avoided, and inclusion is decided by a
simple explainable priority — higher priority first, then most recently added.
What does not fit is moved out of the working set (evicted) or not included
(rejected) with an explicit reason; evidence and the read-back index are never
deleted. No root cause or model-generated summary is ever produced here.
"""

from __future__ import annotations

import copy

from materials import Case, HarnessError
from retrieval import parse_evidence_id, read_evidence

CONTEXT_SCHEMA = "demo_a_foundation/working-context/1"


def _selection_ref(sel: dict) -> str:
    return f"{sel['material_id']}#L{sel['start']}-L{sel['end']}"


def _compact_identity(case: Case) -> str:
    source = case.source
    lines = [f"# {case.case_id}", f"- title: {case.title}"]
    for key in ("repository", "commit", "workflow_path"):
        if key in source:
            lines.append(f"- {key}: {source[key]}")
    run_bits = [f"run {source['run_id']} (attempt {source.get('run_attempt', '?')}, {source.get('event', '?')})"]
    if "run_conclusion" in source:
        run_bits.append(f"conclusion={source['run_conclusion']}")
    lines.append("- " + ", ".join(run_bits))
    for material in case.materials.values():
        if material.line_numbering == "local_excerpt":
            mapping = material.source_mapping or {"status": "unmapped"}
            if mapping.get("status") != "mapped":
                lines.append(f"- unmapped: {material.material_id} (complete source not available)")
    for problem in case.load_problems:
        lines.append(f"- invalid: {problem['material_id']} ({problem['problem']})")
    return "\n".join(lines) + "\n\n"


def _selection_block(sel: dict, lines: list[dict], sha256: str) -> str:
    header = f"## {_selection_ref(sel)} (sha256:{sha256[:8]})\n"
    body = "\n".join(f"{item['line_no']:>5}  {item['text']}".rstrip() for item in lines)
    return header + "```text\n" + body + "\n```\n"


def _render(case: Case, candidates: list[dict], budget_chars: int | None) -> dict:
    """Greedy inclusion: higher priority first, then most recently added."""
    identity = _compact_identity(case)
    if budget_chars is not None and len(identity) > budget_chars:
        return {
            "context_text": "",
            "status": "insufficient",
            "identity_chars": len(identity),
            "included": [],
            "excluded": [
                {**sel, "reason": "insufficient_budget"} for sel in candidates
            ],
        }

    ordered = sorted(candidates, key=lambda s: (-s.get("priority", 0), -s.get("seq", 0)))
    included: list[dict] = []
    excluded: list[dict] = []
    parts: list[str] = []
    used = len(identity)
    for sel in ordered:
        rb = read_evidence(case, _selection_ref(sel), sel["start"], sel["end"])
        block = _selection_block(sel, rb["lines"], rb["sha256"])
        if budget_chars is None or used + len(block) <= budget_chars:
            included.append(sel)
            parts.append(block)
            used += len(block)
        else:
            excluded.append({**sel, "reason": "budget_exceeded"})

    context_text = identity + "".join(parts)
    if budget_chars is None:
        status = "unlimited"
    elif excluded:
        status = "partial"
    else:
        status = "fits"
    return {
        "context_text": context_text,
        "status": status,
        "identity_chars": len(identity),
        "included": included,
        "excluded": excluded,
    }


def _normalize_selection(spec: dict, seq: int) -> dict:
    evidence_id = spec.get("evidence_id")
    if not evidence_id:
        raise HarnessError("invalid_selection", "selection needs an evidence_id", {"spec": spec})
    material_id, start, id_end = parse_evidence_id(evidence_id)
    end = spec.get("end", id_end if id_end is not None else start)
    start = spec.get("start", start)
    return {
        "material_id": material_id,
        "start": start,
        "end": end,
        "priority": int(spec.get("priority", 0)),
        "seq": seq,
    }


def _merge_selection(working: list[dict], new: dict) -> tuple[list[dict], list[str]]:
    """Merge new into the working set: overlapping or adjacent ranges of the
    same material AND same priority become one selection; exact duplicates are
    dropped.

    Ranges of the same material but DIFFERENT priority are intentionally NOT
    merged, so that a high-priority short snippet is not fused into a large
    low-priority neighbour and then evicted as one over-budget block (the
    higher-priority snippet would have fit on its own).

    Returns (new_working_set, refs_absorbed_by_the_merge).
    """
    overlapping = [
        sel
        for sel in working
        if sel["material_id"] == new["material_id"]
        and sel["priority"] == new["priority"]
        and sel["start"] <= new["end"] + 1
        and new["start"] <= sel["end"] + 1
    ]
    if not overlapping:
        return working + [new], []
    merged = {
        "material_id": new["material_id"],
        "start": min([new["start"]] + [s["start"] for s in overlapping]),
        "end": max([new["end"]] + [s["end"] for s in overlapping]),
        "priority": max([new["priority"]] + [s["priority"] for s in overlapping]),
        "seq": max([new["seq"]] + [s["seq"] for s in overlapping]),
    }
    absorbed = [_selection_ref(s) for s in overlapping]
    return [sel for sel in working if sel not in overlapping] + [merged], absorbed


def _index_materials(case: Case) -> list[dict]:
    return [
        {
            "material_id": m.material_id,
            "kind": m.kind,
            "path": m.registered_path,
            "sha256": m.sha256_registered,
            "line_count": len(m.read_verified()),
            "line_numbering": m.line_numbering,
            "source_mapping": m.source_mapping,
            "provenance": m.provenance,
            "limitations": m.limitations,
        }
        for m in case.materials.values()
    ]


def _apply_render(
    case: Case, state: dict, previous_active: list[dict], merged_refs: list[str] | None = None
) -> dict:
    rendered = _render(case, state["working_set"], state["budget_chars"])
    included_refs = {_selection_ref(sel) for sel in rendered["included"]}
    excluded_reasons = {_selection_ref(sel): sel["reason"] for sel in rendered["excluded"]}
    merged_refs = set(merged_refs or [])

    # Migrate statuses of already-known index entries.
    entries: list[dict] = []
    for entry in state["index"]["selections"]:
        ref = _selection_ref(entry)
        if ref in included_refs:
            entry = {**entry, "status": "active"}
            entry.pop("reason", None)
        elif ref in merged_refs:
            entry = {**entry, "status": "merged"}
            entry.pop("reason", None)
        elif entry.get("status") == "active":
            entry = {
                **entry,
                "status": "evicted",
                "reason": excluded_reasons.get(ref, "superseded"),
            }
        entries.append(entry)
    known_refs = {_selection_ref(entry) for entry in entries}

    # Register selections that are new to the index.
    for sel in rendered["included"]:
        if _selection_ref(sel) not in known_refs:
            entries.append({**sel, "status": "active"})
    for sel in rendered["excluded"]:
        if _selection_ref(sel) not in known_refs:
            entries.append({**sel, "status": "rejected", "reason": sel["reason"]})

    state["index"]["selections"] = entries
    state["working_set"] = rendered["included"]
    state["context_text"] = rendered["context_text"]
    state["status"] = rendered["status"]
    state["trim_report"] = {
        "budget_chars": state["budget_chars"],
        "budget_unit": "characters (not tokens)",
        "context_chars": len(rendered["context_text"]),
        "identity_chars": rendered["identity_chars"],
        "included": [_selection_ref(s) for s in rendered["included"]],
        "excluded": [
            {
                "ref": _selection_ref(sel),
                "reason": sel["reason"],
                "status": "evicted"
                if _selection_ref(sel) in {_selection_ref(s) for s in previous_active}
                else "rejected",
            }
            for sel in rendered["excluded"]
        ],
        "note": "budget limits the harness context_text only; index and trim report are harness-local metadata",
    }
    return state


def _new_state(case: Case, budget_chars: int | None) -> dict:
    return {
        "schema": CONTEXT_SCHEMA,
        "case_id": case.case_id,
        "budget_chars": budget_chars,
        "working_set": [],
        "index": {"materials": _index_materials(case), "selections": []},
        "operations": [],
    }


def build_context(case: Case, selections: list[dict], budget_chars: int | None = None) -> dict:
    """Build the initial working context for the given selections."""
    if budget_chars is not None and budget_chars < 1:
        raise HarnessError("invalid_budget", "budget_chars must be >= 1", {"budget_chars": budget_chars})
    state = _new_state(case, budget_chars)
    working: list[dict] = []
    merged_refs: list[str] = []
    for spec in selections:
        working, absorbed = _merge_selection(working, _normalize_selection(spec, seq=len(working) + 1))
        merged_refs.extend(absorbed)
    state["working_set"] = working
    return _apply_render(case, state, previous_active=[], merged_refs=merged_refs)


def update_context(
    case: Case,
    state: dict,
    new_selections: list[dict],
    budget_chars: int | None = None,
) -> dict:
    """Add new observations to the working context and re-render.

    new_selections: list of {"evidence_id": str, "priority"?: int}. Overlapping
    or adjacent ranges of the same material are merged; inclusion is decided
    by priority (higher first) then recency (newer first); what does not fit
    is evicted/rejected with an explicit reason in trim_report.
    """
    if budget_chars is not None and budget_chars < 1:
        raise HarnessError("invalid_budget", "budget_chars must be >= 1", {"budget_chars": budget_chars})
    state = copy.deepcopy(state)
    if budget_chars is not None:
        state["budget_chars"] = budget_chars
    previous_active = [dict(sel) for sel in state["working_set"]]

    next_seq = max([sel.get("seq", 0) for sel in state["working_set"]] + [0]) + 1
    working = [dict(sel) for sel in state["working_set"]]
    merged_refs: list[str] = []
    for spec in new_selections:
        working, absorbed = _merge_selection(working, _normalize_selection(spec, seq=next_seq))
        merged_refs.extend(absorbed)
        next_seq += 1
    state["working_set"] = working
    return _apply_render(case, state, previous_active=previous_active, merged_refs=merged_refs)


def render_context(case: Case, state: dict) -> dict:
    """Recompute context_text / status / trim_report for a restored state."""
    state = copy.deepcopy(state)
    previous_active = [dict(sel) for sel in state["working_set"]]
    return _apply_render(case, state, previous_active=previous_active)

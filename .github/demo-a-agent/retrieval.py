"""search_evidence / read_evidence: keyword search and positional read-back.

Search scans the original text of every registered material of the current
case and returns hits with stable evidence IDs ("<material_id>#L<line>") that
carry the original line number. Reading an evidence ID back verifies the
material digest and splits lines from the same bytes that were verified, so a
file that changed after the case was loaded is reported instead of being
mixed into the answer. Search follows the same read rule.

Long reads can be segmented: pass max_lines to read a window of the requested
range; the result carries a truncation marker and the next position to
continue from, instead of dumping the whole log at once.
"""

from __future__ import annotations

from materials import Case, HarnessError


def parse_evidence_id(evidence_id: str) -> tuple[str, int, int | None]:
    """Parse "<material_id>#L<line>" or "<material_id>#L<a>-L<b>".

    Returns (material_id, anchor_line, end_line_or_None).
    """
    if "#" not in evidence_id:
        raise HarnessError(
            "invalid_evidence_id",
            f"evidence id must look like '<material_id>#L<line>' (optionally '#L<a>-L<b>'), "
            f"got {evidence_id!r}",
            {"evidence_id": evidence_id},
        )
    material_id, _, line_part = evidence_id.partition("#")
    if not material_id or not line_part.startswith("L"):
        raise HarnessError(
            "invalid_evidence_id",
            f"evidence id must look like '<material_id>#L<line>', got {evidence_id!r}",
            {"evidence_id": evidence_id},
        )
    body = line_part[1:]
    if "-" in body:
        a_raw, _, b_raw = body.partition("-")
        if not a_raw.isdigit() or not b_raw.replace("L", "").isdigit():
            raise HarnessError(
                "invalid_evidence_id",
                f"evidence id range is malformed, got {evidence_id!r}",
                {"evidence_id": evidence_id},
            )
        return material_id, int(a_raw), int(b_raw.replace("L", ""))
    if not body.isdigit():
        raise HarnessError(
            "invalid_evidence_id",
            f"evidence id line number is malformed, got {evidence_id!r}",
            {"evidence_id": evidence_id},
        )
    return material_id, int(body), None


def search_evidence(case: Case, keywords: list[str], limit: int | None = None) -> dict:
    """Search the current case's materials for any of the given keywords.

    Matching is a case-insensitive substring test per line, OR across
    keywords. Each hit reports the material, the original line number and the
    full line text. A search that matches nothing returns status "no_matches"
    instead of an empty success. Reads are digest-verified.

    Materials marked as invalidated by the restored session are reported as
    hits with status "invalidated" (and never re-read), so callers can tell
    stale references apart from real matches.
    """
    if not keywords:
        raise HarnessError("invalid_query", "at least one keyword is required")
    if limit is not None and limit < 1:
        raise HarnessError("invalid_query", "limit must be >= 1", {"limit": limit})

    lowered = [(kw, kw.lower()) for kw in keywords]
    hits: list[dict] = []
    invalidated_hits: list[dict] = []
    total_matches = 0
    truncated = False

    # First, surface invalidated materials explicitly.
    for material_id, info in case.invalidated_materials.items():
        invalidated_hits.append(
            {
                "evidence_id": f"{material_id}#L?",
                "material_id": material_id,
                "kind": "invalidated",
                "line_no": None,
                "matched_keywords": [],
                "snippet": (
                    f"material invalidated for the restored session: "
                    f"{info.get('status', 'invalid')}"
                ),
            }
        )

    for material in case.materials.values():
        lines = material.read_verified()  # digest-verified, same bytes
        for line_no, text in enumerate(lines, start=1):
            low = text.lower()
            matched = [kw for kw, low_kw in lowered if low_kw in low]
            if not matched:
                continue
            total_matches += 1
            if limit is None or len(hits) < limit:
                hits.append(
                    {
                        "evidence_id": f"{material.material_id}#L{line_no}",
                        "material_id": material.material_id,
                        "kind": material.kind,
                        "line_no": line_no,
                        "matched_keywords": matched,
                        "snippet": text,
                    }
                )
            else:
                truncated = True

    if not hits and not invalidated_hits:
        return {
            "status": "no_matches",
            "keywords": keywords,
            "limit": limit,
            "hits": [],
        }
    status = "ok" if hits else "invalidated_only"
    return {
        "status": status,
        "keywords": keywords,
        "limit": limit,
        "total_matches": total_matches,
        "returned": len(hits) + len(invalidated_hits),
        "truncated": truncated,
        "hits": hits,
        "invalidated_hits": invalidated_hits,
    }


def read_evidence(
    case: Case,
    evidence_id: str,
    start: int | None = None,
    end: int | None = None,
    max_lines: int | None = None,
) -> dict:
    """Read the original text at an evidence ID (optionally a line range).

    max_lines caps how many lines are returned at once; when it cuts the
    requested range, the result is marked truncated and carries
    continue_from, the next line to read. Fails loudly when the material is
    unknown, the file changed since registration (digest mismatch), or the
    requested range is invalid.
    """
    if max_lines is not None and max_lines < 1:
        raise HarnessError("invalid_query", "max_lines must be >= 1", {"max_lines": max_lines})

    material_id, anchor, id_end = parse_evidence_id(evidence_id)
    material = case.material(material_id)

    # Digest-verified read from one byte snapshot: the returned text is
    # guaranteed to be the content the digest was computed over.
    lines = material.read_verified()
    line_count = len(lines)
    if not (1 <= anchor <= line_count):
        raise HarnessError(
            "invalid_evidence_id",
            f"evidence id anchor line {anchor} is outside material "
            f"'{material_id}' (1..{line_count})",
            {"evidence_id": evidence_id, "line_count": line_count},
        )

    if start is None:
        start = anchor
    if end is None:
        end = id_end if id_end is not None else anchor
    if not (1 <= start <= end <= line_count):
        raise HarnessError(
            "invalid_range",
            f"invalid line range {start}-{end} for material '{material_id}' "
            f"with {line_count} lines",
            {"material_id": material_id, "start": start, "end": end, "line_count": line_count},
        )

    requested = end - start + 1
    truncated = False
    continue_from = None
    actual_end = end
    if max_lines is not None and requested > max_lines:
        actual_end = start + max_lines - 1
        truncated = True
        continue_from = actual_end + 1

    return {
        "evidence_id": evidence_id,
        "material_id": material_id,
        "kind": material.kind,
        "path": material.registered_path,
        "sha256": material.sha256_registered,
        "line_numbering": material.line_numbering,
        "source_mapping": material.source_mapping,
        "range": {"start": start, "end": actual_end},
        "requested_range": {"start": start, "end": end},
        "line_count": line_count,
        "truncated": truncated,
        "continue_from": continue_from,
        "lines": [
            {"line_no": i, "text": lines[i - 1]} for i in range(start, actual_end + 1)
        ],
    }

"""load_case: read a case manifest and register its local materials.

A manifest is a small JSON file describing one real CI case: where the run
information comes from, and which local files hold the original text of the
workflow, logs and run metadata. Each material is bound to its content by a
SHA-256 digest recorded at registration time, so any later change or loss of
the file is detected and reported instead of being silently mixed in.

Materials record their line-numbering semantics: "source" lines are the
material's own original lines; "local_excerpt" lines are local excerpt lines
whose mapping to a complete source is tracked in source_mapping (unmapped
until an actual match has been made).

Paths in a manifest may reference external locations through {placeholders}
that are resolved from a config dict (e.g. {"repo_snapshot": "../..."}), so
sessions and manifests stay relative and portable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


class HarnessError(Exception):
    """Structured error with a machine-readable type and details."""

    def __init__(self, error_type: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"error_type": self.error_type, "message": self.message, "details": self.details}


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def resolve_path(raw_path: str, base_dir: Path, config: dict | None) -> Path:
    """Resolve a manifest path, expanding {placeholder} segments from config.

    Plain relative paths resolve against the manifest directory; placeholder
    values resolve against the config provider's own directory convention
    (here: the demo root, i.e. the parent of the manifest's parent).
    """

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if not config or key not in config:
            raise HarnessError(
                "config_missing",
                f"path {raw_path!r} needs config key '{key}' which is not provided",
                {"missing_key": key, "raw_path": raw_path},
            )
        return str(config[key])

    expanded = _PLACEHOLDER.sub(_sub, raw_path)
    if expanded != raw_path and config and "config_dir" in config:
        # Placeholder values are relative to the config file's directory.
        return (Path(config["config_dir"]) / expanded).resolve()
    return (base_dir / expanded).resolve()


@dataclass
class Material:
    material_id: str
    kind: str
    registered_path: str  # path exactly as written in the manifest (relative)
    path: Path  # resolved absolute path
    sha256_registered: str
    summary: str
    provenance: str
    limitations: list[str] = field(default_factory=list)
    line_numbering: str = "source"  # "source" | "local_excerpt"
    source_mapping: dict | None = None

    def read_verified(self) -> list[str]:
        """Verify the digest and split lines from the SAME bytes.

        Reading this way guarantees that the verified content and the returned
        content are identical even if the file changes on disk concurrently.
        """
        data = self.path.read_bytes()
        current = hashlib.sha256(data).hexdigest()
        if current != self.sha256_registered:
            raise HarnessError(
                "material_digest_mismatch",
                f"material '{self.material_id}' content differs from the registered digest "
                f"(file changed since registration, or the manifest is stale)",
                {
                    "material_id": self.material_id,
                    "path": str(self.path),
                    "sha256_registered": self.sha256_registered,
                    "sha256_current": current,
                },
            )
        return data.decode("utf-8").splitlines()

    # Backwards-compatible alias used by earlier callers.
    def read_lines(self) -> list[str]:
        return self.read_verified()

    def verify_digest(self) -> None:
        current = sha256_of_file(self.path)
        if current != self.sha256_registered:
            raise HarnessError(
                "material_digest_mismatch",
                f"material '{self.material_id}' content differs from the registered digest "
                f"(file changed since registration, or the manifest is stale)",
                {
                    "material_id": self.material_id,
                    "path": str(self.path),
                    "sha256_registered": self.sha256_registered,
                    "sha256_current": current,
                },
            )


@dataclass
class Case:
    case_id: str
    title: str
    source: dict
    materials: dict[str, Material]
    manifest_path: Path
    load_problems: list[dict] = field(default_factory=list)
    # Populated by session.load_session when restore-time validation finds
    # a recorded material whose digest differs from the manifest or whose
    # file went missing. Keys are material_id; values mirror the verification
    # shape (status, detail). Empty when the case was loaded directly.
    invalidated_materials: dict[str, dict] = field(default_factory=dict)

    def material(self, material_id: str) -> Material:
        if material_id in self.invalidated_materials:
            inv = self.invalidated_materials[material_id]
            raise HarnessError(
                f"material_{inv.get('status', 'invalid')}",
                (
                    f"material '{material_id}' is invalidated for the restored "
                    f"session: {inv.get('detail', inv)}"
                ),
                {
                    "material_id": material_id,
                    "invalidation": inv,
                },
            )
        material = self.materials.get(material_id)
        if material is None:
            raise HarnessError(
                "unknown_material",
                f"material '{material_id}' is not registered (or not currently valid) "
                f"in case '{self.case_id}'",
                {
                    "registered_materials": list(self.materials),
                    "load_problems": self.load_problems,
                },
            )
        return material


def _load_material(entry: dict, manifest_dir: Path, config: dict | None, index: int) -> Material:
    for key in ("material_id", "kind", "path", "sha256", "summary"):
        if key not in entry:
            raise HarnessError(
                "manifest_invalid",
                f"materials[{index}] is missing required field '{key}'",
                {"index": index, "missing_field": key},
            )
    path = resolve_path(entry["path"], manifest_dir, config)
    return Material(
        material_id=entry["material_id"],
        kind=entry["kind"],
        registered_path=entry["path"],
        path=path,
        sha256_registered=entry["sha256"],
        summary=entry["summary"],
        provenance=entry.get("provenance", "unknown"),
        limitations=list(entry.get("limitations", [])),
        line_numbering=entry.get("line_numbering", "source"),
        source_mapping=entry.get("source_mapping"),
    )


def _identity_problem_for_job_log(entry: dict, case_source: dict) -> tuple[str | None, dict]:
    """A ``ci_job_log`` material must present a VERIFIABLE cache identity binding.

    The manifest's own ``identity_verified`` / ``identity_status`` fields are
    CLAIMS, not proof. This gate re-checks the binding against the live cache
    meta before the material may register:

    - no binding (or the fields were omitted entirely)  -> identity_unverified
    - manifest sha256 differs from the bound log digest -> identity_unverified
    - bound cache missing / fails its digest check      -> identity_unverified
    - cache meta changed since the binding was made     -> identity_unverified
    - cache meta contradicts the binding identity       -> identity_mismatch
    - cache is positively MISMATCHED (wrong run/attempt) -> identity_mismatch
    - cache has no successful cross-check               -> identity_unverified
    - binding contradicts the case's declared source     -> identity_mismatch

    Returns (problem_type, detail); problem None means the binding verified.
    """
    binding = entry.get("identity_binding")
    if not isinstance(binding, dict):
        return "identity_unverified", {
            "reason": "missing_identity_binding",
            "message": (
                "ci_job_log material carries no verifiable cache identity binding; "
                "a manifest-asserted VERIFIED (or omitted identity fields) alone "
                "is not accepted"
            ),
        }
    if entry.get("sha256") != binding.get("log_sha256"):
        return "identity_unverified", {
            "reason": "material_not_bound_content",
            "message": (
                "manifest sha256 differs from the bound cache log sha256; the "
                "recorded identity check does not apply to this content"
            ),
        }
    # Deferred import: sources imports materials at module level, so the
    # cache-reading helpers are imported here at call time instead.
    from sources import _cache_paths, _load_cache, identity_status  # noqa: PLC0415

    cache_dir = Path(str(binding.get("cache_dir") or ""))
    try:
        log_path, meta_path = _cache_paths(
            cache_dir,
            binding["repo"], binding["run_id"], binding["run_attempt"], binding["job_id"],
        )
    except (KeyError, TypeError):
        return "identity_unverified", {
            "reason": "malformed_identity_binding",
            "message": "identity_binding is missing repo/run_id/run_attempt/job_id",
        }
    meta = _load_cache(meta_path, log_path)
    if meta is None:
        return "identity_unverified", {
            "reason": "cache_missing_or_integrity_failed",
            "message": (
                "the bound cache (log+meta) is missing or fails its digest check; "
                "a local log file alone is not verified material"
            ),
        }
    if binding.get("meta_sha256") is not None:
        current_meta_sha = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        if current_meta_sha != binding["meta_sha256"]:
            return "identity_unverified", {
                "reason": "cache_meta_changed_since_registration",
                "message": (
                    "the cache meta changed after the binding was made; the "
                    "recorded verification result no longer applies"
                ),
            }
    for meta_key, bind_key in (
        ("repo", "repo"), ("run_id", "run_id"),
        ("run_attempt", "run_attempt"), ("job_id", "job_id"),
    ):
        if meta.get(meta_key) != binding.get(bind_key):
            return "identity_mismatch", {
                "reason": "binding_cache_identity_conflict",
                "message": (
                    f"cache meta {meta_key}={meta.get(meta_key)!r} contradicts "
                    f"binding {bind_key}={binding.get(bind_key)!r}"
                ),
            }
    status = identity_status(meta)
    if status == "MISMATCH":
        return "identity_mismatch", {
            "reason": "run_attempt_mismatch",
            "message": (
                "the bound cache is positively known to belong to a different "
                "run/attempt than claimed"
            ),
            "identity_check": meta.get("identity_check"),
        }
    if status != "VERIFIED":
        return "identity_unverified", {
            "reason": (meta.get("identity_check") or {}).get("reason") or "cache_not_verified",
            "message": "the bound cache has no successful identity cross-check",
        }
    for src_key, bind_key in (
        ("repository", "repo"), ("run_id", "run_id"),
        ("run_attempt", "run_attempt"), ("job_id", "job_id"),
    ):
        expected = case_source.get(src_key)
        if expected is not None and str(expected) != str(binding.get(bind_key)):
            return "identity_mismatch", {
                "reason": "case_source_identity_conflict",
                "message": (
                    f"case source {src_key}={expected!r} contradicts binding "
                    f"{bind_key}={binding.get(bind_key)!r}"
                ),
            }
    return None, {}


def load_case(manifest_path: str | Path, config: dict | None = None, strict: bool = True) -> Case:
    """Load one case from its manifest and verify every registered material.

    strict=True (default) raises on any missing file or digest mismatch.
    strict=False returns the case with only the verified materials and the
    problems listed in case.load_problems, so callers can keep working with
    the remaining reliable materials and refuse the invalid ones.
    """
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise HarnessError(
            "manifest_not_found",
            f"case manifest not found: {manifest}",
            {"manifest_path": str(manifest)},
        )
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(
            "manifest_invalid_json",
            f"case manifest is not valid JSON: {manifest} ({exc})",
            {"manifest_path": str(manifest)},
        ) from exc

    for key in ("case_id", "title", "source", "materials"):
        if key not in data:
            raise HarnessError(
                "manifest_invalid",
                f"manifest is missing required field '{key}'",
                {"manifest_path": str(manifest), "missing_field": key},
            )

    manifest_dir = manifest.parent
    materials: dict[str, Material] = {}
    problems: list[dict] = []
    for index, entry in enumerate(data["materials"]):
        material = _load_material(entry, manifest_dir, config, index)
        # Identity gate. For ci_job_log materials the gate is a VERIFIABLE
        # cache identity binding (re-checked here against the live cache);
        # manifest-asserted booleans are never trusted on their own, and
        # omitted identity fields can NOT bypass the gate. Other material
        # kinds keep the previous behavior (a DECLARED non-VERIFIED status is
        # still refused). Refused materials are NOT registered (cannot be
        # retrieved, cannot enter build_context, cannot start a formal
        # diagnosis); the reason stays machine-readable in problems and in
        # case.invalidated_materials.
        claimed_status = entry.get("identity_status")
        claimed_verified = entry.get("identity_verified")
        identity_problem: str | None = None
        identity_detail: dict = {
            "identity_status": claimed_status,
            "identity_verified": claimed_verified,
        }
        if material.kind == "ci_job_log":
            identity_problem, detail = _identity_problem_for_job_log(entry, data["source"])
            identity_detail.update(detail)
        elif claimed_status is not None or claimed_verified is not None:
            if not (claimed_verified is True and claimed_status == "VERIFIED"):
                identity_problem = (
                    "identity_mismatch"
                    if claimed_status == "MISMATCH"
                    else "identity_unverified"
                )
        if identity_problem:
            problems.append(
                {
                    "problem": identity_problem,
                    "material_id": material.material_id,
                    "path": str(material.path),
                    "registered_path": material.registered_path,
                    **identity_detail,
                }
            )
            continue
        if not material.path.is_file():
            problems.append(
                {
                    "problem": "material_missing",
                    "material_id": material.material_id,
                    "path": str(material.path),
                    "registered_path": material.registered_path,
                }
            )
            continue
        current = sha256_of_file(material.path)
        if current != material.sha256_registered:
            problems.append(
                {
                    "problem": "material_digest_mismatch",
                    "material_id": material.material_id,
                    "path": str(material.path),
                    "sha256_registered": material.sha256_registered,
                    "sha256_current": current,
                }
            )
            continue
        materials[material.material_id] = material

    if problems and strict:
        raise HarnessError(
            "materials_invalid",
            f"case '{data['case_id']}' cannot be loaded: {len(problems)} material problem(s), "
            "see details",
            {"problems": problems},
        )

    # Identity-refused materials are surfaced through invalidated_materials so
    # that any later reference fails with a specific, machine-readable reason
    # (material_identity_unverified / material_identity_mismatch) instead of
    # the generic unknown_material, and search reports them as invalidated hits.
    invalidated: dict[str, dict] = {}
    for problem in problems:
        if problem["problem"] in ("identity_unverified", "identity_mismatch"):
            invalidated[problem["material_id"]] = {
                "status": problem["problem"],
                "detail": problem,
            }

    return Case(
        case_id=data["case_id"],
        title=data["title"],
        source=data["source"],
        materials=materials,
        manifest_path=manifest,
        load_problems=problems,
        invalidated_materials=invalidated,
    )


def case_summary(case: Case) -> dict:
    """A JSON-friendly overview of a loaded case and its materials."""
    return {
        "status": "loaded" if not case.load_problems else "loaded_partial",
        "case_id": case.case_id,
        "title": case.title,
        "source": case.source,
        "manifest_path": str(case.manifest_path),
        "load_problems": case.load_problems,
        "invalidated_materials": case.invalidated_materials,
        "materials": [
            {
                "material_id": m.material_id,
                "kind": m.kind,
                "path": m.registered_path,
                "sha256": m.sha256_registered,
                "line_count": len(m.read_verified()),
                "line_numbering": m.line_numbering,
                "source_mapping": m.source_mapping,
                "summary": m.summary,
                "provenance": m.provenance,
                "limitations": m.limitations,
                "digest_verified": True,
            }
            for m in case.materials.values()
        ],
    }

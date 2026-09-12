"""CLI for the minimal CI post-hoc harness (demo_a_foundation).

Core capabilities (one simple CLI; the same module functions are the future
Agent tool surface):

  load           <manifest>                       load a case and verify materials
  search         <manifest|--session> KEYWORD..   keyword search over case materials
  read           <manifest|--session> EVIDENCE    read original text (segmented via --max-lines)
  context        <manifest> --sel SPEC..          build the budgeted working context
  update-context --session F --add SPEC..          add observations, evict under budget
  resume         --session F                      restore a session in a new process
  fetch-log      <manifest>                       targeted read-only fetch of the job log
  map-excerpt    <manifest> --excerpt M --source M  map a local excerpt into a complete source

Every command accepts --out PATH to also write its output to a file. Errors are
structured JSON on stderr with exit code 1. Offline by default: only
fetch-log touches the network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from context import build_context, render_context, update_context
from config import load_all, load_runtime_config, load_structural_config
from materials import HarnessError, case_summary, load_case
from provider import LLMProvider
from retrieval import parse_evidence_id, read_evidence, search_evidence
from session import load_session, save_session
from sources import fetch_job_log, manual_import, match_excerpt

DEMO_ROOT = Path(__file__).resolve().parent


def _load_config() -> dict:
    return load_structural_config()


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _write_out(path: str | None, text: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # The file is written byte-identically to the in-memory text (UTF-8, LF)
    # so the budget check (len(context_text)) matches what is on disk, then
    # read back and compared: console display must never run against an
    # unverified persistence.
    data = text.encode("utf-8")
    with open(target, "wb") as fh:
        fh.write(data)
    if target.read_bytes() != data:
        raise HarnessError(
            "output_write_unverified",
            f"output file did not verify after write: {target}",
            {"path": str(target)},
        )


def _safe_print_output(text: str) -> None:
    """Print for console display without crashing on console encoding limits.

    Persistence happens before this is called (--out), or nothing was expected
    to persist (no --out). A console that cannot encode a character (e.g. GBK
    vs U+FEFF or emoji) gets a backslashreplace-escaped display instead of an
    exception. Display-only: it never alters what was persisted.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        try:
            display = text.encode(enc, errors="backslashreplace").decode(
                enc, errors="backslashreplace"
            )
        except (UnicodeEncodeError, LookupError):
            display = text.encode("ascii", errors="backslashreplace").decode("ascii")
        print(display)


def _parse_selection(spec: str) -> dict:
    material_id, start, end = parse_evidence_id(spec)
    return {"evidence_id": spec, "start": start, "end": end}


def _context_output(state: dict, fmt: str) -> str:
    if fmt == "md":
        # The markdown output is exactly context_text: what a future model
        # would receive, hard-limited by the budget.
        return state["context_text"]
    return _dump(
        {
            "case_id": state["case_id"],
            "budget_chars": state["budget_chars"],
            "status": state["status"],
            "context_text": state["context_text"],
            "context_chars": len(state["context_text"]),
            "working_set": state["working_set"],
            "trim_report": state["trim_report"],
            "index": state["index"],
        }
    )


def _append_operation(state: dict, tool: str, args: dict, observation: dict) -> None:
    state.setdefault("operations", [])
    seq = max([op.get("seq", 0) for op in state["operations"]] + [0]) + 1
    state["operations"].append({"seq": seq, "tool": tool, "args": args, "observation": observation})


def _resolve_case_and_session(args, strict: bool = True):
    """Return (case, state_or_None, session_path_or_None)."""
    config = _load_config()
    if getattr(args, "session", None):
        loaded = load_session(args.session, config=config)
        return loaded["case"], loaded["state"], args.session
    case = load_case(args.manifest, config=config, strict=strict)
    return case, None, None


def _save_state_session(session_path: str | None, state: dict, case) -> None:
    if session_path:
        save_session(state, case, session_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Minimal CI post-hoc harness (demo_a_foundation)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="load a case manifest and verify materials")
    p_load.add_argument("manifest")
    p_load.add_argument("--out")

    p_search = sub.add_parser("search", help="search case materials for keywords")
    p_search.add_argument("manifest", nargs="?", default=None)
    p_search.add_argument("--session", default=None)
    p_search.add_argument("keywords", nargs="+")
    p_search.add_argument("--limit", type=int)
    p_search.add_argument("--out")

    p_read = sub.add_parser("read", help="read original text at an evidence id")
    p_read.add_argument("manifest", nargs="?", default=None)
    p_read.add_argument("--session", default=None)
    p_read.add_argument("evidence_id")
    p_read.add_argument("--start", type=int)
    p_read.add_argument("--end", type=int)
    p_read.add_argument("--max-lines", type=int)
    p_read.add_argument("--out")

    p_ctx = sub.add_parser("context", help="build the budgeted working context")
    p_ctx.add_argument("manifest")
    p_ctx.add_argument(
        "--sel",
        action="append",
        default=[],
        required=True,
        metavar="SPEC",
        help="selection: <material_id>#L<a> or <material_id>#L<a>-L<b> (repeatable)",
    )
    p_ctx.add_argument("--budget", type=int, default=None, help="character budget for context_text")
    p_ctx.add_argument("--format", choices=["md", "json"], default="md")
    p_ctx.add_argument("--out")
    p_ctx.add_argument("--out-session", default=None, help="save the working context as a session")

    p_upd = sub.add_parser("update-context", help="add observations to a session's working context")
    p_upd.add_argument("--session", required=True)
    p_upd.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="SPEC",
        help="selection to add: <material_id>#L<a>[-L<b>][@priority] (repeatable)",
    )
    p_upd.add_argument("--budget", type=int, default=None)
    p_upd.add_argument("--format", choices=["md", "json"], default="md")
    p_upd.add_argument("--out")

    p_resume = sub.add_parser("resume", help="restore a session (new process) and verify it")
    p_resume.add_argument("--session", required=True)
    p_resume.add_argument("--out")

    p_fetch = sub.add_parser("fetch-log", help="targeted read-only fetch of the complete job log")
    p_fetch.add_argument("manifest")
    p_fetch.add_argument("--timeout", type=float, default=15.0)
    p_fetch.add_argument("--retries", type=int, default=1)
    p_fetch.add_argument("--token", default=None, help="override GitHub token (otherwise uses env)")
    p_fetch.add_argument("--out")

    p_map = sub.add_parser("map-excerpt", help="map a local excerpt into a complete source")
    p_map.add_argument("manifest")
    p_map.add_argument("--excerpt", required=True, help="material id of the local excerpt")
    p_map.add_argument("--source", required=True, help="material id of the complete source")
    p_map.add_argument("--out")

    p_import = sub.add_parser(
        "import-log",
        help="manually import an already-fetched log into the cache (backup path)",
    )
    p_import.add_argument("manifest")
    p_import.add_argument("--input", required=True, help="path to a file containing the full log")
    p_import.add_argument("--label", default="manual_import", help="source_label for the cache meta")
    p_import.add_argument("--out")

    p_config = sub.add_parser(
        "config-check",
        help="show effective structural + runtime configuration (secrets redacted)",
    )
    p_config.add_argument("--env-file", default=None, help="override .env path")
    p_config.add_argument("--out")

    p_diag = sub.add_parser(
        "diagnose",
        help="run the minimal single-agent diagnostic loop over a case",
    )
    p_diag.add_argument("manifest")
    p_diag.add_argument("--max-steps", type=int, default=None)
    p_diag.add_argument("--out")

    args = parser.parse_args(argv)

    try:
        if args.command == "load":
            case = load_case(args.manifest, config=_load_config())
            output = _dump(case_summary(case))

        elif args.command == "search":
            case, state, session_path = _resolve_case_and_session(args)
            result = search_evidence(case, args.keywords, args.limit)
            if session_path:
                _append_operation(
                    state,
                    "search",
                    {"keywords": args.keywords, "limit": args.limit},
                    {
                        "status": result["status"],
                        "total_matches": result.get("total_matches", 0),
                        "hit_ids": [h["evidence_id"] for h in result.get("hits", [])],
                    },
                )
                _save_state_session(session_path, state, case)
            output = _dump(result)

        elif args.command == "read":
            case, state, session_path = _resolve_case_and_session(args)
            result = read_evidence(
                case, args.evidence_id, args.start, args.end, args.max_lines
            )
            if session_path:
                _append_operation(
                    state,
                    "read",
                    {
                        "evidence_id": args.evidence_id,
                        "start": args.start,
                        "end": args.end,
                        "max_lines": args.max_lines,
                    },
                    {
                        "range": result["range"],
                        "truncated": result["truncated"],
                        "continue_from": result["continue_from"],
                        "line_numbering": result["line_numbering"],
                    },
                )
                _save_state_session(session_path, state, case)
            output = _dump(result)

        elif args.command == "context":
            case = load_case(args.manifest, config=_load_config())
            selections = [_parse_selection(spec) for spec in args.sel]
            state = build_context(case, selections, args.budget)
            _append_operation(
                state,
                "build_context",
                {"selections": args.sel, "budget_chars": args.budget},
                {
                    "status": state["status"],
                    "context_chars": len(state["context_text"]),
                    "included": state["trim_report"]["included"],
                },
            )
            if args.out_session:
                save_session(state, case, args.out_session)
            output = _context_output(state, args.format)

        elif args.command == "update-context":
            loaded = load_session(args.session, config=_load_config())
            case, state = loaded["case"], loaded["state"]
            new_selections = []
            for spec in args.add:
                priority = 0
                if "@" in spec:
                    spec, _, prio_raw = spec.rpartition("@")
                    if not prio_raw.lstrip("-").isdigit():
                        raise HarnessError(
                            "invalid_selection",
                            f"priority in {args.add!r} must be an integer",
                            {"spec": spec},
                        )
                    priority = int(prio_raw)
                material_id, start, end = parse_evidence_id(spec)
                new_selections.append(
                    {
                        "evidence_id": spec,
                        "start": start,
                        "end": end,
                        "priority": priority,
                    }
                )
            state = update_context(case, state, new_selections, args.budget)
            _append_operation(
                state,
                "update_context",
                {"add": args.add, "budget_chars": args.budget},
                {
                    "status": state["status"],
                    "context_chars": len(state["context_text"]),
                    "included": state["trim_report"]["included"],
                    "excluded": state["trim_report"]["excluded"],
                },
            )
            save_session(state, case, args.session)
            output = _context_output(state, args.format)

        elif args.command == "resume":
            loaded = load_session(args.session, config=_load_config())
            case, state = loaded["case"], loaded["state"]
            state = render_context(case, state)
            output = _dump(
                {
                    "resumed": True,
                    "verification": loaded["verification"],
                    "budget_chars": state["budget_chars"],
                    "status": state["status"],
                    "context_chars": len(state["context_text"]),
                    "working_set": [
                        f"{s['material_id']}#L{s['start']}-L{s['end']}" for s in state["working_set"]
                    ],
                    "operations_count": len(state["operations"]),
                }
            )

        elif args.command == "fetch-log":
            case = load_case(args.manifest, config=_load_config())
            source = case.source
            for key in ("repository", "run_id", "run_attempt"):
                if key not in source:
                    raise HarnessError(
                        "manifest_invalid",
                        f"manifest source is missing '{key}' (needed for the targeted fetch)",
                        {"missing_field": key},
                    )
            job_id = source.get("job_id")
            if not job_id:
                raise HarnessError(
                    "manifest_invalid",
                    "manifest source is missing 'job_id' (needed for the targeted fetch)",
                    {"missing_field": "job_id"},
                )
            token = args.token
            token_source: str | None = "cli:override"
            if token is None:
                runtime = load_runtime_config()
                if runtime.is_filled("DEMO_A_GITHUB_TOKEN"):
                    token = runtime.get("DEMO_A_GITHUB_TOKEN")
                    token_source = "env:DEMO_A_GITHUB_TOKEN"
            result = fetch_job_log(
                repo=source["repository"],
                run_id=int(source["run_id"]),
                attempt=int(source["run_attempt"]),
                job_id=int(job_id),
                cache_dir=DEMO_ROOT / "cache",
                timeout=args.timeout,
                retries=args.retries,
                token=token,
                token_source=token_source,
            )
            output = _dump(result)

        elif args.command == "map-excerpt":
            case = load_case(args.manifest, config=_load_config())
            excerpt = case.material(args.excerpt)
            source = case.material(args.source)
            excerpt_lines = excerpt.read_verified()
            source_lines = source.read_verified()
            result = match_excerpt(excerpt_lines, source_lines)
            result.update(
                {
                    "excerpt_material": excerpt.material_id,
                    "excerpt_line_count": len(excerpt_lines),
                    "source_material": source.material_id,
                    "source_sha256": source.sha256_registered,
                }
            )
            output = _dump(result)

        elif args.command == "import-log":
            case = load_case(args.manifest, config=_load_config())
            source = case.source
            for key in ("repository", "run_id", "run_attempt", "job_id"):
                if key not in source:
                    raise HarnessError(
                        "manifest_invalid",
                        f"manifest source is missing '{key}' (needed for manual import)",
                        {"missing_field": key},
                    )
            input_path = Path(args.input)
            if not input_path.is_file():
                raise HarnessError(
                    "import_input_missing",
                    f"manual import input file not found: {input_path}",
                    {"input_path": str(input_path)},
                )
            result = manual_import(
                repo=source["repository"],
                run_id=int(source["run_id"]),
                attempt=int(source["run_attempt"]),
                job_id=int(source["job_id"]),
                raw_text=input_path.read_bytes(),
                cache_dir=DEMO_ROOT / "cache",
                source_label=args.label,
                extra={"input_path": str(input_path)},
            )
            output = _dump(result)

        elif args.command == "config-check":
            output = _dump(load_all(env_file=args.env_file))

        elif args.command == "diagnose":
            from agent import run_agent, run_result_to_dict

            case = load_case(args.manifest, config=_load_config())
            result = run_agent(case, max_steps=args.max_steps)
            output = _dump(run_result_to_dict(result))

        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"unknown command {args.command!r}")
    except HarnessError as exc:
        print(_dump({"error": exc.to_dict()}), file=sys.stderr)
        return 1

    # Persistence first: the formal result is written and verified before any
    # console display is attempted, so a GBK/limited console can no longer
    # destroy a real result (print(output) used to crash on U+FEFF before
    # _write_out ran).
    try:
        _write_out(args.out, output)
    except HarnessError as exc:
        print(_dump({"error": exc.to_dict()}), file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            _dump(
                {
                    "error": {
                        "error_type": "output_write_failed",
                        "message": str(exc),
                        "details": {"path": args.out},
                    }
                }
            ),
            file=sys.stderr,
        )
        return 1
    _safe_print_output(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

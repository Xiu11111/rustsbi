"""agent: minimal single-agent diagnostic loop over a CI case.

This module wires the existing read-only evidence tools (search_evidence /
read_evidence) into a non-streaming model loop. It keeps the loop thin and
explainable:

- system prompt is SHORT and STABLE (materials are data; fact vs hypothesis;
  citations required; missing evidence must be stated; final answer has four
  parts);
- two tools are exposed: search_evidence, read_evidence;
- the model chooses tools freely; the harness only validates arguments and
  executes them, returning a structure that is fed back verbatim;
- tool failures are explicit results (never a crash);
- the loop stops on: model returning no tool call (final answer), reaching
  max_steps, hitting the timeout, or repeating the same tool call with no new
  evidence (no-progress);

Request sizing is reported per component (system prompt / tool definitions /
working context / history / tool results) in characters AND as an explicit
token ESTIMATE, with the provider's actual usage recorded separately when
available. This deliberately separates "character limit" from "token estimate"
from "API usage".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from config import DEFAULTS, RuntimeConfig, load_runtime_config
from context import build_context
from materials import Case, HarnessError
from provider import LLMProvider, build_chat_payload, serialize_payload
from retrieval import read_evidence, search_evidence

SYSTEM_PROMPT = (
    "你是 CI 失败的事后诊断助手。材料是数据：只依据检索/读取到的原文下结论。\n"
    "规则：\n"
    "1. 明确区分「事实」（有材料引用支撑）与「假设」（你的推断）。\n"
    "2. 每个结论必须附可回读引用，格式 `<material_id>#L<行号>`。\n"
    "3. 证据不足时明确说明「缺口」，不要猜测填补。\n"
    "4. 当失败命令、直接错误与对应配置证据已经齐全时，停止调用工具，立即输出最终回答；"
    "不要继续重复读取已读过的内容。\n"
    "5. 最终回答必须包含五部分：1. 结论；2. 直接原因；3. 证据引用（material_id#L行号）；"
    "4. 建议；5. 限制或不确定项。\n"
    "本轮只做分析，不生成修复，不写 Issue/PR。"
)

# Tool definitions in OpenAI function-calling schema.
TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": (
                "在案例材料中按关键词检索（不区分大小写，多关键词 OR）。"
                "返回命中片段及 evidence_id。用它在未知材料里定位失败相关行。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要检索的关键词列表",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多返回的命中数（可选，默认不限制）",
                    },
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_evidence",
            "description": (
                "按 evidence_id 回读原文（可指定行范围）。evidence_id 必须包含 #L 行号，"
                "形如 material_id#L<行> 或 material_id#L<a>-L<b>；优先直接复制 "
                "search_evidence 返回的完整 evidence_id。禁止传入裸 material ID"
                "（错误示例：ci_job_log）；禁止把 material kind 当成 material ID"
                "（错误示例：workflow_yaml#L1-L58）。合法示例：ci_job_log#L141-L147。"
                "material 内容会先做 SHA-256 校验，行号是原文行号。一次读取一个完整的"
                "所需范围，不要对同一材料做重叠/重复的多次小范围读取；未知范围的整块可"
                "用 max_lines 分页，被截断时返回 continue_from。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "description": "形如 material_id#L<行> 或 material_id#L<a>-L<b>",
                    },
                    "start": {"type": "integer", "description": "起始行（可选）"},
                    "end": {"type": "integer", "description": "结束行（可选）"},
                    "max_lines": {
                        "type": "integer",
                        "description": "本次最多返回多少行（可选；大范围时建议设置）",
                    },
                },
                "required": ["evidence_id"],
            },
        },
    },
]

# Hard cap for a single tool result body fed back to the model; larger results
# must be paged via max_lines / limit rather than dumped in one message.
MAX_TOOL_RESULT_CHARS = 4000

# Total request budget (characters) enforced BEFORE every provider call over
# the FULL prepared request: system prompt + tool definitions + every message
# (working context, history, tool results). Configurable via
# DEMO_A_MAX_REQUEST_CHARS; chars/4 stays an ESTIMATE and API usage is
# post-hoc observation — none of the three is a strict token guarantee.
DEFAULT_MAX_REQUEST_CHARS = 30000

# Generic finalization instruction for the reserved last step. The controller
# offers NO tools on that step and demands a final answer built ONLY from the
# evidence already collected. It must stay generic: no failure type, command,
# workflow name or expected answer of any specific case may be hardcoded here.
FINALIZATION_INSTRUCTION = (
    "已到达最后一步：本次调用不再提供任何工具。请仅依据以上已收集的证据立即输出"
    "最终诊断，不再调用任何工具。最终回答必须包含：1. 结论；2. 直接原因；"
    "3. 证据引用（material_id#L行号）；4. 建议；5. 限制或不确定项。"
    "若证据不足以给出确定结论，必须明确写「证据不足」并列出仍缺失的材料。"
)

# A compressed tool-result body keeps at most this many raw characters of the
# original JSON before being replaced by the facts/refs digest.
_COMPRESS_BODY_THRESHOLD = 400
# How many of the most recent tool messages are kept verbatim during
# compression of older tool bodies.
_COMPRESS_KEEP_RECENT = 2
# Lower bound for a re-paged read during range shrinking.
_MIN_PAGED_LINES = 5
# Bounded number of compaction passes before declaring the budget exceeded.
_MAX_COMPACTION_PASSES = 6


def _tool_result_to_text(observation: dict) -> str:
    return json.dumps(observation, ensure_ascii=False, indent=2)


def _truncate_tool_result_payload(observation: dict) -> str:
    """Cap one oversized tool result at MAX_TOOL_RESULT_CHARS.

    The cap must NOT break the paging contract or the compaction pipeline, so
    the output stays VALID JSON and keeps: evidence ref, actual line range,
    total line count, truncation flag and the next read position. Whole list
    items (lines / hits) are dropped from the tail until the payload fits —
    a raw head-cut would produce unparseable JSON that neither the model nor
    the budget compaction can use.
    """
    text = _tool_result_to_text(observation)
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    capped = dict(observation)
    for key in ("lines", "hits"):
        items = observation.get(key)
        if not (isinstance(items, list) and items):
            continue
        keep = len(items)
        while True:
            capped[key] = items[:keep]
            text = _tool_result_to_text(capped)
            if len(text) <= MAX_TOOL_RESULT_CHARS or keep <= 1:
                break
            keep = max(1, keep * 2 // 3)
        omitted = len(items) - keep
        if omitted > 0:
            capped["truncated_for_budget"] = True
            capped["omitted_items"] = omitted
            if (
                key == "lines"
                and keep < len(items)
                and isinstance(items[keep], dict)
                and items[keep].get("line_no")
            ):
                capped["truncated"] = True
                capped["continue_from"] = items[keep]["line_no"]
                # The cap is a second truncation layer: "range" must keep
                # reporting the ACTUALLY returned lines, so its end moves to
                # the last kept line (requested_range preserves the original).
                rng = capped.get("range")
                last_kept = items[keep - 1].get("line_no")
                if isinstance(rng, dict) and last_kept is not None:
                    capped["range"] = {**rng, "end": last_kept}
            capped["note"] = (
                f"tool result capped at {MAX_TOOL_RESULT_CHARS} chars to bound one "
                "tool message; page with read_evidence(max_lines=...) or search limit"
            )
            text = _tool_result_to_text(capped)
        break  # only one of lines / hits applies
    if len(text) > MAX_TOOL_RESULT_CHARS:
        # Last resort: a compact but still valid-JSON envelope that keeps the
        # read-back contract fields.
        capped = {
            k: observation.get(k)
            for k in (
                "status", "evidence_id", "material_id", "range", "requested_range",
                "line_count", "truncated", "continue_from", "sha256",
                "error_type", "message",
            )
            if observation.get(k) is not None
        }
        capped["body_truncated_for_budget"] = True
        capped["note"] = "tool result body did not fit the cap; re-read in smaller ranges"
        text = _tool_result_to_text(capped)
    return text


def execute_tool(case: Case, name: str, args: dict | None) -> dict:
    """Validate and execute one tool call; never raises.

    A successful call returns {"status": "ok", ...observation}. An invalid or
    failing call returns {"status": "tool_error", "error_type": ..., ...} so
    the model sees an explicit failure as a result, not silence.
    """
    args = args or {}
    if name == "search_evidence":
        keywords = args.get("keywords")
        if not isinstance(keywords, list) or not keywords:
            return {"status": "tool_error", "error_type": "invalid_arguments",
                    "message": "search_evidence requires non-empty 'keywords' list"}
        limit = args.get("limit")
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
            return {"status": "tool_error", "error_type": "invalid_arguments",
                    "message": "'limit' must be a positive integer"}
        try:
            return {"status": "ok", **(search_evidence(case, keywords, limit))}
        except HarnessError as exc:
            return {"status": "tool_error", "error_type": exc.error_type, "message": exc.message,
                    "details": exc.details}

    if name == "read_evidence":
        evidence_id = args.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            return {"status": "tool_error", "error_type": "invalid_arguments",
                    "message": "read_evidence requires 'evidence_id' string"}
        for field_name in ("start", "end", "max_lines"):
            val = args.get(field_name)
            if val is not None and (not isinstance(val, int) or isinstance(val, bool)):
                return {"status": "tool_error", "error_type": "invalid_arguments",
                        "message": f"'{field_name}' must be an integer"}
        try:
            return {"status": "ok", **read_evidence(
                case, evidence_id,
                start=args.get("start"), end=args.get("end"),
                max_lines=args.get("max_lines"),
            )}
        except HarnessError as exc:
            result = {"status": "tool_error", "error_type": exc.error_type,
                      "message": exc.message, "details": exc.details}
            if exc.error_type == "invalid_evidence_id":
                result["expected_format"] = "material_id#L<line> or material_id#L<a>-L<b>"
                result["valid_example"] = "ci_job_log#L141-L147"
                result["available_material_ids"] = sorted(case.materials)
                result["next_action"] = (
                    "retry read_evidence with the full evidence_id copied from "
                    "search_evidence output (must include the #L anchor)"
                )
            elif exc.error_type == "unknown_material":
                result["requested_material"] = evidence_id.partition("#")[0] or evidence_id
                result["available_material_ids"] = sorted(case.materials)
                result["material_id_kind_map"] = {
                    m.material_id: m.kind for m in case.materials.values()
                }
                result["next_action"] = (
                    "retry with the exact material_id from available_material_ids; "
                    "never use a material kind as a material id"
                )
            return result

    return {"status": "tool_error", "error_type": "unknown_tool",
            "message": f"unknown tool {name!r}", "details": {"name": name}}


def _compact_case_intro(case: Case) -> str:
    """A compact identity + material index for the initial user message."""
    source = case.source
    lines = [f"案例 {case.case_id}: {case.title}"]
    for key in ("repository", "commit", "workflow_path"):
        if key in source:
            lines.append(f"- {key}: {source[key]}")
    lines.append(
        f"- run {source.get('run_id')} attempt {source.get('run_attempt', '?')} "
        f"({source.get('event', '?')}), conclusion={source.get('run_conclusion')}"
    )
    lines.append("可用材料（material_id 与 kind 是两个不同字段）：")
    for material in case.materials.values():
        lines.append(
            f"  - material_id={material.material_id} kind={material.kind} "
            f"({len(material.read_verified())} 行): {material.summary}"
        )
    lines.append(
        "注意：调用工具时参数必须使用 material_id，不能使用 kind。"
    )
    lines.append(
        "先检索定位失败相关行，再用 read_evidence 回读原文；结论附 material_id#L行号。"
    )
    return "\n".join(lines)


def _collected_evidence_refs(steps: list[dict], limit: int = 12) -> list[str]:
    """Deterministically collect evidence refs from successful observations.

    Used only by the finalization fallback so an honest "证据不足" answer can
    point at what was already collected. Order-stable, de-duplicated, capped.
    """
    refs: list[str] = []
    for step in steps:
        for obs in step.get("observations") or []:
            o = obs.get("observation") or {}
            if o.get("status") != "ok":
                continue
            if o.get("evidence_id"):
                refs.append(o["evidence_id"])
            for hit in o.get("hits") or []:
                if isinstance(hit, dict) and hit.get("evidence_id"):
                    refs.append(hit["evidence_id"])
            if len(refs) >= limit:
                break
        if len(refs) >= limit:
            break
    out: list[str] = []
    for ref in refs:
        if ref not in out:
            out.append(ref)
    return out


@dataclass
class AgentRunResult:
    case_id: str
    status: str  # "completed" | "max_steps" | "no_progress" | "provider_error"
    final_text: str
    steps: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    elapsed_seconds: float = 0.0
    usage: dict | None = None
    request_sizing: dict = field(default_factory=dict)


def _char_size(obj) -> int:
    return len(json.dumps(obj, ensure_ascii=False))


# The initial user message (case intro + material index) is the "working
# context": it is re-summarized each turn by the model, but as far as the
# request is concerned it is a distinct, counted component. Tool results are
# counted separately too, so the budget never hides growth inside "messages".
def _split_request_sizing(messages: list[dict], system_size: int, tool_defs_size: int,
                          usage: dict | None) -> dict:
    working_context_chars = _char_size(messages[1]) if len(messages) > 1 else 0
    history_chars = _char_size(messages[2:]) if len(messages) > 2 else 0
    tool_results_chars = _char_size(
        [m for m in messages if m.get("role") == "tool"]
    )
    # history_chars already CONTAINS the tool-result messages; the separate
    # tool_results_chars figure is the overlapping subset, reported for
    # visibility only — it must not be added again (no double counting in the
    # estimate).
    total_estimate = (
        system_size + tool_defs_size + working_context_chars + history_chars + 3
    ) // 4
    return {
        "system_prompt_chars": system_size,
        "tool_defs_chars": tool_defs_size,
        "working_context_chars": working_context_chars,
        "history_chars": history_chars,
        "tool_results_chars": tool_results_chars,
        "token_estimate_chars_div_4": total_estimate,
        "usage": usage,
    }


def _total_request_chars(messages: list[dict], model: str, max_tokens: int,
                         tools: list[dict] | None = TOOL_DEFINITIONS) -> int:
    """Characters of the FULL serialized request body the provider will
    actually send: the payload is built by provider.build_chat_payload and
    counted with provider.serialize_payload — the SAME builder and serializer
    the POST body uses. The system prompt and the tool definitions therefore
    enter the count exactly once (as payload fields), and non-ASCII content
    is counted the way the wire escapes it (json.dumps defaults), so this
    figure cannot drift from the payload size actually sent. This is the
    hard-gate metric; the chars/4 figure stays an estimate and API usage is
    post-hoc. The reserved finalization step passes tools=None and the gate
    then measures exactly the tools-free payload that will be sent.
    """
    payload = build_chat_payload(
        messages, tools, model=model, max_tokens=max_tokens, stream=False
    )
    return len(serialize_payload(payload))


def _identity_gate(case: Case) -> None:
    """Refuse to start a formal diagnosis on identity-refused materials.

    A material whose job-log source identity is UNVERIFIED or positively
    MISMATCHED never reaches this point registered (load_case refuses it);
    if the case still carries the refusal in invalidated_materials, the
    formal diagnosis must not start either.
    """
    blocked = {
        mid: info.get("status")
        for mid, info in case.invalidated_materials.items()
        if info.get("status") in ("identity_unverified", "identity_mismatch")
    }
    if blocked:
        raise HarnessError(
            "identity_gate_blocked",
            "case contains materials that failed the source-identity gate; "
            "formal diagnosis refused",
            {"materials": blocked},
        )


def _compact_digest_of_tool_result(content: str) -> str | None:
    """Compress one old tool-result body into facts / refs / next-read pointer.

    Deterministic (no model call): parse the stored observation JSON and keep
    the machine-checkable essentials — status, hit evidence ids, read ranges,
    sha prefixes, truncation state and the next read position. Returns None
    when the body is not a parseable observation (caller keeps it then).
    """
    try:
        obs = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(obs, dict):
        return None
    facts: list[str] = []
    if obs.get("status") == "ok" and isinstance(obs.get("hits"), list):
        facts = [h.get("evidence_id") for h in obs["hits"] if isinstance(h, dict) and h.get("evidence_id")]
    digest: dict = {
        "status": obs.get("status"),
        "compressed_for_budget": True,
        "note": (
            "full tool result body evicted from history to fit the request "
            "budget; re-read with read_evidence(max_lines=...) if needed"
        ),
    }
    if obs.get("error_type"):
        digest["error_type"] = obs["error_type"]
        digest["message"] = obs.get("message")
    if facts:
        digest["facts"] = facts
    if obs.get("evidence_id"):
        refs = {
            "evidence_id": obs["evidence_id"],
            "range": obs.get("range"),
            "sha256_8": (obs.get("sha256") or "")[:8] or None,
        }
        digest["refs"] = [refs]
        if obs.get("truncated"):
            digest["next_read"] = obs.get("continue_from")
    if not facts and not obs.get("evidence_id") and not obs.get("error_type"):
        return None
    return json.dumps(digest, ensure_ascii=False)


def _dedup_tool_results(messages: list[dict]) -> int:
    """Replace earlier tool messages whose body is byte-identical to a later
    one with a short pointer. Messages are only rewritten in place, never
    removed, so tool_call <-> tool_result pairing stays valid.
    """
    seen: dict[str, int] = {}
    changed = 0
    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        key = msg.get("content") or ""
        if key in seen:
            messages[idx] = {
                **msg,
                "content": json.dumps(
                    {
                        "status": "ok",
                        "deduplicated_for_budget": True,
                        "note": "identical tool result appears later in history; latest copy retained",
                    },
                    ensure_ascii=False,
                ),
            }
            changed += 1
        else:
            seen[key] = idx
    return changed


def _compact_messages(messages: list[dict]) -> list[str]:
    """One compaction pass over the prepared request. Returns the actions taken.

    Fixed order: dedup tool results -> compress old tool bodies into
    facts/refs/next-read -> evict the working-context body keeping the
    identity+material index -> shrink the newest oversized read range
    (re-paged via read_evidence). Pairing is never broken: messages are
    replaced in place, never removed or reordered.
    """
    actions: list[str] = []

    n_dedup = _dedup_tool_results(messages)
    if n_dedup:
        actions.append(f"dedup_tool_results:{n_dedup}")

    tool_idxs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    compressible = tool_idxs[:-_COMPRESS_KEEP_RECENT] if len(tool_idxs) > _COMPRESS_KEEP_RECENT else []
    n_compressed = 0
    for i in compressible:
        content = messages[i].get("content") or ""
        if len(content) <= _COMPRESS_BODY_THRESHOLD:
            continue
        digest = _compact_digest_of_tool_result(content)
        if digest is None:
            continue
        messages[i] = {**messages[i], "content": digest}
        n_compressed += 1
    if n_compressed:
        actions.append(f"compress_old_tool_bodies:{n_compressed}")

    return actions


def _shrink_newest_read(
    case: Case, messages: list[dict], tool_calls_by_id: dict[str, dict]
) -> str | None:
    """Re-execute the newest successful read_evidence tool call with a smaller
    max_lines and replace its tool message in place. Returns an action label
    or None when nothing was shrinkable.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        call = tool_calls_by_id.get(msg.get("tool_call_id"))
        if not call or call.get("name") != "read_evidence":
            continue
        try:
            obs = json.loads(msg.get("content") or "")
        except (ValueError, TypeError):
            continue
        if not isinstance(obs, dict) or obs.get("status") != "ok" or not obs.get("lines"):
            continue
        current_lines = len(obs["lines"])
        new_max = max(_MIN_PAGED_LINES, current_lines // 2)
        if new_max >= current_lines:
            continue
        args = dict(call.get("arguments") or {})
        args["max_lines"] = new_max
        new_obs = execute_tool(case, "read_evidence", args)
        messages[i] = {
            **msg,
            "content": _tool_result_to_text(new_obs),
        }
        return f"shrink_read_range:{obs['evidence_id']}:{current_lines}->{new_max} lines"
    return None


def run_agent(
    case: Case,
    runtime: RuntimeConfig | None = None,
    provider: LLMProvider | None = None,
    max_steps: int | None = None,
) -> AgentRunResult:
    """Run one diagnostic conversation, non-streaming, with the two tools.

    max_steps defaults to ``DEMO_A_AGENT_MAX_STEPS``. The initial case intro
    plus the evolving evidence are carried in the message history and returned
    in steps for inspection; each step records its tool observations and
    request sizing so callers can audit the loop.

    Gates, in order:
    1. identity gate — a case carrying identity-refused materials never starts;
    2. total request budget — before EVERY provider call, the full prepared
       request (system prompt + tool definitions + all messages) is measured;
       if it exceeds DEMO_A_MAX_REQUEST_CHARS, deterministic compaction runs
       in a fixed order and the request is re-measured. If even the minimal
       content exceeds the budget, the run stops with
       request_budget_exceeded and the provider is NOT called.
    """
    _identity_gate(case)

    runtime = runtime or load_runtime_config()
    provider = provider or LLMProvider(runtime=runtime)
    max_steps = max_steps or (runtime.get_int("DEMO_A_AGENT_MAX_STEPS") or 8)
    max_request_chars = (
        runtime.get_int("DEMO_A_MAX_REQUEST_CHARS") or DEFAULT_MAX_REQUEST_CHARS
    )
    # The gate measures the payload the provider would send for THIS runtime;
    # when the default provider is used these are exactly its values.
    payload_model = runtime.get("DEMO_A_LLM_MODEL") or DEFAULTS["DEMO_A_LLM_MODEL"]
    payload_max_tokens = runtime.get_int("DEMO_A_LLM_MAX_OUTPUT_TOKENS") or int(
        DEFAULTS["DEMO_A_LLM_MAX_OUTPUT_TOKENS"]
    )

    start_ts = time.time()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _compact_case_intro(case)},
    ]
    steps: list[dict] = []
    gate_events: list[dict] = []
    tool_calls_by_id: dict[str, dict] = {}
    provider_calls = 0
    total_usage: dict | None = None
    seen_calls: list[str] = []
    final_text = ""
    status = "completed"
    stop_reason = "final_answer"

    for step_no in range(1, max_steps + 1):
        tool_defs_size = _char_size(TOOL_DEFINITIONS)
        system_size = _char_size(SYSTEM_PROMPT)
        # The reserved finalization step: NO tools are offered and a generic
        # instruction demands the final answer from the collected evidence.
        is_final_step = step_no == max_steps
        step_tools = None if is_final_step else TOOL_DEFINITIONS
        if is_final_step:
            messages.append({"role": "user", "content": FINALIZATION_INSTRUCTION})

        # ---- Total request budget gate (BEFORE the provider call) ----
        prepared_chars = _total_request_chars(
            messages, payload_model, payload_max_tokens, tools=step_tools
        )
        gate_record: dict = {
            "step": step_no,
            "limit_chars": max_request_chars,
            "prepared_chars": prepared_chars,
            "actions": [],
            "finalization": is_final_step,
        }
        if prepared_chars > max_request_chars:
            for _pass in range(_MAX_COMPACTION_PASSES):
                actions = _compact_messages(messages)
                shrunk = _shrink_newest_read(case, messages, tool_calls_by_id)
                if shrunk:
                    actions.append(shrunk)
                gate_record["actions"].extend(actions)
                prepared_chars = _total_request_chars(messages, payload_model, payload_max_tokens)
                if prepared_chars <= max_request_chars:
                    break
                if not actions:
                    # A pass with no possible action cannot make room; further
                    # passes would spin without changing anything.
                    break
            gate_record["prepared_chars_after"] = prepared_chars
            if prepared_chars > max_request_chars:
                gate_record["provider_called"] = False
                gate_events.append(gate_record)
                status = "request_budget_exceeded"
                stop_reason = (
                    "request_budget_exceeded: "
                    f"prepared request of {prepared_chars} chars still exceeds "
                    f"DEMO_A_MAX_REQUEST_CHARS={max_request_chars} after compaction; "
                    "provider not called"
                )
                break
        gate_record["provider_called"] = True
        gate_events.append(gate_record)

        try:
            resp = provider.chat(messages, tools=step_tools)
        except Exception as exc:  # ProviderError or unexpected
            status = "provider_error"
            stop_reason = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
            gate_record["provider_called"] = False
            gate_record["provider_error"] = stop_reason
            break
        provider_calls += 1

        usage = resp.get("usage") or {}
        if usage:
            total_usage = {
                "prompt_tokens": (total_usage or {}).get("prompt_tokens", 0) + usage.get("prompt_tokens", 0),
                "completion_tokens": (total_usage or {}).get("completion_tokens", 0) + usage.get("completion_tokens", 0),
                "total_tokens": (total_usage or {}).get("total_tokens", 0) + usage.get("total_tokens", 0),
            }

        assistant_msg = resp["message"]
        messages.append(assistant_msg)

        if is_final_step:
            # Reserved finalization: take the text answer; never execute tools
            # (none were offered) and never spend an extra provider call.
            final_text = (assistant_msg.get("content") or "").strip()
            if not final_text:
                # Honest deterministic fallback: report insufficiency with the
                # refs already collected; NEVER fabricate a diagnosis.
                refs = _collected_evidence_refs(steps)
                fallback = "证据不足：最后一步未生成文本总结。"
                if refs:
                    fallback += "已收集证据引用：" + "、".join(refs) + "。"
                fallback += "缺失材料：无法确定。"
                final_text = fallback
            stop_reason = "final_answer"
            status = "completed"
            steps.append({
                "step": step_no,
                "finish_reason": resp.get("finish_reason"),
                "finalization": True,
                "ignored_tool_calls": len(resp.get("tool_calls") or []),
                "tool_calls": [],
                "observations": [],
                "sizing": _split_request_sizing(messages, system_size, tool_defs_size, usage),
                "request_gate": gate_record,
            })
            break

        tool_calls = resp["tool_calls"]
        for tc in tool_calls:
            tool_calls_by_id[tc["id"]] = {"name": tc["name"], "arguments": tc["arguments"]}

        if not tool_calls:
            final_text = assistant_msg.get("content") or ""
            stop_reason = "final_answer"
            status = "completed"
            break

        step_record = {
            "step": step_no,
            "finish_reason": resp.get("finish_reason"),
            "tool_calls": [{"id": tc["id"], "name": tc["name"], "arguments": tc["arguments"]}
                           for tc in tool_calls],
            "observations": [],
            "sizing": _split_request_sizing(messages, system_size, tool_defs_size, usage),
            "request_gate": gate_record,
        }

        progress_made = False
        for tc in tool_calls:
            call_fingerprint = f"{tc['name']}:{json.dumps(tc['arguments'], ensure_ascii=False, sort_keys=True)}"
            if tc["arguments_parse_error"]:
                observation = {
                    "status": "tool_error",
                    "error_type": "arguments_not_json",
                    "message": "tool call arguments did not parse as JSON",
                    "arguments_raw": tc["arguments_raw"],
                }
            elif call_fingerprint in seen_calls:
                observation = {
                    "status": "tool_error",
                    "error_type": "duplicate_call",
                    "message": "this exact tool call was already made; no new evidence",
                    "arguments": tc["arguments"],
                    "duplicate": True,
                    "do_not_repeat": True,
                    "next_action": "use existing evidence or produce final answer",
                }
            else:
                observation = execute_tool(case, tc["name"], tc["arguments"])
                if observation.get("status") == "ok":
                    progress_made = True
                seen_calls.append(call_fingerprint)

            step_record["observations"].append({
                "call_id": tc["id"],
                "name": tc["name"],
                "observation": observation,
            })
            # Feed back the tool result as an explicit tool message.
            result_text = _truncate_tool_result_payload(observation)
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result_text}
            )

        # Tool results are counted separately from the prior messages.
        tool_results_size = _char_size(
            [m for m in messages if m.get("role") == "tool"][-len(tool_calls):]
        )
        step_record["sizing"]["tool_results_chars"] = tool_results_size
        steps.append(step_record)

        if not progress_made:
            status = "no_progress"
            stop_reason = "all tool calls this step repeated prior calls or failed args"
            break

    elapsed = time.time() - start_ts

    if status == "completed" and step_no >= max_steps and final_text == "":
        status = "max_steps"
        stop_reason = f"reached max_steps={max_steps} without a final answer"

    # Keep only the most recent assistant/tool turns in request_sizing for
    # audit clarity; the full conversation is in steps.
    return AgentRunResult(
        case_id=case.case_id,
        status=status,
        final_text=final_text,
        steps=steps,
        stop_reason=stop_reason,
        elapsed_seconds=round(elapsed, 3),
        usage=total_usage,
        request_sizing={
            "system_prompt_chars": _char_size(SYSTEM_PROMPT),
            "tool_defs_chars": _char_size(TOOL_DEFINITIONS),
            "num_steps": len(steps),
            "max_request_chars": max_request_chars,
            "provider_calls": provider_calls,
            "gate_events": gate_events,
        },
    )


def run_result_to_dict(result: AgentRunResult) -> dict:
    return {
        "case_id": result.case_id,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "usage": result.usage,
        "request_sizing": result.request_sizing,
        "final_text": result.final_text,
        "steps": result.steps,
    }


__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_DEFINITIONS",
    "execute_tool",
    "run_agent",
    "run_result_to_dict",
    "AgentRunResult",
]
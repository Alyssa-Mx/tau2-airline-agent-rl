"""Lean harness：只改客服模型"看到的输入"，不动权重、不动判分。tau2 `LLMAgent` 的子类。

tau2 轨迹里记录的仍是原始工具返回和模型最终那次输出，所以判分不受影响。四项上下文精简：

1. 工具渐进式披露（progressive disclosure）
   请求里的 `tools=` 始终是完整定义（vLLM 的工具解析器要靠 schema 判参数类型，删了嵌套参数会被当成字符串）；
   只改模型看到的渲染：通过 chat_template_kwargs.expanded_tools 告诉模板哪些工具渲染完整定义。
   默认 8 个读工具完整、6 个写库工具只渲染"名字 + 一句描述"（存根）。
   模型调用了存根工具 → 丢弃这次生成（不进状态、不进轨迹、不占步数），展开该工具后重新生成。
   每轮开始时展开集合重置，所以历史里永远是存根。固定开销 5040 → 3208 token。
2. 工具返回无损压缩：进入 agent 状态前改写 —— 航班搜索每班一行，其余去掉 JSON 引号和空字段。不是摘要，没有模型参与。
3. 历史对象只保留最新状态：发给模型前（不改状态），除最近 2 条、以及"每个预订号 / 用户号 / 航线+日期 各保留最近一次"
   之外的工具返回，换成一句 14 token 的占位 `[older <tool> result omitted; call it again if needed]`。
4. 历史里的刷屏回复（整轮 "!!!!"）换成一句占位，打断"刷一次就停不下来"。

思考模式的两层兜底（Qwen3.5 原生思考，enable_thinking 由 agent-llm-args 里的 chat_template_kwargs 传入）：
  - LEAN_THINK_CLOSE_CONTINUE=1：思考没写 </think> 就 EOS、正文和工具调用都空时，把它自己的思考补上 </think>，
    在同一轮续写（continue_final_message，不重新采样）。定版评测 164/164 次续写成功。
  - LEAN_EMPTY_RETRY=N：续写后仍空，用相同参数重采，最多 N 次（最后的保险）。
  服务端另需 qwen3_toolend 解析器插件（qwen3_toolend_reasoning_parser.py）：思考里直接写出的 <tool_call> 视为思考结束。

开关（1 开 0 关；定版 = 全开）：
  LEAN_TOOL_DEFS  LEAN_COMPACT_RESULTS  LEAN_MASK_OLDER  LEAN_DROP_FLOODS  LEAN_REGEN_FLOOD_FALLBACK
  LEAN_THINK_CLOSE_CONTINUE  LEAN_EMPTY_RETRY=3  LEAN_KEEP_RECENT=2  LEAN_MAX_EXPANSIONS=2  LEAN_LOG_DIR=<审计日志目录>
  全部关掉（LEAN_*=0）时与 tau2 原版 llm_agent 调用逻辑相同，用作同引擎对照组。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, Message, MultiToolMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate

AGENT_NAME = "lean_agent"
WRITE_PREFIXES = ("book_", "cancel_", "update_", "send_", "exchange_", "return_", "modify_")
_LOCK = threading.Lock()
_on = lambda k, d: os.environ.get(k, d) == "1"  # noqa: E731


def _tool_name(t: Tool) -> str:
    n = getattr(t, "name", None)
    if n:
        return str(n)
    sch = t.openai_schema
    return str(sch.get("function", sch).get("name"))


# ---------------- 2. 工具返回无损压缩 ----------------
def _hm(t) -> str:
    return re.sub(r"^(\d\d:\d\d):\d\d", r"\1", t or "")


def _leg(f: dict) -> str:
    s = f.get("available_seats") or {}
    p = f.get("prices") or {}
    cab = " ".join(f"{c}:{s.get(c)}@${p.get(c)}" for c in ("basic_economy", "economy", "business") if c in s or c in p)
    stt = "" if f.get("status") == "available" else f" [{f.get('status')}]"
    d = f" {f['date']}" if f.get("date") else ""
    return f"{f.get('flight_number')} {f.get('origin')}>{f.get('destination')}{d} {_hm(f.get('scheduled_departure_time_est'))}-{_hm(f.get('scheduled_arrival_time_est'))}{stt} {cab}"


def _generic(o) -> str:
    if isinstance(o, dict):
        return "{" + ", ".join(f"{k}: {_generic(v)}" for k, v in o.items() if v not in (None, "", [], {})) + "}"
    if isinstance(o, list):
        return "[" + "; ".join(_generic(x) for x in o) + "]"
    return str(o)


def compact_result(name: str, content: str) -> str:
    """search_direct_flight 509 → 182 token（−64%）；search_onestop_flight 1103 → 457（−59%）；其余去引号和空字段。"""
    try:
        o = json.loads(content)
    except Exception:
        return content
    if name == "search_direct_flight" and isinstance(o, list):
        return "\n".join(_leg(f) for f in o) if o else "no flights found"
    if name == "search_onestop_flight" and isinstance(o, list):
        return "\n".join(" + ".join(_leg(f) for f in pair) for pair in o) if o else "no flights found"
    if isinstance(o, (dict, list)):
        return _generic(o)
    return content


def is_degenerate(text: Optional[str]) -> bool:
    """整轮 "!!!!"（token id 0 刷满）。开了 vLLM 重复检测后刷屏在 32 个 "!" 处被截断，门槛取 24。"""
    c = (text or "").strip()
    return len(c) >= 24 and c.count("!") / len(c) > 0.9


class LeanState(LLMAgentState):
    conv_id: str = ""
    tool_meta: Dict[str, dict] = {}  # tool_call_id -> {"name", "key"}


class LeanAgent(LLMAgent[LeanState]):
    def __init__(self, tools: List[Tool], domain_policy: str, llm: str, llm_args: Optional[dict] = None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        names = [_tool_name(t) for t in tools]
        self.write_names = {n for n in names if n.startswith(WRITE_PREFIXES)}
        self.base_expanded = sorted(n for n in names if n not in self.write_names)   # 读工具始终完整
        self.tool_defs = _on("LEAN_TOOL_DEFS", "1")
        self.compact_results = _on("LEAN_COMPACT_RESULTS", "1")
        self.mask_older = _on("LEAN_MASK_OLDER", "1")
        self.drop_floods = _on("LEAN_DROP_FLOODS", "1")
        self.regen_flood_fallback = _on("LEAN_REGEN_FLOOD_FALLBACK", "0")
        self.think_close_continue = _on("LEAN_THINK_CLOSE_CONTINUE", "0")
        self.empty_retry = int(os.environ.get("LEAN_EMPTY_RETRY", "0"))
        self.keep_recent = int(os.environ.get("LEAN_KEEP_RECENT", "2"))
        self.max_expansions = int(os.environ.get("LEAN_MAX_EXPANSIONS", "2"))
        self.log_dir = os.environ.get("LEAN_LOG_DIR") or None
        self._schemas = {_tool_name(t): (t.openai_schema.get("function", t.openai_schema).get("parameters") or {}) for t in tools}
        logger.info(f"[lean] tool_defs={self.tool_defs} compact_results={self.compact_results} "
                    f"mask_older={self.mask_older} drop_floods={self.drop_floods}")

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> LeanState:
        base = super().get_init_state(message_history)
        return LeanState(system_messages=base.system_messages, messages=base.messages, conv_id=uuid.uuid4().hex[:12])

    # ---- 1. 工具定义：告诉模板哪些工具渲染完整定义 ----
    def _args_for(self, expanded: List[str]) -> dict:
        args = deepcopy(self.llm_args)
        if not self.tool_defs:
            return args
        eb = dict(args.get("extra_body") or {})
        ctk = dict(eb.get("chat_template_kwargs") or {})
        ctk["expanded_tools"] = list(expanded)
        eb["chat_template_kwargs"] = ctk
        args["extra_body"] = eb
        return args

    # ---- 2. 工具返回进入状态前压缩，并记下它属于哪个实体（用于第 3 项的"钉住"） ----
    def _ingest_tool(self, tm: ToolMessage, state: LeanState) -> ToolMessage:
        name, key = "tool", None
        for m in reversed(state.messages):
            if isinstance(m, AssistantMessage) and m.tool_calls:
                hit = next((c for c in m.tool_calls if c.id == tm.id), None)
                if hit:
                    name = hit.name
                    a = hit.arguments if isinstance(hit.arguments, dict) else {}
                    key = a.get("reservation_id") or a.get("user_id")
                    # 航班搜索既没有预订号也没有用户号；用户是从这次搜索里选的航班、后面订票要照抄航班号
                    # → 按"工具 + 航线 + 日期"钉住最近一次搜索
                    if not key and name.startswith("search_") and a.get("origin") and a.get("destination"):
                        key = f"{name}:{a.get('origin')}>{a.get('destination')}@{a.get('date')}"
                    break
        if not key:
            mm = re.search(r'"reservation_id": "(\w+)"', tm.content or "")
            key = mm.group(1) if mm else None
        state.tool_meta[tm.id] = {"name": name, "key": key}
        if not self.compact_results:
            return tm
        new = deepcopy(tm)
        new.content = compact_result(name, tm.content or "")
        return new

    # ---- 3 + 4. 发给模型的历史视图（不改状态、不改轨迹） ----
    def _view(self, state: LeanState) -> list:
        msgs = state.messages
        if not (self.mask_older or self.drop_floods):
            return msgs
        tool_idx = [i for i, m in enumerate(msgs) if isinstance(m, ToolMessage)]
        keep = set(tool_idx[-self.keep_recent:]) if self.keep_recent > 0 else set()
        latest = {}
        for i in tool_idx:
            k = state.tool_meta.get(msgs[i].id, {}).get("key")
            if k:
                latest[k] = i
        keep |= set(latest.values())
        out = []
        for i, m in enumerate(msgs):
            if self.mask_older and isinstance(m, ToolMessage) and i not in keep:
                nm = deepcopy(m)
                nm.content = f"[older {state.tool_meta.get(m.id, {}).get('name', 'tool')} result omitted; call it again if needed]"
                out.append(nm)
            elif self.drop_floods and isinstance(m, AssistantMessage) and not m.is_tool_call() and is_degenerate(m.content):
                nm = deepcopy(m)
                nm.content = "(previous reply was garbled and not delivered)"
                out.append(nm)
            else:
                out.append(m)
        return out

    def _audit(self, rec: dict) -> None:
        """每次展开 / 续写 / 重采都记一条，兜底的触发次数本身就是评测报告里的指标。"""
        if not self.log_dir:
            return
        try:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            with _LOCK, open(Path(self.log_dir) / f"lean_{os.getpid()}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as ex:
            logger.warning(f"[lean] audit failed: {ex}")

    def _args_ok(self, call) -> bool:
        """调用参数符合完整定义：必填参数都在，且没有定义里不存在的参数名。"""
        sch = self._schemas.get(call.name)
        if sch is None:
            return False
        props = sch.get("properties") or {}
        args = call.arguments if isinstance(call.arguments, dict) else {}
        return set(sch.get("required") or []) <= set(args) and (not props or set(args) <= set(props))

    def _close_and_continue(self, msg, messages, expanded, state):
        """思考没写 </think> 就结束、正文和工具调用都空：把它自己的思考补上 </think> 在同一轮续写（不重新采样）。"""
        if not self.think_close_continue or (msg.content and msg.content.strip()) or msg.tool_calls:
            return msg
        raw = getattr(msg, "raw_data", None) or {}
        try:
            m0 = (raw.get("choices") or [{}])[0].get("message") or {}
        except AttributeError:
            m0 = {}
        reasoning = (m0.get("reasoning_content") or m0.get("reasoning") or "").strip("\n")
        if not reasoning:
            self._audit({"conv_id": state.conv_id, "event": "think_close_no_reasoning"})
            return msg
        args = self._args_for(expanded)
        eb = dict(args.get("extra_body") or {})
        ctk = dict(eb.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = False                      # 思考已由我们闭合
        eb.update(chat_template_kwargs=ctk, continue_final_message=True, add_generation_prompt=False)
        eb.pop("thinking_token_budget", None)
        args["extra_body"] = eb
        prefill = AssistantMessage(role="assistant", content="<think>\n" + reasoning + "\n</think>\n\n")
        cont = generate(model=self.llm, tools=self.tools, messages=list(messages) + [prefill], call_name="agent_response", **args)
        ok = bool((cont.content and cont.content.strip()) or cont.tool_calls)
        self._audit({"conv_id": state.conv_id, "event": "think_close_continue", "ok": ok, "reasoning_chars": len(reasoning),
                     "n_calls": len(cont.tool_calls or []), "content_head": (cont.content or "")[:160]})
        return cont if ok else msg

    def _generate_next_message(self, message, state: LeanState) -> AssistantMessage:
        if isinstance(message, UserMessage) and message.is_audio:
            raise ValueError("User message cannot be audio. Use VoiceLLMAgent instead.")
        if isinstance(message, MultiToolMessage):
            state.messages.extend(self._ingest_tool(t, state) for t in message.tool_messages)
        elif isinstance(message, ToolMessage):
            state.messages.append(self._ingest_tool(message, state))
        else:
            state.messages.append(message)
        messages = list(state.system_messages) + self._view(state)
        callable_now = set(self.base_expanded)
        expanded = list(self.base_expanded)
        attempts, msgs = [], []
        t0 = time.perf_counter()
        for k in range(self.max_expansions + 1):
            msg = generate(model=self.llm, tools=self.tools, messages=messages, call_name="agent_response", **self._args_for(expanded))
            msg = self._close_and_continue(msg, messages, expanded, state)
            for r in range(self.empty_retry):
                if (msg.content and msg.content.strip()) or msg.tool_calls:
                    break
                self._audit({"conv_id": state.conv_id, "event": "empty_retry", "try": r + 1})
                msg = generate(model=self.llm, tools=self.tools, messages=messages, call_name="agent_response", **self._args_for(expanded))
                msg = self._close_and_continue(msg, messages, expanded, state)
            if self.empty_retry and not (msg.content and msg.content.strip()) and not msg.tool_calls:
                self._audit({"conv_id": state.conv_id, "event": "empty_retry_exhausted"})
            msgs.append(msg)
            calls = list(msg.tool_calls or [])
            hidden = [c.name for c in calls if c.name not in callable_now] if self.tool_defs else []
            attempts.append({"expanded": sorted(set(expanded) & self.write_names),
                             "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in calls],
                             "hidden": hidden, "content_head": (msg.content or "")[:160],
                             "prompt_tokens": msg.usage.get("prompt_tokens") if isinstance(msg.usage, dict) else None})
            if not hidden or k == self.max_expansions:
                break
            # 调了存根工具：展开它，丢弃这次生成，重新生成
            for h in hidden:
                if h not in expanded:
                    expanded.append(h)
                callable_now.add(h)
        if self.regen_flood_fallback and len(msgs) > 1 and not msg.tool_calls and is_degenerate(msg.content):
            # 展开后重生成的那一轮刷屏：退回最近一次没刷屏、且参数符合完整定义的调用（不多调模型）；没有就保留
            prev = next((m for m in reversed(msgs[:-1]) if m.tool_calls and all(self._args_ok(c) for c in m.tool_calls)), None)
            self._audit({"conv_id": state.conv_id, "event": "regen_flood_fallback", "used_earlier_call": prev is not None,
                         "prompt_tokens": attempts[-1]["prompt_tokens"]})
            if prev is not None:
                msg = prev
        if len(attempts) > 1:
            self._audit({"conv_id": state.conv_id, "n_attempts": len(attempts), "seconds": round(time.perf_counter() - t0, 2),
                         "final_still_hidden": bool(attempts[-1]["hidden"]), "attempts": attempts})
        return msg


def create_lean_agent(tools, domain_policy, **kwargs):
    return LeanAgent(tools=tools, domain_policy=domain_policy, llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"))


def register(registry, name: str = AGENT_NAME) -> None:
    if name in registry.get_agents():
        return
    registry.register_agent_factory(create_lean_agent, name)

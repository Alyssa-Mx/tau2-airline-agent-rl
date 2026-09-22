"""verl 0.9 × τ²-bench 的多轮 agent loop：在 GRPO rollout 里原样跑 tau2 的整套仿真。

设计：tau2 的编排器（用户模拟器、工具执行、终止判定、判分）一行不改，只把"客服 LLM"换成 verl 的采样服务。
每条 rollout = 一整段多轮对话，拼成一条 token 序列：
    assistant 生成的 token → response_mask = 1（进 loss）
    user / tool 消息的 token → response_mask = 0（只作上下文）

三个训练臂由环境变量切换（其余逐项相同）：
    ARM=binary   官方 0/1：被截断的对话一律 0
    ARM=partial  TAU2_REWARD_V3=1      completion-aware 0 / 0.5 / 1（tau2_reward.py）
    ARM=mask     TAU2_MASK_OVERLONG=1  DAPO 式超长过滤：被截断的对话 response_mask 全置 0，不进 loss、不进组统计
                                       （必须配 adv_estimator=grpo_masked，见 tau2_masked_adv.py）
    partial 与 mask 是同一批对话上的两种相反处方，同时开会把拿 0.5 的对话一起 mask 掉 → 启动即报错。

截断与结束分类（写进 extra_fields.kill_reason）：
    budget        整段对话的生成预算（response_length，16384）用完
    assist_turns  客服轮数到上限（30）
    repeat        同一句客服回复累计 TAU2_REPEAT_KILL 次（8）—— 复读死循环
    以上三种都按 tau2 的 max_steps（撞步数）收尾，与撞 40 步同样处理。
    网关 / 基础设施异常，或 tau2 自己记成 user_error / infrastructure_error / unexpected_error
      → masked：response_mask 全 0、不进 loss（这不是策略的错）。
    模型自己的问题（例如空输出）→ 0 分，不 mask。

每条 rollout 全量落盘到 $TAU2_ROLLOUT_DIR（完整消息、结束原因、判分细项、耗时），由单独的写线程完成。

环境变量：TAU2_DATA_DIR、TAU2_DOMAIN、TAU2_USER_LLM、TAU2_USER_API_BASE、TAU2_USER_API_KEY、TAU2_USER_TEMP、
TAU2_MAX_STEPS、TAU2_MAX_ERRORS、TAU2_THREADS、TAU2_TURN_MAX_TOKENS、TAU2_REPEAT_KILL、
TAU2_REWARD_V3、TAU2_PARTIAL、TAU2_MASK_OVERLONG、TAU2_ROLLOUT_DIR。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.rollout_trace import rollout_trace_op

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import AssistantMessage, MultiToolMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import TerminationReason, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.build import _build_env_kwargs, build_environment, build_user
from tau2.runner.helpers import get_tasks
from tau2.runner.simulation import run_simulation
from tau2_reward import completion_aware_reward

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TURN_MAX_TOKENS = int(os.getenv("TAU2_TURN_MAX_TOKENS", "0"))
REPEAT_KILL = int(os.getenv("TAU2_REPEAT_KILL", "0"))
REWARD_V3 = os.getenv("TAU2_REWARD_V3", "0") == "1"
MASK_OVERLONG = os.getenv("TAU2_MASK_OVERLONG", "0") == "1"
if MASK_OVERLONG and REWARD_V3:
    raise RuntimeError("TAU2_MASK_OVERLONG 与 TAU2_REWARD_V3 不能同时开：超长 mask 会把拿部分分的对话一起丢掉")
_OVERLONG_KILL = ("budget", "assist_turns", "repeat")
PARTIAL = float(os.getenv("TAU2_PARTIAL", "0.5"))
ROLLOUT_DIR = os.getenv("TAU2_ROLLOUT_DIR", "")

# tau2 自己把这些结束原因记成"用户侧 / 基础设施出错"，不是客服模型的问题 → mask
MASK_TERMINATIONS = frozenset({"user_error", "infrastructure_error", "unexpected_error"})
_INFRA_EXC_HINTS = ("Timeout", "Connection", "RateLimit", "ServiceUnavailable", "APIError", "InternalServerError",
                    "BadGateway", "APIStatusError", "Overloaded")

_TASK_CACHE: dict[str, Any] = {}
_TASK_LOCK = threading.Lock()
_DUMP_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _load_task(domain: str, task_id: str):
    with _TASK_LOCK:
        if domain not in _TASK_CACHE:
            _TASK_CACHE[domain] = {t.id: t for t in get_tasks(domain)}
    return _TASK_CACHE[domain][str(task_id)]


class AgentEmptyOutput(RuntimeError):
    """模型一个 token 都没吐（只有 EOS / 空白）。tau2 的 validate 会抛 ValueError；按 tau2 语义这条 sim 算 agent 出错、0 分。"""


class BudgetExceeded(RuntimeError):
    """生成预算 / 客服轮数用完：按"步数用尽"收尾，与撞 40 步同样处理。"""


class RepeatLoop(RuntimeError):
    """同一句客服回复累计 TAU2_REPEAT_KILL 次：判定为复读死循环，按"步数用尽"收尾。"""


def _is_infra_exception(e: BaseException) -> bool:
    mod = type(e).__module__ or ""
    return mod.startswith(("litellm", "openai", "httpx", "httpcore", "requests", "urllib3", "aiohttp")) or \
        any(h in type(e).__name__ for h in _INFRA_EXC_HINTS)


def _to_chat_dicts(messages) -> list[dict]:
    """tau2 消息 → chat template 用的 dict。tool_call.arguments 保持 dict（Qwen3.5 模板要 |items）。"""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content or ""})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content or ""})
        elif isinstance(m, AssistantMessage):
            d: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.is_tool_call():
                d["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": dict(tc.arguments or {})}}
                                   for tc in m.tool_calls]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content or "", "tool_call_id": m.id})
    return out


class VerlTokenAgent(LLMAgent):
    """tau2 的 LLMAgent，但生成走 verl 采样服务，并逐段记录 token 与 response_mask。"""

    def __init__(self, tools, domain_policy, loop: "Tau2AgentLoop", sampling_params: dict, request_id: str):
        super().__init__(tools=tools, domain_policy=domain_policy, llm="verl", llm_args={})
        self._loop = loop
        self._sampling = sampling_params
        self._request_id = request_id
        self._tool_schema_objs = [OpenAIFunctionToolSchema(**t.openai_schema) for t in tools]
        self._tool_schema_dicts = [t.openai_schema for t in tools]
        self.prompt_ids: list[int] = []
        self.n_prompt: int = 0          # 初始 prompt 长度（system + tools + 开场白）
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.assistant_turns = 0
        self.gen_secs = 0.0
        self.kill_reason: Optional[str] = None
        self._seen_texts: Counter = Counter()

    def _await(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop.loop).result()

    def _generate_next_message(self, message, state: LLMAgentState) -> AssistantMessage:
        new_msgs = message.tool_messages if isinstance(message, MultiToolMessage) else ([message] if message is not None else [])
        state.messages.extend(new_msgs)
        if not self.prompt_ids:
            ids = self._await(self._loop.apply_chat_template(_to_chat_dicts(state.system_messages + state.messages), tools=self._tool_schema_dicts))
            self.prompt_ids = list(ids); self.n_prompt = len(ids)
        else:
            # 只追加新消息的 token（环境侧，mask=0）；模型生成过的 token 原样保留，保证 logprob 与训练序列逐 token 对齐
            ids = self._await(self._loop.apply_chat_template(_to_chat_dicts(new_msgs), remove_system_prompt=True))
            ids = list(self._loop.turn_separator) + list(ids)
            self.prompt_ids += ids; self.response_mask += [0] * len(ids)
            if self.response_logprobs:
                self.response_logprobs += [0.0] * len(ids)
        remaining = self._loop.response_length - len(self.response_mask)
        if remaining <= 0:
            self.kill_reason = "budget"
            raise BudgetExceeded(f"response budget {self._loop.response_length} exhausted")
        if self._loop.max_assistant_turns and self.assistant_turns >= self._loop.max_assistant_turns:
            self.kill_reason = "assist_turns"
            raise BudgetExceeded("max_assistant_turns reached")
        sampling = self._sampling
        if TURN_MAX_TOKENS > 0:
            sampling = dict(self._sampling); sampling["max_tokens"] = max(1, min(TURN_MAX_TOKENS, remaining))
        t0 = time.time()
        out = self._await(self._loop.server_manager.generate(
            request_id=self._request_id, prompt_ids=self.prompt_ids, sampling_params=sampling))
        self.gen_secs += time.time() - t0
        self.assistant_turns += 1
        self.prompt_ids += list(out.token_ids); self.response_mask += [1] * len(out.token_ids)
        if out.log_probs:
            self.response_logprobs += list(out.log_probs)
        text, calls = self._await(self._loop.tool_parser.extract_tool_calls(list(out.token_ids), self._tool_schema_objs))
        tool_calls = []
        for c in calls:
            try:
                args = json.loads(c.arguments) if isinstance(c.arguments, str) else dict(c.arguments)
            except Exception:
                args = {"_raw": c.arguments}
            tool_calls.append(ToolCall(id=getattr(c, "tool_call_id", None) or f"call_{uuid4().hex[:8]}", name=c.name, arguments=args))
        content = (text.strip() or None) if text else None
        if content is None and not tool_calls:
            # 空消息过不了 tau2 的 validate，ValueError 会一路抛到 TaskRunner 把训练打断 → 这里显式收尾
            raise AgentEmptyOutput(f"empty assistant output ({len(out.token_ids)} tokens)")
        if REPEAT_KILL > 0 and content:
            self._seen_texts[content] += 1
            if self._seen_texts[content] >= REPEAT_KILL:
                self.kill_reason = "repeat"
                raise RepeatLoop(f"same assistant reply repeated {REPEAT_KILL} times")
        return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls or None)


def _dump_record(path: str, record: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # 落盘失败不许打断训练
        logger.warning(f"[tau2_agent] rollout dump failed: {e}")


@register("tau2_agent")
class Tau2AgentLoop(AgentLoopBase):
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = mt.max_assistant_turns
        self.response_length = self.rollout_config.response_length
        self.tool_parser = ToolParser.get_tool_parser(mt.format, self.tokenizer)
        self.domain = os.environ.get("TAU2_DOMAIN", "airline")
        self.user_llm = os.environ.get("TAU2_USER_LLM", "openai/deepseek-v4-flash")
        self.user_llm_args = {"temperature": float(os.environ.get("TAU2_USER_TEMP", "0"))}
        if os.environ.get("TAU2_USER_API_BASE"):
            self.user_llm_args["api_base"] = os.environ["TAU2_USER_API_BASE"]
            self.user_llm_args["api_key"] = os.environ.get("TAU2_USER_API_KEY", "dummy")
        self.max_steps = int(os.environ.get("TAU2_MAX_STEPS", "40"))
        self.max_errors = int(os.environ.get("TAU2_MAX_ERRORS", "10"))
        if Tau2AgentLoop._executor is None:
            Tau2AgentLoop._executor = ThreadPoolExecutor(max_workers=int(os.environ.get("TAU2_THREADS", "256")))
        self.dump_path = ""
        if ROLLOUT_DIR:
            os.makedirs(ROLLOUT_DIR, exist_ok=True)
            self.dump_path = os.path.join(ROLLOUT_DIR, f"rollouts_{socket.gethostname()}_{os.getpid()}.jsonl")

    def _run_sim(self, task, sampling_params: dict, request_id: str, seed: int):
        cfg = TextRunConfig(domain=self.domain, agent="llm_agent", user="user_simulator", llm_agent="verl", llm_args_agent={},
                            llm_user=self.user_llm, llm_args_user=self.user_llm_args,
                            max_steps=self.max_steps, max_errors=self.max_errors, seed=seed)
        env_kwargs = _build_env_kwargs(cfg, task)
        environment = build_environment(self.domain, solo_mode=False, env_kwargs=env_kwargs)
        agent = VerlTokenAgent(environment.get_tools(), environment.get_policy(), self, sampling_params, request_id)
        agent.masked = False; agent.sim_error = None
        user = build_user("user_simulator", environment, task, llm=self.user_llm, llm_args=self.user_llm_args, persona_config=None, solo_mode=False)
        orch = Orchestrator(domain=self.domain, agent=agent, user=user, environment=environment, task=task,
                            max_steps=self.max_steps, max_errors=self.max_errors, seed=seed, solo_mode=False, simulation_id=request_id)
        try:
            sim = run_simulation(orch, evaluation_type=EvaluationType.ALL, env_kwargs=env_kwargs)
        except (BudgetExceeded, RepeatLoop) as e:
            # 预算 / 轮数上限 / 复读：按"步数用尽"收尾，已有轨迹照常交给评测器
            logger.warning(f"[tau2_agent] task {task.id}: {agent.kill_reason}: {e}")
            orch.done = True; orch.termination_reason = TerminationReason.MAX_STEPS
            sim = orch._finalize(); sim.policy = environment.get_policy()
            try:
                sim.reward_info = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain=self.domain, env_kwargs=env_kwargs)
            except Exception as ee:
                logger.warning(f"[tau2_agent] evaluate after {agent.kill_reason} failed: {ee}"); sim.reward_info = None
        except Exception as e:
            # 其余异常不许打断训练：收尾、0 分、记类型；网关 / 基础设施类另外 mask（不进 loss、不进组统计）
            agent.sim_error = type(e).__name__
            agent.masked = _is_infra_exception(e)
            logger.warning(f"[tau2_agent] task {task.id}: sim error {type(e).__name__} masked={agent.masked}: {str(e)[:200]}")
            orch.done = True
            orch.termination_reason = TerminationReason.AGENT_ERROR
            try:
                sim = orch._finalize(); sim.policy = environment.get_policy()
            except Exception:
                sim = SimpleNamespace(messages=[], termination_reason=f"sim_error:{type(e).__name__}")
            sim.reward_info = None
        agent.v3_reward, agent.v3_info = None, {}
        if REWARD_V3 and not isinstance(sim, SimpleNamespace):
            try:
                agent.v3_reward, agent.v3_info = completion_aware_reward(sim, task, self.domain, env_kwargs, PARTIAL)
            except Exception as e:  # 判分出错不许打断训练：按 0 分记下原因
                logger.warning(f"[tau2_agent] reward failed on task {task.id}: {type(e).__name__}: {str(e)[:200]}")
                agent.v3_reward, agent.v3_info = 0.0, {"error": type(e).__name__}
        term_value = getattr(getattr(sim, "termination_reason", None), "value", str(getattr(sim, "termination_reason", "")))
        if term_value in MASK_TERMINATIONS:
            agent.masked = True
        return agent, sim

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info", {}) or {}
        task_id = str(extra.get("task_id") or kwargs.get("reward_model", {}).get("ground_truth"))
        task = _load_task(self.domain, task_id)
        request_id = uuid4().hex
        seed = int(extra.get("seed", 0)) + (hash(request_id) % 100000)
        t0 = time.time()
        agent, sim = await self.loop.run_in_executor(Tau2AgentLoop._executor, self._run_sim, task, sampling_params, request_id, seed)
        outcome = float(sim.reward_info.reward) if sim.reward_info is not None else 0.0
        outcome = 1.0 if outcome >= 1.0 - 1e-6 else 0.0          # 官方 airline 奖励是 0/1 连乘
        term = str(getattr(getattr(sim, "termination_reason", None), "value", getattr(sim, "termination_reason", "")))
        _term_norm = term.lower().replace("terminationreason.", "")
        overlong = (agent.kill_reason in _OVERLONG_KILL) or (_term_norm == "max_steps")
        masked_overlong = bool(MASK_OVERLONG and overlong)
        masked = bool(getattr(agent, "masked", False)) or masked_overlong
        v3 = float(getattr(agent, "v3_reward", None) or 0.0)
        if masked:
            reward_score = 0.0
        elif REWARD_V3:
            reward_score = v3
        else:
            reward_score = outcome
        if not agent.response_mask:
            # 一次都没生成（极端情况）：给一个空回复段，避免 verl 侧长度为 0
            agent.response_mask = [1]; agent.prompt_ids = agent.prompt_ids + [self.tokenizer.eos_token_id]
        response_ids = agent.prompt_ids[-len(agent.response_mask):]
        prompt_ids = agent.prompt_ids[: len(agent.prompt_ids) - len(agent.response_mask)]
        response_mask = agent.response_mask[: self.response_length]
        if masked:
            response_mask = [0] * len(response_mask)
        names = [tc.name for m in (sim.messages or []) if isinstance(m, AssistantMessage) and m.tool_calls for tc in m.tool_calls]
        sim_secs = time.time() - t0
        metrics = {"tau2/reward": outcome, "tau2/assistant_turns": agent.assistant_turns, "tau2/gen_secs": agent.gen_secs,
                   "tau2/sim_secs": sim_secs, "tau2/n_messages": len(sim.messages),
                   "tau2/budget_exceeded": float(agent.kill_reason == "budget"),
                   "tau2/repeat_killed": float(agent.kill_reason == "repeat"),
                   "tau2/masked": float(masked), "tau2/masked_overlong": float(masked_overlong), "tau2/overlong": float(overlong),
                   "tau2/sim_error": float(bool(getattr(agent, "sim_error", None))),
                   "tau2/reward_v3": v3, "tau2/partial_credit": float(0.0 < v3 < 1.0)}
        if self.dump_path:
            record = {"time": time.time(), "task_id": task_id, "request_id": request_id, "seed": seed,
                      "temperature": sampling_params.get("temperature"), "outcome": outcome,
                      "reward_score": reward_score, "reward_v3_on": REWARD_V3, "reward_v3": getattr(agent, "v3_reward", None),
                      "v3_info": getattr(agent, "v3_info", {}), "masked": masked, "masked_overlong": masked_overlong,
                      "overlong": overlong, "kill_reason": agent.kill_reason,
                      "sim_error": getattr(agent, "sim_error", None), "termination": term,
                      "reward_breakdown": getattr(sim.reward_info, "reward_breakdown", None) if sim.reward_info is not None else None,
                      "transfer": "transfer_to_human_agents" in names, "assistant_turns": agent.assistant_turns,
                      "n_messages": len(sim.messages), "response_tokens": len(agent.response_mask),
                      "assistant_tokens": int(sum(agent.response_mask)), "gen_secs": agent.gen_secs, "sim_secs": sim_secs,
                      "messages": [m.model_dump(mode="json") if hasattr(m, "model_dump") else str(m) for m in (sim.messages or [])]}
            await self.loop.run_in_executor(_DUMP_EXECUTOR, _dump_record, self.dump_path, record)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask,
            response_logprobs=agent.response_logprobs[: self.response_length] if agent.response_logprobs else None,
            reward_score=reward_score,
            num_turns=len(sim.messages),
            metrics=metrics,
            extra_fields={"task_id": task_id, "termination_reason": term, "kill_reason": agent.kill_reason or "",
                          "masked": masked, "masked_overlong": masked_overlong, "outcome": outcome, "reward_v3": v3,
                          "reward_breakdown": json.dumps(getattr(sim.reward_info, "reward_breakdown", None), default=str)},
        )

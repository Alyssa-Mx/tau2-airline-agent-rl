"""Completion-aware reward（0 / 0.5 / 1）：被截断的对话按题型给部分分。

只依赖 tau2；训练（tau2_agent_loop.py）和离线单测共用这一份。判分口径 = TAU2_DATA_DIR 下 tasks.json 的
reward_basis，训练时指向 τ²-bench 官方数据 → DB（数据库终态）× COMMUNICATE（该告诉用户的信息）。

| 对话怎么结束的                                   | 题型（按 gold 动作）         | 奖励 |
|--------------------------------------------------|------------------------------|------|
| 正常结束（user_stop / agent_stop）               | 任意                         | 官方判分 0 / 1 |
| 被截断（撞 40 步 / 生成预算用完 / 客服轮数用完 / 复读终止） | 写库题（gold 含写工具） | 改记正常结束重判：通过 → 0.5，否则 0 |
|                                                  | 只读题（gold 只有查询）      | 重判通过 **且** gold 查询全部调用过 → 0.5，否则 0 |
|                                                  | 无 gold 动作（应拒绝的题）   | 0：数据库本来就不该改，"判对"不代表干了活 |
| 其他（agent 出错等）                             | 任意                         | 0 |

为什么这样分（docs/01_failure_analysis.md）：把 base 的 70 条截断对话用官方判分器改记正常结束重判，
只读题 16/18 已经查完说对、只是没收尾；写库题只有 5/31 能过且全部伴随长上下文刷屏；
无 gold 题"能过"的 13/21 条里 12 条是刷屏跑飞的对话 —— 三种情况原来都被记成同一个 0。
"""
from __future__ import annotations

from typing import Optional

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

WRITE_TOOLS = frozenset({"book_reservation", "cancel_reservation", "update_reservation_flights",
                         "update_reservation_baggages", "update_reservation_passengers", "send_certificate"})
NORMAL_TERMS = frozenset({"user_stop", "agent_stop"})
PREMATURE_TERMS = frozenset({"max_steps"})   # agent loop 里的预算 / 轮数 / 复读截断也记成 max_steps


def task_type(task) -> str:
    acts = getattr(getattr(task, "evaluation_criteria", None), "actions", None) or []
    if any(a.name in WRITE_TOOLS for a in acts):
        return "write"
    return "read" if acts else "none"


def gold_actions_done(task, messages) -> bool:
    """gold 动作是否全部被调用过（名字 + compare_args 参数匹配，用 tau2 自带的 Action.compare_with_tool_call）。"""
    acts = getattr(getattr(task, "evaluation_criteria", None), "actions", None) or []
    calls = [tc for m in (messages or []) if isinstance(m, AssistantMessage) and m.tool_calls for tc in m.tool_calls]
    return all(any(a.compare_with_tool_call(tc) for tc in calls) for a in acts)


def _term_value(sim) -> str:
    t = getattr(sim, "termination_reason", None)
    return str(getattr(t, "value", t) or "")


def completion_aware_reward(sim, task, domain: str, env_kwargs: Optional[dict] = None,
                            partial: float = 0.5) -> tuple[float, dict]:
    """返回 (奖励, 说明)。sim 是 tau2 SimulationRun；正常结束但 sim.reward_info 为空时现场判分。"""
    term = _term_value(sim)
    ttype = task_type(task)
    info = {"term": term, "task_type": ttype, "premature": term in PREMATURE_TERMS}
    if term in NORMAL_TERMS:
        ri = getattr(sim, "reward_info", None)
        if ri is None:
            ri = evaluate_simulation(sim, task, EvaluationType.ALL, solo_mode=False, domain=domain, env_kwargs=env_kwargs or {})
        r = 1.0 if float(ri.reward) >= 1.0 - 1e-6 else 0.0
        info["official"] = r
        return r, info
    if term not in PREMATURE_TERMS:
        return 0.0, info
    if ttype == "none":
        info["rule"] = "no_gold_zero"
        return 0.0, info
    # 被截断：给它一次"收尾"的机会 —— 改记 user_stop 用同一个官方判分器再判
    sim2 = sim.model_copy(update={"termination_reason": TerminationReason.USER_STOP})
    ri = evaluate_simulation(sim2, task, EvaluationType.ALL, solo_mode=False, domain=domain, env_kwargs=env_kwargs or {})
    regrade_ok = float(ri.reward) >= 1.0 - 1e-6
    info["regrade_ok"] = regrade_ok
    info["regrade_breakdown"] = {str(getattr(k, "value", k)): v for k, v in (ri.reward_breakdown or {}).items()}
    ok = regrade_ok
    if ttype == "read":
        # 只读题数据库本来不变，重判"通过"不够，还要求 gold 查询真的都做过
        info["gold_done"] = gold_actions_done(task, sim.messages)
        ok = ok and info["gold_done"]
    return (float(partial) if ok else 0.0), info


# 训练时实际调用的名字（保留，便于与训练日志字段 reward_v3 对应）
reward_v3 = completion_aware_reward

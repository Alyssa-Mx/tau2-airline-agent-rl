"""grpo_masked：标准 GRPO 优势，但把 response_mask 全 0 的样本排除出组统计。DAPO 式超长过滤对照臂（ARM=mask）用。

为什么必须单独写（读 verl 0.9 源码确认）：
  verl 自带的 `grpo`（core_algos.compute_grpo_outcome_advantage）只按 uid 分组算 (r − mean)/(std + ε)，
  完全不看 response_mask。被 mask 的样本（reward 记 0）仍会压低组均值，把同组其余样本的优势算错。
  开超长过滤后每轮 21–36 条被 mask，不排除就会系统性偏。

规则（与 verl 的 grpo 逐项相同，只多两条）：
  - 被 mask 的样本（response_mask.sum == 0）：不进组统计，优势 0；
  - mask 后组里只剩 ≤1 条：优势 0（verl 对单条组会返回 score 本身，n=1 时没有对照意义）；
  - 其余：(r − mean)/(std + ε)，torch.std（无偏），ε=1e-6。

注册方式：优势在训练主进程里算，主进程只会通过
`actor_rollout_ref.rollout.agent.agent_loop_manager_class` 的全限定名导入 manager 类，
所以把 register 放在本模块、再用一个空子类指过来，保证注册发生在第一次算优势之前。
"""
from __future__ import annotations

import logging
from collections import defaultdict

import torch

from verl.experimental.agent_loop import AgentLoopManager as _BaseAgentLoopManager
from verl.trainer.ppo.core_algos import register_adv_est

logger = logging.getLogger(__file__)


def masked_scalar_advantages(scores: torch.Tensor, valid: torch.Tensor, index,
                             epsilon: float = 1e-6) -> tuple[torch.Tensor, dict]:
    """scores: (bs,) 每条序列的奖励；valid: (bs,) bool（False = 被 mask）。返回 (bs,) 标量优势 + 计数。"""
    scores = scores.detach().float()
    adv = torch.zeros_like(scores)
    groups: dict = defaultdict(list)
    for i in range(scores.shape[0]):
        if bool(valid[i]):
            groups[index[i]].append(i)
    stats = dict(groups=len(groups), with_spread=0, no_spread=0, singleton=0,
                 masked=int((~valid).sum().item()), n=int(scores.shape[0]))
    for ids in groups.values():
        if len(ids) < 2:
            stats["singleton"] += 1
            continue
        t = torch.tensor(ids)
        o = scores[t]
        if o.max() != o.min():
            stats["with_spread"] += 1
            adv[t] = (o - o.mean()) / (o.std() + epsilon)
        else:
            stats["no_spread"] += 1          # 组内全同（全对或全错）→ 优势 0，与 GRPO 一致
    return adv, stats


@register_adv_est("grpo_masked")
def compute_grpo_masked_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index=None,
                                  epsilon: float = 1e-6, config=None, **kwargs):
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        valid = response_mask.sum(dim=-1) > 0
        adv, stats = masked_scalar_advantages(scores, valid, index, epsilon=epsilon)
        nz = adv[adv.abs() > 0]
        stats["adv_abs_max"] = round(float(adv.abs().max()), 4) if adv.numel() else 0.0
        stats["adv_nonzero"] = int(nz.numel())
        # TaskRunner 里的 logger 不进 train.log，所以同时 print(flush)
        print(f"[masked_adv] {stats}", flush=True)
        logger.warning(f"[masked_adv] {stats}")
        adv = adv.to(token_level_rewards.dtype).unsqueeze(-1) * response_mask
    return adv, adv


class Tau2MaskedAgentLoopManager(_BaseAgentLoopManager):
    """行为与 verl 原版 AgentLoopManager 完全相同；唯一目的是让训练主进程导入本模块、完成上面的注册。"""

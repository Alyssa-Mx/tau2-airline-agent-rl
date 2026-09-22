"""grpo_masked 的离线单测：CPU 可跑，不起 ray、不碰 GPU。

没装 verl 的机器上用两个空模块顶替 verl 的注册接口，只测纯函数 masked_scalar_advantages。
    python rl/tests/test_masked_adv.py
"""
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
try:
    import verl  # noqa: F401
except ImportError:  # 顶替 verl 的两个导入点
    agent_loop = types.ModuleType("verl.experimental.agent_loop")
    agent_loop.AgentLoopManager = object
    core_algos = types.ModuleType("verl.trainer.ppo.core_algos")
    core_algos.register_adv_est = lambda name: (lambda f: f)
    for name, mod in {"verl": types.ModuleType("verl"), "verl.experimental": types.ModuleType("verl.experimental"),
                      "verl.experimental.agent_loop": agent_loop, "verl.trainer": types.ModuleType("verl.trainer"),
                      "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"), "verl.trainer.ppo.core_algos": core_algos}.items():
        sys.modules[name] = mod
from tau2_masked_adv import masked_scalar_advantages  # noqa: E402

fails = []


def approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def check(name, cond):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)


# 1) 混合组：与 verl GRPO 逐项相同 (r − mean)/(std + ε)，std 无偏
s = torch.tensor([1., 0., 1., 0., 1., 0.])
adv, st = masked_scalar_advantages(s, torch.ones(6, dtype=torch.bool), ["a"] * 6)
m, sd = s.mean(), s.std()
check("混合组与 GRPO 一致", all(approx(adv[i], (s[i] - m) / (sd + 1e-6)) for i in range(6)))
check("混合组计数", st["with_spread"] == 1 and st["masked"] == 0)

# 2) 全同组（全对 / 全错）→ 优势 0
for val in (0., 1.):
    adv, st = masked_scalar_advantages(torch.full((6,), val), torch.ones(6, dtype=torch.bool), ["a"] * 6)
    check(f"全{'对' if val else '错'}组优势为 0", float(adv.abs().max()) == 0.0 and st["no_spread"] == 1)

# 3) 被 mask 的样本不进组统计：6 条里 3 条被 mask，剩下 1/0/0
s = torch.tensor([1., 0., 0., 0., 0., 0.])
adv, st = masked_scalar_advantages(s, torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool), ["a"] * 6)
kept = s[:3]
m, sd = kept.mean(), kept.std()
check("mask 不进组均值", all(approx(adv[i], (s[i] - m) / (sd + 1e-6)) for i in range(3)))
check("被 mask 的优势为 0", float(adv[3:].abs().max()) == 0.0)
check("masked 计数", st["masked"] == 3)
check("组均值是 1/3 而不是 1/6（verl 原版 grpo 会算成 1/6）", approx(m, 1 / 3))

# 4) mask 后只剩 1 条 → 优势 0
adv, st = masked_scalar_advantages(torch.tensor([1., 0., 0.]), torch.tensor([1, 0, 0], dtype=torch.bool), ["a"] * 3)
check("mask 后单条组优势为 0", float(adv.abs().max()) == 0.0 and st["singleton"] == 1)

# 5) 多组互不污染
adv, st = masked_scalar_advantages(torch.tensor([1., 0., 1., 1.]), torch.ones(4, dtype=torch.bool), ["a", "a", "b", "b"])
check("组 a 有方差、组 b 全同", float(adv[2:].abs().max()) == 0.0 and float(adv[:2].abs().max()) > 0)
check("组计数", st["groups"] == 2 and st["with_spread"] == 1 and st["no_spread"] == 1)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)

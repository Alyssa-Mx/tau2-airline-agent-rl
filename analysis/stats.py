"""评测统计：只用标准库，读 data/episodes.csv。

定义（τ²-bench 惯例，每题 k 遍）：
  pass@1   全部对话的通过率（主指标）
  pass@4   4 遍里至少通过 1 次的题占比（能力上限）；多于 4 遍时用无偏估计 1 − C(n−c,4)/C(n,4)
  pass^4   4 遍全部通过的题占比（稳定性，τ-bench 官方口径）；多于 4 遍时用 C(c,4)/C(n,4)
  截断率   撞 40 步（termination == max_steps）的对话占比
判分：宽松 = 正常结束 且 DB × COMMUNICATE == 1（τ²-bench airline 官方 reward_basis）；严格 = 再加 ACTION。

比较一律用逐题配对 bootstrap：先对每道题算两组的差，再对题目重采样 10000 次取 95% 区间。
20 道题上单组 pass@1 的区间宽达 ±0.13–0.15，两组绝对分相减没有分辨力；配对把题目难度消掉之后才有。
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from math import comb
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
_CACHE: dict = {}


def load(name: str = "episodes.csv") -> list[dict]:
    if name not in _CACHE:
        rows = list(csv.DictReader(open(DATA / name)))
        for r in rows:
            for k, v in r.items():
                if v in ("", None):
                    continue
                try:
                    r[k] = int(v)
                except ValueError:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
        _CACHE[name] = rows
    return _CACHE[name]


def arm(name: str, split: str | None = "test", max_trial: int | None = None, rows=None) -> list[dict]:
    rows = load() if rows is None else rows
    return [r for r in rows if r["arm"] == name and (split is None or r["split"] == split)
            and (max_trial is None or r["trial"] < max_trial)]


def pool(*arms_rows) -> list[dict]:
    """把几组（例如第 5、6 轮）合并成一组：每题的遍数相加。"""
    return [r for rows in arms_rows for r in rows]


def by_task(rows, key="pass_loose") -> dict:
    d = defaultdict(list)
    for r in rows:
        d[str(r["task_id"])].append(r[key])
    return d


def rate(rows, key="pass_loose") -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def task_pass_at_k(v: list, k: int = 4) -> float:
    n, c = len(v), sum(v)
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def task_pass_hat_k(v: list, k: int = 4) -> float:
    n, c = len(v), sum(v)
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def pass_at_k(rows, k=4) -> float:
    d = by_task(rows)
    return sum(task_pass_at_k(v, k) for v in d.values()) / len(d)


def pass_hat_k(rows, k=4) -> float:
    d = by_task(rows)
    return sum(task_pass_hat_k(v, k) for v in d.values()) / len(d)


def summary(rows) -> dict:
    return dict(n=len(rows), pass1=rate(rows), pass4=pass_at_k(rows), passhat4=pass_hat_k(rows),
                trunc=rate(rows, "truncated"), strict=rate(rows, "pass_strict"), flood=rate(rows, "flood"),
                turns=sum(r["agent_turns"] for r in rows) / len(rows),
                calls_per_turn=sum(r["n_tool_calls"] for r in rows) / max(1, sum(r["agent_turns"] for r in rows)))


_METRIC = {
    "pass1": lambda v: sum(v) / len(v),
    "pass4": task_pass_at_k,
    "passhat4": task_pass_hat_k,
}


def _boot(per_task: dict, n: int = 10000):
    xs = [per_task[t] for t in sorted(per_task)]
    rng = random.Random(0)
    m = sorted(sum(xs[rng.randrange(len(xs))] for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), m[int(0.025 * n)], m[int(0.975 * n)]


def paired(a_rows, b_rows, metric: str = "pass1"):
    """逐题配对差 a − b，返回 (差, 下界, 上界, 题数)。metric ∈ pass1 / pass4 / passhat4 / trunc。"""
    key = "truncated" if metric == "trunc" else "pass_loose"
    f = _METRIC.get(metric, _METRIC["pass1"])
    pa, pb = by_task(a_rows, key), by_task(b_rows, key)
    common = sorted(set(pa) & set(pb))
    d = {t: f(pa[t]) - f(pb[t]) for t in common}
    return _boot(d) + (len(common),)


def fmt_pair(res, pct: bool = False) -> str:
    d, lo, hi, _ = res
    sig = "*" if lo > 0 or hi < 0 else " "
    if pct:
        return f"{d * 100:+.1f}pp [{lo * 100:+.1f}, {hi * 100:+.1f}]{sig}"
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"

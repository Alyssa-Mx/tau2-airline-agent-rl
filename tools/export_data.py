"""把 tau2 原始仿真（results.json）和 verl 训练 rollout 压成仓库里的两张表。

只在原始实验机器上跑一次；仓库里的 analysis/ 只读这里导出的 CSV，不需要 tau2 或 GPU。
导出时丢掉对话原文与所有 LLM 调用参数（results.json 里带有网关密钥），只留判分与行为字段。

用法：
  PYTHONPATH=<tau2-bench>/src TAU2_DATA_DIR=<tau2-bench>/data \
  python tools/export_data.py --sim-root <data_strict/simulations> --base-root <base 20 遍 simulations 目录> \
      --runs-root <verl runs 目录> --out data/

输出：
  data/episodes.csv         评测：每条对话一行
  data/train_rollouts.csv   训练：每条 rollout 一行（每轮 30 题 × 6 条 = 180 条）
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.runner.helpers import get_tasks

WRITE_TOOLS = {"book_reservation", "cancel_reservation", "update_reservation_flights",
               "update_reservation_baggages", "update_reservation_passengers", "send_certificate"}
TASKS = {str(t.id): t for t in get_tasks("airline")}
TEST_IDS = {str(i) for i in range(50) if i % 5 in (0, 1)}          # 20 道 held-out 测试题


def task_type(tid: str) -> str:
    acts = [a.name for a in (TASKS[tid].evaluation_criteria.actions or [])]
    return "write" if any(a in WRITE_TOOLS for a in acts) else ("read" if acts else "none")


def is_flood(text) -> bool:
    c = (text or "").strip()
    return len(c) >= 24 and c.count("!") / len(c) > 0.9


def gold_done(tid: str, calls: list[dict]) -> bool:
    """gold 动作是否全部被调用过（名字 + compare_args 参数匹配，同 tau2 Action.compare_with_tool_call）。"""
    for a in TASKS[tid].evaluation_criteria.actions or []:
        ok = False
        for tc in calls:
            if tc.get("name") != a.name:
                continue
            args = tc.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    continue
            keys = list(args.keys()) if a.compare_args is None else a.compare_args
            if all(args.get(k) == (a.arguments or {}).get(k) for k in keys):
                ok = True
                break
        if not ok:
            return False
    return True


def regrade(sd: dict, tid: str):
    """把提前终止的对话改记 user_stop，用官方判分器重判 DB × COMMUNICATE。"""
    try:
        sim = SimulationRun.model_validate(sd)
        sim.termination_reason = TerminationReason.USER_STOP
        ri = evaluate_simulation(sim, TASKS[tid], EvaluationType.ALL, solo_mode=False, domain="airline", env_kwargs={})
        bd = {str(k).split(".")[-1].upper(): v for k, v in (ri.reward_breakdown or {}).items()}
        return int(float(bd.get("DB", 0)) * float(bd.get("COMMUNICATE", 1)) >= 1 - 1e-6)
    except Exception as e:  # noqa: BLE001
        print(f"  regrade failed on task {tid}: {type(e).__name__}")
        return ""


def episode_rows(arm: str, engine: str, paths: list[str]):
    seen = set()
    for p in paths:
        for sd in json.load(open(p))["simulations"]:
            tid, trial = str(sd["task_id"]), int(sd.get("trial") or 0)
            if (tid, trial) in seen:
                continue
            seen.add((tid, trial))
            ri = sd.get("reward_info") or {}
            bd = ri.get("reward_breakdown") or {}
            term = str(sd.get("termination_reason")).lower().replace("terminationreason.", "")
            db, comm = float(bd.get("DB", 1)), float(bd.get("COMMUNICATE", 1))
            normal = term in ("user_stop", "agent_stop")
            msgs = sd.get("messages") or []
            asst = [m for m in msgs if m.get("role") == "assistant"]
            calls = [tc for m in asst for tc in (m.get("tool_calls") or [])]
            truncated = term == "max_steps"
            yield dict(
                arm=arm, engine=engine, split="test" if tid in TEST_IDS else "train", task_id=int(tid),
                task_type=task_type(tid), trial=trial,
                pass_loose=int(normal and bool(bd) and db * comm >= 1 - 1e-6),
                pass_strict=int(float(ri.get("reward") or 0) >= 1 - 1e-6),
                termination=term, truncated=int(truncated),
                flood=int(any(is_flood(m.get("content")) for m in asst)),
                db=db if bd else "", comm=comm if bd else "",
                agent_turns=len(asst), n_messages=len(msgs), n_tool_calls=len(calls),
                n_write_calls=sum(1 for c in calls if c.get("name") in WRITE_TOOLS),
                gold_all_done=int(gold_done(tid, calls)),
                regrade_pass=regrade(sd, tid) if truncated else "",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-root", required=True)
    ap.add_argument("--base-root", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    sim = lambda run: sorted(glob.glob(f"{a.sim_root}/{run}/results.json"))          # noqa: E731
    nail = lambda run: sorted(glob.glob(f"{a.sim_root}/{run}_nail_base_t*/results.json"))  # noqa: E731

    arms: list[tuple[str, str, list[str]]] = [
        ("base_legacy", "legacy", sorted(glob.glob(f"{a.base_root}/*/results.json"))),
    ]
    for k in range(1, 7):
        arms.append((f"binary_r{k}", "legacy", nail(f"eval_rw01_r{k}_nc_0915")))
        arms.append((f"partial_r{k}", "legacy",
                     nail(f"eval_rwv3_it{k}_nc_0915" if k <= 3 else f"eval_rwv3c_it{k - 3}_nc_0915")))
        arms.append((f"mask_r{k}", "legacy", nail(f"eval_m01_it{k}_nc_0915")))
    arms += [
        ("legacy_lean_nothink", "legacy", sim("lean_formal_lean4_50x4_0915")),
        ("legacy_orig_think", "legacy", sim("lean_formal_basethink_50x4_0915")),
        ("legacy_lean_think", "legacy", sim("lean_formal_lean4think_50x4_0915")),
        ("legacy_base_train30", "legacy", sim("lean_formal_base30_4x_0915")),
        ("native_base_orig_nothink", "native", sim("lean_native_base_orig_nothink")),
        ("native_base_lean_nothink", "native", sim("lean_native_base_lean_nothink")),
        ("native_lean_think", "native", sim("lean_native_leanthink2_50x4_0916")),
        ("native_rl6_orig_nothink", "native", sim("lean_native_rl6_orig_nothink")),
        ("native_rl6_lean_nothink", "native", sim("lean_native_rl6_lean_nothink")),
        ("native_rl6_lean_think", "native", sim("lean_native_rl6_lean_think")),
        ("native_rl6_orig_think", "native", sim("lean_native_rl6_orig_think")),
        ("deepseek_selfplay", "api", sim("ds_ds_vanilla_50x4")),
    ]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm, engine, paths in arms:
        assert paths, f"no results for {arm}"
        got = list(episode_rows(arm, engine, paths))
        print(f"{arm:<28} {len(got):>4} episodes from {len(paths)} file(s)")
        rows += got
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- 训练 rollout ----------------
    runs = [("partial", 0, "verl_rwv3_0915"), ("partial", 3, "verl_rwv3c_0915"),
            ("binary", 0, "verl_rw01_0915"), ("binary", 3, "verl_rw01c_0915"), ("mask", 0, "verl_m01_0916")]
    trows = []
    for arm, offset, run in runs:
        recs = sorted((json.loads(l) for fp in glob.glob(f"{a.runs_root}/{run}/rollouts/*.jsonl")
                       for l in open(fp) if l.strip()), key=lambda r: r.get("time", 0))
        for i, r in enumerate(recs):
            term = str(r.get("termination") or "").lower().replace("terminationreason.", "")
            overlong = term == "max_steps" or r.get("kill_reason") in ("budget", "assist_turns", "repeat")
            trows.append(dict(
                arm=arm, round=offset + i // 180 + 1, task_id=int(r["task_id"]), task_type=task_type(str(r["task_id"])),
                reward=float(r.get("reward_score") or 0.0), outcome=float(r.get("outcome") or 0.0),
                overlong=int(overlong), masked=int(bool(r.get("masked"))), kill_reason=r.get("kill_reason") or "",
                termination=term, agent_turns=r.get("assistant_turns"), response_tokens=r.get("response_tokens"),
                infra_error=int(bool(r.get("sim_error")))))
        print(f"{run:<18} {len(recs)} rollouts → rounds {offset + 1}–{offset + len(recs) // 180}")
    with open(out / "train_rollouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader()
        w.writerows(trows)


if __name__ == "__main__":
    main()

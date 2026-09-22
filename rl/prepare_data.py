#!/usr/bin/env python3
"""把 τ²-bench airline 题库变成 verl 的 parquet。每行一道题，对话由 agent loop 现场生成，prompt 只是占位。

固定划分：task_id % 5 ∈ {0, 1} 的 20 道是 held-out 测试题，其余 30 道训练。
用法：python rl/prepare_data.py --data-dir <tau2-bench>/data --out data_rl/
"""
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

TRAIN_IDS = [str(i) for i in range(50) if i % 5 in (2, 3, 4)]
TEST_IDS = [str(i) for i in range(50) if i % 5 in (0, 1)]


def rows(ids, split):
    return [{
        "data_source": "tau2_airline",
        "prompt": [{"role": "user", "content": f"tau2 task {tid}"}],   # 占位；agent loop 按 extra_info.task_id 重建对话
        "ability": "agent",
        "reward_model": {"style": "rule", "ground_truth": str(tid)},
        "extra_info": {"task_id": str(tid), "split": split, "index": i, "seed": i},
        "agent_name": "tau2_agent",
    } for i, tid in enumerate(ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = {t["id"] for t in json.load(open(os.path.join(a.data_dir, "tau2/domains/airline/tasks.json")))}
    missing = [i for i in TRAIN_IDS + TEST_IDS if i not in tasks]
    assert not missing, f"tasks.json 里没有: {missing[:5]}"
    os.makedirs(a.out, exist_ok=True)
    for name, ids in (("train", TRAIN_IDS), ("val", TEST_IDS)):
        pq.write_table(pa.Table.from_pylist(rows(ids, name)), os.path.join(a.out, f"{name}.parquet"))
        print(name, len(ids), "rows ->", os.path.join(a.out, f"{name}.parquet"))


if __name__ == "__main__":
    main()

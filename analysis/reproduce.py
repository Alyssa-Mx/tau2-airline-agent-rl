"""重算 README 与 docs/ 里的全部数字。只读 data/*.csv，标准库即可，约 10 秒。
    python analysis/reproduce.py            # 全部
    python analysis/reproduce.py failure    # 只看某一节：failure / reward / training / harness / stacking
"""
import sys
from collections import Counter, defaultdict

from stats import arm, fmt_pair, load, paired, pool, rate, summary

W = lambda s="": print(s)  # noqa: E731


def row(label, rows):
    s = summary(rows)
    W(f"  {label:<34}{s['n']:>5}  pass@1 {s['pass1']:.3f}  pass@4 {s['pass4']:.3f}  pass^4 {s['passhat4']:.3f}"
      f"  截断 {s['trunc']:6.1%}  刷屏对话 {s['flood']:6.1%}  严格 {s['strict']:.3f}  客服轮 {s['turns']:4.1f}  工具/轮 {s['calls_per_turn']:.2f}")


def failure():
    W("=" * 110 + "\n§1 失败归因：原模型在 20 道测试题 × 20 遍 = 400 段对话（旧口径引擎）\n" + "=" * 110)
    base = arm("base_legacy")
    fail = [r for r in base if not r["pass_loose"]]
    trunc = [r for r in base if r["truncated"]]
    W(f"  通过 {len(base) - len(fail)} / {len(base)}（pass@1 {rate(base):.3f}），失败 {len(fail)}")
    W(f"  失败里撞 40 步被截断 {len(trunc)}（占失败 {len(trunc) / len(fail):.1%}）；"
      f"正常结束但数据库终态错 {sum(1 for r in fail if not r['truncated'] and r['db'] == 0)}；其他 {len(fail) - len(trunc) - sum(1 for r in fail if not r['truncated'] and r['db'] == 0)}")
    done = sum(r["gold_all_done"] for r in trunc if r["task_type"] != "none")   # 无 gold 动作的题"全部完成"是空真，不算
    W(f"  截断的 {len(trunc)} 条里：伴随刷屏 {sum(r['flood'] for r in trunc)}（{rate(trunc, 'flood'):.0%}）；"
      f"gold 动作已全部执行 {done}（{done / len(trunc):.0%}，只计有 gold 动作的题）")
    W("  把截断对话改记正常结束、用官方判分器重判（DB × COMMUNICATE）：")
    tot_ok = 0
    for tt, name in (("write", "写库题"), ("read", "只读题"), ("none", "无 gold（应拒绝）")):
        t = [r for r in trunc if r["task_type"] == tt]
        ok = [r for r in t if r["regrade_pass"] == 1]
        extra = f"，gold 查询全做完 {sum(r['gold_all_done'] for r in ok)}" if tt == "read" else ""
        W(f"    {name:<16} 截断 {len(t):>3}  重判能过 {len(ok):>3}（其中伴随刷屏 {sum(r['flood'] for r in ok)}{extra}）")
        if tt != "none":
            tot_ok += len(ok)
    W(f"  → 写库 + 只读题里重判能过的 {tot_ok} / {len(trunc)} = {tot_ok / len(trunc):.0%}：活已经干完，只是没在 40 步内收尾")
    W("\n  失败结构（占全部对话）：")
    for label, rows in (("原模型（400）", base),):
        c = Counter("撞步·刷屏" if r["truncated"] and r["flood"] else "撞步·无刷屏" if r["truncated"]
                    else "正常结束·库错" if not r["pass_loose"] and r["db"] == 0 else "通过" if r["pass_loose"] else "其他" for r in rows)
        W(f"    {label}: " + "  ".join(f"{k} {v}（{v / len(rows):.1%}）" for k, v in c.most_common()))


def reward():
    W("\n" + "=" * 110 + "\n§2 奖励设计：三个训练臂只差奖励 / 超长处理，每轮 checkpoint 测 20 题 × 4 遍（旧口径引擎）\n" + "=" * 110)
    names = (("partial", "0/0.5/1 completion-aware"), ("binary", "0/1 官方"), ("mask", "0/1 + DAPO 超长过滤"))
    for k in range(1, 7):
        parts = []
        for a, _ in names:
            s = summary(arm(f"{a}_r{k}"))
            parts.append(f"{a:<8}{s['pass1']:.3f}/{s['pass4']:.3f}/{s['passhat4']:.3f}/{s['trunc']:5.1%}")
        W(f"  第 {k} 轮  " + "   ".join(parts) + "    （pass@1 / pass@4 / pass^4 / 截断）")
    P = lambda a, ks: pool(*[arm(f"{a}_r{k}") for k in ks])  # noqa: E731
    W("\n  逐题配对（* = 95% 区间不含 0）")
    W(f"    [主比较，事先约定] 第 6 轮 partial − binary  pass@1 {fmt_pair(paired(arm('partial_r6'), arm('binary_r6')))}")
    W(f"    第 5–6 轮合并 partial − binary  pass@1 {fmt_pair(paired(P('partial', (5, 6)), P('binary', (5, 6))))}"
      f"   截断 {fmt_pair(paired(P('partial', (5, 6)), P('binary', (5, 6)), 'trunc'), pct=True)}")
    W(f"    6 轮全部合并 partial − binary  pass@1 {fmt_pair(paired(P('partial', range(1, 7)), P('binary', range(1, 7))))}"
      f"   截断 {fmt_pair(paired(P('partial', range(1, 7)), P('binary', range(1, 7)), 'trunc'), pct=True)}")
    W(f"    第 6 轮 mask − partial  pass@1 {fmt_pair(paired(arm('mask_r6'), arm('partial_r6')))}")
    W(f"    第 5–6 轮合并 mask − partial  pass@1 {fmt_pair(paired(P('mask', (5, 6)), P('partial', (5, 6))))}"
      f"   截断 {fmt_pair(paired(P('mask', (5, 6)), P('partial', (5, 6)), 'trunc'), pct=True)}")
    W(f"    第 5–6 轮合并 mask − binary  pass@1 {fmt_pair(paired(P('mask', (5, 6)), P('binary', (5, 6))))}")
    base = arm("base_legacy")
    W("\n  相对原模型（同引擎，原模型 20 遍）")
    for a in ("partial", "binary", "mask"):
        r6 = arm(f"{a}_r6")
        W(f"    {a:<8} 第 6 轮  pass@1 {fmt_pair(paired(r6, base))}  pass@4 {fmt_pair(paired(r6, base, 'pass4'))}"
          f"  截断 {fmt_pair(paired(r6, base, 'trunc'), pct=True)}")
    W()
    row("原模型（20 遍）", base)
    for a, label in names:
        row(f"{a} 第 6 轮", arm(f"{a}_r6"))
        row(f"{a} 第 5–6 轮合并", P(a, (5, 6)))


def training():
    W("\n" + "=" * 110 + "\n§2b 训练侧：每轮 180 段 rollout（30 道训练题 × 6 条）\n" + "=" * 110)
    tr = load("train_rollouts.csv")
    for a in ("partial", "binary", "mask"):
        rows = [r for r in tr if r["arm"] == a]
        line_o, line_s, line_g, line_h, line_t = [], [], [], [], []
        for k in range(1, 7):
            rr = [r for r in rows if r["round"] == k]
            g = defaultdict(list)
            for r in rr:
                if not r["masked"]:
                    g[r["task_id"]].append(r["reward"])
            line_o.append(f"{sum(r['overlong'] for r in rr):>3}")
            line_s.append(f"{rate(rr, 'outcome'):.3f}")
            line_g.append(f"{sum(1 for v in g.values() if len(v) > 1 and len(set(v)) > 1):>3}")
            line_h.append(f"{sum(1 for r in rr if 0 < r['reward'] < 1):>3}")
            line_t.append(f"{sum(r['agent_turns'] for r in rr) / len(rr):4.1f}")
        W(f"  {a:<8} 截断条数 " + " ".join(line_o) + "   | 有梯度的组/30 " + " ".join(line_g))
        W(f"  {'':<8} 官方分   " + " ".join(line_s) + "   | 拿 0.5 的条数 " + " ".join(line_h) + "   | 客服轮 " + " ".join(line_t))
        W(f"  {'':<8} 网关报错 {sum(r['infra_error'] for r in rows)}，被 mask {sum(r['masked'] for r in rows)}")


def harness():
    W("\n" + "=" * 110 + "\n§3 推理外壳：同一台原生口径引擎，20 道测试题 × 4 遍（不训练）\n" + "=" * 110)
    o = arm("native_base_orig_nothink")
    arms_ = (("原始 · 不思考（对照）", o), ("精简 · 不思考", arm("native_base_lean_nothink")),
             ("精简 · 思考（定版）", arm("native_lean_think")), ("DeepSeek-v4-Flash 自演双方", arm("deepseek_selfplay")))
    for label, rows in arms_:
        row(label, rows)
    W()
    for label, rows in arms_[1:]:
        W(f"    {label} − 对照  pass@1 {fmt_pair(paired(rows, o))}  pass^4 {fmt_pair(paired(rows, o, 'passhat4'))}"
          f"  截断 {fmt_pair(paired(rows, o, 'trunc'), pct=True)}")
    W(f"    定版 − 精简·不思考（思考的作用）  pass@1 {fmt_pair(paired(arm('native_lean_think'), arm('native_base_lean_nothink')))}")
    lt50, ds50 = arm("native_lean_think", split=None), arm("deepseek_selfplay", split=None)
    W(f"\n  全部 50 题（不训练的臂没有训练/测试之分）：定版 {rate(lt50):.3f}  DeepSeek 自演 {rate(ds50):.3f}"
      f"   定版 − DeepSeek {fmt_pair(paired(lt50, ds50))}  截断 {fmt_pair(paired(lt50, ds50, 'trunc'), pct=True)}")
    W("\n  旧口径引擎上的 2×2（全部 50 题 × 4；原始·不思考 = 测试题取原模型前 4 遍 + 训练题 30 题补测）")
    b50 = pool(arm("base_legacy", max_trial=4), arm("legacy_base_train30", split=None))
    cells = (("原始·不思考", b50), ("精简·不思考", arm("legacy_lean_nothink", split=None)),
             ("原始·思考", arm("legacy_orig_think", split=None)), ("精简·思考", arm("legacy_lean_think", split=None)))
    for label, rows in cells:
        row(label, rows)
    for label, rows in cells[1:]:
        W(f"    {label} − 原始·不思考  pass@1 {fmt_pair(paired(rows, b50))}")
    W(f"    定版（原生）− 原始·不思考（旧）  pass@1 {fmt_pair(paired(lt50, b50))}   ← 跨引擎，含调用口径切换")


def stacking():
    W("\n" + "=" * 110 + "\n§4 两条线能不能叠加：把部分分奖励第 6 轮的权重放进原生口径引擎（20 题 × 4）\n" + "=" * 110)
    for h in ("orig_nothink", "lean_nothink", "lean_think", "orig_think"):
        base = arm(f"native_base_{h}") if h != "lean_think" else arm("native_lean_think")
        rl = arm(f"native_rl6_{h}")
        b = f"{rate(base):.3f}" if base else "  —  "
        extra = f"   RL6 − 原模型 {fmt_pair(paired(rl, base))}" if base else ""
        W(f"  {h:<14} 原模型 {b}   RL6 {rate(rl):.3f}{extra}")


SECTIONS = dict(failure=failure, reward=reward, training=training, harness=harness, stacking=stacking)
if __name__ == "__main__":
    for name in (sys.argv[1:] or SECTIONS):
        SECTIONS[name]()

"""生成 assets/ 下的全部配图（浅色 + 深色两版，README 用 <picture> 按读者主题切换）。
数字全部现算自 data/*.csv / *.json，不手填。需要 matplotlib。
    python analysis/make_figures.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401

from stats import DATA, arm, load, pool, rate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e8e7e3", neutral="#d6d5d0",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e87", grid="#2e2e2c", neutral="#4a4a47",
                 s1="#3987e5", s2="#d95926", s3="#199e70"),
}
ARM_COLOR = {"partial": "s1", "binary": "s2", "mask": "s3"}
ARM_LABEL = {"partial": "0 / 0.5 / 1  completion-aware (ours)", "binary": "0 / 1  binary", "mask": "0 / 1 + DAPO overlong filtering"}


def style(ax, c, ygrid=True, xgrid=False):
    ax.set_facecolor(c["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(c["grid"])
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=c["ink2"], labelsize=8.5, length=0)
    if ygrid:
        ax.yaxis.grid(True, color=c["grid"], linewidth=1)
    if xgrid:
        ax.xaxis.grid(True, color=c["grid"], linewidth=1)
    ax.set_axisbelow(True)


def title(fig, c, t, sub):
    fig.text(0.012, 0.965, t, fontsize=12, fontweight="bold", color=c["ink"], va="top")
    fig.text(0.012, 0.905, sub, fontsize=9, color=c["ink2"], va="top")


def save(fig, name, theme):
    fig.savefig(OUT / f"{name}_{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- 1. 失败归因
def fig_failure(theme):
    c = THEMES[theme]
    base = arm("base_legacy")
    n = len(base)
    groups = [
        ("Passed", sum(r["pass_loose"] for r in base), c["neutral"]),
        ("Truncated at 40 steps,\nwith '!!!!' flooding", sum(1 for r in base if r["truncated"] and r["flood"]), c["s2"]),
        ("Truncated at 40 steps,\nno flooding", sum(1 for r in base if r["truncated"] and not r["flood"]), c["s3"]),
        ("Ended normally,\nwrong final DB", sum(1 for r in base if not r["pass_loose"] and not r["truncated"]), c["s1"]),
    ]
    fig = plt.figure(figsize=(8, 4.9), facecolor=c["surface"])
    title(fig, c, "Where the original Qwen3.5-4B fails",
          f"{n} conversations = 20 held-out tasks × 20 trials. Two thirds of the failures are truncations, not wrong answers.")
    ax = fig.add_axes([0.03, 0.63, 0.94, 0.12])
    style(ax, c, ygrid=False)
    left = 0
    for i, (lab, v, col) in enumerate(groups):
        ax.barh(0, v, left=left, height=0.6, color=col, edgecolor=c["surface"], linewidth=2)
        if i == 0:
            ax.text(left + v / 2, 0, f"Passed  {v} ({v / n:.0%})", ha="center", va="center", fontsize=8.5, color=c["ink"])
        left += v
    ax.set_xlim(0, n)
    ax.set_ylim(-0.4, 0.4)
    ax.axis("off")
    short = ["Truncated, with '!!!!' flooding", "Truncated, no flooding", "Ended normally, wrong final DB"]
    handles = [matplotlib.patches.Patch(color=col) for _, _, col in groups[1:]]
    fig.legend(handles, [f"{t}  {v} ({v / n:.0%})" for t, (_, v, _) in zip(short, groups[1:])], loc="center",
               bbox_to_anchor=(0.5, 0.565), ncol=3, frameon=False, fontsize=8, labelcolor=c["ink2"], handlelength=1.0,
               handleheight=1.0, columnspacing=2.0)
    trunc = [r for r in base if r["truncated"]]
    ax2 = fig.add_axes([0.20, 0.08, 0.77, 0.27])
    style(ax2, c, ygrid=False, xgrid=True)
    rows = [("write", "Write tasks\n(must change the DB)"), ("read", "Read-only tasks"), ("none", "No-action tasks\n(should refuse)")]
    for i, (tt, lab) in enumerate(rows):
        t = [r for r in trunc if r["task_type"] == tt]
        ok = [r for r in t if r["regrade_pass"] == 1]
        y = len(rows) - 1 - i
        ax2.barh(y, len(t), height=0.42, color=c["neutral"])
        ax2.barh(y, len(ok), height=0.42, color=c["ink2"] if tt != "none" else c["muted"])
        note = "all with flooding" if tt == "write" else ("work done, just never closed" if tt == "read" else f"{sum(r['flood'] for r in ok)} of {len(ok)} are flooding — no credit")
        ax2.text(len(t) + 0.6, y, f"{len(ok)} / {len(t)} pass if re-graded as finished  ·  {note}", va="center", fontsize=8, color=c["ink2"])
    ax2.set_yticks(range(len(rows)))
    ax2.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8.5, color=c["ink2"])
    ax2.set_xlim(0, 75)
    ax2.set_xlabel("truncated conversations", fontsize=8.5, color=c["ink2"])
    fig.text(0.012, 0.43, f"The {len(trunc)} truncated conversations, re-graded by the official evaluator as if the agent had been allowed to finish",
             fontsize=9, color=c["ink"], fontweight="bold")
    save(fig, "fig_failure", theme)


# ---------------------------------------------------------------- 2. 奖励设计
def fig_reward(theme):
    c = THEMES[theme]
    tr = load("train_rollouts.csv")
    fig = plt.figure(figsize=(8, 3.9), facecolor=c["surface"])
    title(fig, c, "Same GRPO recipe, only the treatment of truncated rollouts differs",
          "Left: held-out pass@1 of each round's checkpoint (20 tasks × 4 trials).  Right: truncated rollouts per training round (of 180).")
    ax1 = fig.add_axes([0.07, 0.13, 0.36, 0.62])
    ax2 = fig.add_axes([0.58, 0.13, 0.36, 0.62])
    for ax in (ax1, ax2):
        style(ax, c)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("training round", fontsize=8.5, color=c["ink2"])
    xs = list(range(1, 7))
    for a in ("binary", "mask", "partial"):
        col = c[ARM_COLOR[a]]
        ev = [rate(arm(f"{a}_r{k}")) for k in xs]
        ax1.plot(xs, ev, color=col, lw=2, solid_capstyle="round", zorder=3 if a == "partial" else 2)
        ax1.scatter(xs, ev, s=26, color=col, edgecolor=c["surface"], linewidth=1.5, zorder=4)
        ax1.text(6.18, ev[-1], f"{ev[-1]:.3f}", va="center", fontsize=8, color=c["ink2"])
        ov = [sum(r["overlong"] for r in tr if r["arm"] == a and r["round"] == k) for k in xs]
        ax2.plot(xs, ov, color=col, lw=2, solid_capstyle="round", zorder=3 if a == "partial" else 2, label=ARM_LABEL[a])
        ax2.scatter(xs, ov, s=26, color=col, edgecolor=c["surface"], linewidth=1.5, zorder=4)
        ax2.text(6.18, ov[-1], f"{ov[-1]}", va="center", fontsize=8, color=c["ink2"])
    ax1.set_ylim(0.5, 0.85)
    ax1.set_ylabel("pass@1", fontsize=8.5, color=c["ink2"])
    ax2.set_ylim(0, 40)
    ax2.set_ylabel("truncated rollouts", fontsize=8.5, color=c["ink2"])
    h, l = ax2.get_legend_handles_labels()
    order = [2, 0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(0.06, 0.86), ncol=3,
               frameon=False, fontsize=8.5, labelcolor=c["ink2"], handlelength=1.6, columnspacing=1.6)
    save(fig, "fig_reward", theme)


# ---------------------------------------------------------------- 3. 上下文预算 + 刷屏门槛
def fig_context(theme):
    c = THEMES[theme]
    bud = json.load(open(DATA / "prompt_budget.json"))
    fp = json.load(open(DATA / "flood_probe.json"))
    fig = plt.figure(figsize=(8, 4.4), facecolor=c["surface"])
    title(fig, c, "Shorter prompts, and why length matters here",
          "Left: fixed tokens paid on every call.  Right: on vLLM 0.19.1, long prompts make Qwen3.5-4B replies collapse into '!!!!'.")
    ax = fig.add_axes([0.13, 0.30, 0.30, 0.44])
    style(ax, c, ygrid=False, xgrid=True)
    sysn = bud["system_without_tools"]
    for y, (lab, tools_) in enumerate((("Lean", bud["tool_schema_tokens"]["lean"]), ("Original", bud["tool_schema_tokens"]["original"]))):
        ax.barh(y, sysn, height=0.45, color=c["neutral"], edgecolor=c["surface"], linewidth=2,
                label="policy + instructions" if y == 0 else None)
        ax.barh(y, tools_, left=sysn, height=0.45, color=c["s1"], edgecolor=c["surface"], linewidth=2,
                label="14 tool schemas" if y == 0 else None)
        ax.text(sysn / 2, y, f"{sysn:,}", ha="center", va="center", fontsize=8, color=c["ink"])
        ax.text(sysn + tools_ / 2, y, f"{tools_:,}", ha="center", va="center", fontsize=8, color="#ffffff")
        ax.text(sysn + tools_ + 120, y, f"{sysn + tools_:,}", va="center", fontsize=8.5, color=c["ink"], fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.22), ncol=2, frameon=False, fontsize=7.5, labelcolor=c["ink2"],
              handlelength=1.2, columnspacing=1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Lean", "Original"], fontsize=8.5, color=c["ink2"])
    ax.set_xlim(0, 6200)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("fixed prompt tokens per call", fontsize=8.5, color=c["ink2"])

    ax2 = fig.add_axes([0.56, 0.30, 0.40, 0.50])
    style(ax2, c)
    pts = defaultdict(lambda: [0, 0])
    for r in fp["vllm_length_sweep"]:
        pts[r["prompt_tokens"]][0] += r["floods"]
        pts[r["prompt_tokens"]][1] += r["n"]
    xs = sorted(pts)
    ys = [pts[x][0] / pts[x][1] for x in xs]
    ax2.plot([x / 1000 for x in xs], ys, color=c["s2"], lw=2, label="vLLM default (chunked prefill), n=24/pt")
    ax2.scatter([x / 1000 for x in xs], ys, s=26, color=c["s2"], edgecolor=c["surface"], linewidth=1.5, zorder=4)
    hf = fp["hf_transformers_same_weights"]
    nc = fp["vllm_no_chunked_prefill"]
    ax2.scatter([hf["prompt_tokens"] / 1000], [hf["floods"] / hf["n"]], s=40, marker="D", color=c["s3"], edgecolor=c["surface"],
                linewidth=1.5, zorder=5, label="same weights on HF transformers")
    ax2.scatter([nc["prompt_tokens"] / 1000], [nc["floods"] / nc["n"]], s=44, marker="s", color=c["s1"], edgecolor=c["surface"],
                linewidth=1.5, zorder=5, label="vLLM, prefill in one chunk")
    for x, lab, dx, ha in ((6.756, "lean harness\nmedian peak", -0.25, "right"), (10.551, "original prompt\nmedian peak", 0.25, "left")):
        ax2.axvline(x, color=c["muted"], lw=1, zorder=1)
        ax2.text(x + dx, 0.95, lab, fontsize=7, color=c["ink2"], va="top", ha=ha)
    ax2.set_ylim(-0.04, 1.0)
    ax2.set_xlim(4, 23.5)
    ax2.set_xlabel("prompt length (k tokens)", fontsize=8.5, color=c["ink2"])
    ax2.set_ylabel("share of replies that flood", fontsize=8.5, color=c["ink2"])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=1, frameon=False, fontsize=7.5, labelcolor=c["ink2"],
               handlelength=1.4)
    save(fig, "fig_context", theme)


# ---------------------------------------------------------------- 4. 推理外壳结果
def fig_harness(theme):
    c = THEMES[theme]
    arms_ = [("Original prompt, no thinking", arm("native_base_orig_nothink"), c["neutral"]),
             ("Lean harness, no thinking", arm("native_base_lean_nothink"), c["neutral"]),
             ("Lean harness + thinking (final)", arm("native_lean_think"), c["s1"]),
             ("DeepSeek-v4-Flash playing both sides", arm("deepseek_selfplay"), c["muted"])]
    fig = plt.figure(figsize=(8, 3.3), facecolor=c["surface"])
    title(fig, c, "Qwen3.5-4B without any training: harness only",
          "20 held-out tasks × 4 trials, same vLLM server and sampling for the three 4B rows.")
    for j, (key, lab, fmt, lim) in enumerate((("pass_loose", "pass@1", "{:.3f}", 1.0), ("truncated", "truncated at 40 steps", "{:.1%}", 0.3))):
        ax = fig.add_axes([0.36 + j * 0.33, 0.12, 0.25, 0.62])
        style(ax, c, ygrid=False, xgrid=True)
        for i, (name, rows, col) in enumerate(arms_):
            y = len(arms_) - 1 - i
            v = rate(rows, key)
            ax.barh(y, v, height=0.45, color=col)
            ax.text(v + lim * 0.02, y, fmt.format(v), va="center", fontsize=8.5, color=c["ink"] if i == 2 else c["ink2"],
                    fontweight="bold" if i == 2 else "normal")
        ax.set_xlim(0, lim)
        ax.set_yticks(range(len(arms_)))
        ax.set_yticklabels([] if j else [a[0] for a in arms_][::-1], fontsize=8.5, color=c["ink2"])
        ax.set_title(lab, fontsize=9, color=c["ink2"], loc="left")
        if key == "truncated":
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    save(fig, "fig_harness", theme)


if __name__ == "__main__":
    for th in THEMES:
        fig_failure(th)
        fig_reward(th)
        fig_context(th)
        fig_harness(th)
    print("figures ->", OUT)

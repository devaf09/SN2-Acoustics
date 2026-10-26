"""results/expNN.txt から docs/images/ の図を作り直す。

    python tools/make_figures.py

- exp02_04_accuracy_vs_compute.png: 計算量（FFT 回数、全員を動かす = 100%）と全体の正解率。専門家 4 人・16 クラス、課題ごと
- exp03_colors.png: 振り分け役（板 8 枚・16 枚）の色の数 T と振り分けの正解率（横軸は 2 倍ごと = 対数）
色は 3 つまで（データ可視化の基準パレットの 1〜3 番: 青・橙・緑。全ての組み合わせで色覚の違いに強いことが確かめられている範囲）。
"""
from __future__ import annotations

import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": ["Yu Gothic", "Meiryo", "Noto Sans JP", "sans-serif"], "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def load(exp):
    return [json.loads(l) for l in open(os.path.join(ROOT, "results", f"exp{exp}.txt"), encoding="utf-8") if l.strip()]


def agg(vals):
    vals = [v for v in vals if v is not None]
    m = st.mean(vals)
    se = st.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
    return 100 * m, 100 * se


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 図 1: 計算量と正解率 ---------------------------------------------------------------------
pts = defaultdict(lambda: defaultdict(list))          # pts[task][(group, name)] = [(acc, pct)...]
for r in load("02"):
    if "runs" not in r or r.get("variant", "fixed") != "fixed":
        continue
    for run in r["runs"]:
        if run["K"] != 4:
            continue
        for m, v in run["results"].items():
            if m.startswith("all-"):
                g = "all"
            elif m.startswith(("R1", "R2")) and ("top1" in m or m.endswith("top2-jcal")):
                g = "router"
            else:
                continue
            pts[r["task"]][(g, m)].append((v["acc"], v["ffts_pct"]))
for r in load("04"):
    for m, v in r["results"].items():
        if m in ("shared-top1", "shared-top2-jcal"):
            pts[r["task"]][("shared", f"f{r['front']}b{r['back']}:{m}")].append((v["acc"], v["ffts_pct"]))

label = {"all": "全員を動かす（比べる相手）", "router": "別々の振り分け役（R1・R2）", "shared": "前段を共有（実験04）"}
color = {"all": C1, "router": C2, "shared": C3}
marker = {"all": "s", "router": "o", "shared": "D"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
for ax, task, title in zip(axes, ("order", "freq"), ("順序が決め手の課題", "文字の頻度が決め手の課題")):
    style(ax)
    best = max(agg([a for a, _ in v])[0] for (g, _), v in pts[task].items() if g == "all")
    ax.axvspan(0, 35, color=C3, alpha=0.07, linewidth=0)
    ax.axhline(best, color=C1, linewidth=1, linestyle="--")
    ax.text(66, best + 1, f"全員を動かす最良 {best:.1f}%", color=INK2, va="bottom", ha="left", fontsize=8)
    ax.text(1.5, 2, "計算量 35% 以下（目標）", color=INK2, fontsize=8)
    seen = set()
    for (g, name), v in pts[task].items():
        m_acc, se = agg([a for a, _ in v])
        pct = v[0][1]
        ax.errorbar(pct, m_acc, yerr=se, fmt=marker[g], ms=7, color=color[g], ecolor=color[g], elinewidth=1,
                    markeredgecolor=SURF, markeredgewidth=1.5, label=label[g] if g not in seen else None, zorder=3)
        seen.add(g)
        if g == "shared" and name.startswith("f8b6") and name.endswith("jcal"):
            ax.annotate(f"前段 8 + 後段 6\n上位 2 人 + 較正\n{m_acc:.1f}%", (pct, m_acc), xytext=(pct + 8, m_acc - 22),
                        fontsize=8, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 102)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("計算量（FFT の回数、全員を動かす = 100%）")
axes[0].set_ylabel("全体の正解率（%）")
axes[0].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("専門家 4 人・16 クラス：計算量と正解率（3 seed の平均、ひげは標準誤差）", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "exp02_04_accuracy_vs_compute.png"), dpi=150)
plt.close(fig)

# ---- 図 2: 色の数 T と振り分けの正解率 -------------------------------------------------------------
res = defaultdict(list)
for r in load("03"):
    for m, v in r["results"].items():
        if m.startswith("R2-") and m.endswith("top1"):
            d, T = m[4:].split("-")[0].split("T")
            res[(r["task"], int(d), int(T))].append(v["route_top1"])
fig, ax = plt.subplots(figsize=(6.2, 4))
style(ax)
for task, c, nm in (("order", C2, "順序の課題"), ("freq", C1, "頻度の課題")):
    for d, ls in ((8, "-"), (16, ":")):
        Ts = sorted(T for (t, dd, T) in res if t == task and dd == d)
        ys = [agg(res[(task, d, T)]) for T in Ts]
        ax.errorbar(Ts, [y for y, _ in ys], yerr=[e for _, e in ys], color=c, linestyle=ls, linewidth=2, marker="o", ms=6,
                    markeredgecolor=SURF, label=f"{nm}・板 {d} 枚")
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8])
ax.set_xticklabels(["1", "2", "4", "8"])
ax.axhline(25, color=INK2, linewidth=1, linestyle="--")
ax.text(1, 26, "ランダム（25%）", color=INK2, fontsize=8)
ax.set_ylim(0, 100)
ax.set_xlabel("振り分け役の色の数 T（2 倍ごと）")
ax.set_ylabel("振り分けの正解率（上位 1、%）")
ax.set_title("実験03：色の数を減らすと、順序の情報が消える", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "exp03_colors.png"), dpi=150)
plt.close(fig)
print("書き出し:", os.listdir(OUT))

# ---- 図 3: 2D（実験06・07）の計算量と正解率 ----------------------------------------------------------
if os.path.exists(os.path.join(ROOT, "results", "exp06.txt")) and os.path.exists(os.path.join(ROOT, "results", "exp07.txt")):
    p6 = defaultdict(lambda: defaultdict(list))
    flat = defaultdict(list)
    for r in load("06"):
        if r["args"]["reject"] == 0:
            continue
        for m, v in r["results"].items():
            if m == "flat-1net":
                flat[r["task"]].append(v["acc"])
            elif m.startswith("all-"):
                p6[r["task"]][("all", m)].append((v["acc"], v["ffts_pct"]))
            elif m.startswith(("R1", "R2")):
                p6[r["task"]][("router", m)].append((v["acc"], v["ffts_pct"]))
    for r in load("07"):
        for m, v in r["results"].items():
            if m.startswith("shared-top"):
                p6[r["task"]][("shared", f"f{r['front']}b{r['back']}:{m}")].append((v["acc"], v["ffts_pct"]))
    titles = {"synth4": "合成 4 課題（専門家 4 人・33 クラス）", "fashion_sim": "Fashion（似た服を同じ専門家に）",
              "fashion_mix": "Fashion（似た服を別の専門家に）"}
    label2 = {"all": "全員を動かす", "router": "別々の振り分け役 + 元の専門家", "shared": "前段を共有（実験07）"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, task in zip(axes, ("synth4", "fashion_sim", "fashion_mix")):
        style(ax)
        best = max(agg([a_ for a_, _ in v])[0] for (g, _), v in p6[task].items() if g == "all")
        fl = agg(flat[task])[0]
        ax.axvspan(0, 35, color=C3, alpha=0.07, linewidth=0)
        ax.axhline(best, color=C1, linewidth=1, linestyle="--")
        ax.text(60, best + 0.6, f"全員を動かす最良 {best:.1f}%", color=INK2, fontsize=8)
        ax.axhline(fl, color=INK2, linewidth=1, linestyle=":")
        ax.text(60, fl - 2.2, f"1 本で全クラス {fl:.1f}%（計算 {100 / (4 if task == 'synth4' else 5):.0f}%）", color=INK2, fontsize=8)
        seen = set()
        for (g, name), v in p6[task].items():
            m_acc, se = agg([a_ for a_, _ in v])
            ax.errorbar(v[0][1], m_acc, yerr=se, fmt=marker[g], ms=7, color=color[g], ecolor=color[g], elinewidth=1,
                        markeredgecolor=SURF, markeredgewidth=1.5, label=label2[g] if g not in seen else None, zorder=3)
            seen.add(g)
        ax.set_xlim(0, 105)
        ax.set_ylim(40, 95)
        ax.set_title(titles[task], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("計算量（FFT の回数、元の専門家を全員動かす = 100%）")
    axes[0].set_ylabel("全体の正解率（%）")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("2D の実験06・07：計算量と正解率（3 seed の平均、ひげは標準誤差。緑の帯 = 計算量 35% 以下）", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp06_07_2d.png"), dpi=150)
    plt.close(fig)
    print("書き出し: exp06_07_2d.png")

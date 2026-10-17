"""実験 A1b の seed をまとめる（平均 ± 標準誤差 = 標準偏差/√seed 数）。
使い方: python experiments/a1b_summary.py order   → results/a1b_order_summary.txt"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
task = sys.argv[1] if len(sys.argv) > 1 else "order"
runs = [json.load(open(p, encoding="utf-8"))
        for p in sorted(glob.glob(os.path.join(ROOT, "results", "a1b", task, "seed*", "route.json")))]


def ms(xs, f="{:.1f}", scale=100):
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f.format(xs[0])
    return (f + "±" + f).format(st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)))


lines = [f"課題 {task}  seed {[r['seed'] for r in runs]}  steps {runs[0]['steps']}  装置 {runs[0]['device_name']}  "
         f"（平均 ± 標準誤差）", ""]
for ki, K in enumerate([rr["K"] for rr in runs[0]["runs"]]):
    rs = [r["runs"][ki] for r in runs]
    solo = [st.mean(rr["solo"]) for rr in rs]
    lines.append(f"## K={K} 専門家（{4 * K} クラス、テスト {rs[0]['n_test']} 件/seed）  専門家単独の正解率 {ms(solo)}%")
    lines.append("| 方式 | 全体の正解率 % | 振り分け top1 % | 振り分け top2 % | FFT/入力 | 計算量 % | 実測 ms/入力 | 学習する数（うち振り分け役） |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in rs[0]["results"]:
        v = [rr["results"][name] for rr in rs]
        tm = v[0]["route_topm"] is not None and not name.endswith("top1")
        lines.append(f"| {name} | {ms([x['acc'] for x in v])} | {ms([x['route_top1'] for x in v])} | "
                     f"{ms([x['route_topm'] for x in v]) if tm else '-'} | {v[0]['ffts']} | {v[0]['ffts_pct']:.1f} | "
                     f"{ms([x['ms'] for x in v], '{:.3f}', 1)} | {v[0]['params']} ({v[0]['extra_params'] if v[0]['extra_params'] is not None else '-'}) |")
    lines.append("")
txt = "\n".join(lines)
print(txt)
open(os.path.join(ROOT, "results", f"a1b_{task}_summary.txt"), "w", encoding="utf-8").write(txt + "\n")

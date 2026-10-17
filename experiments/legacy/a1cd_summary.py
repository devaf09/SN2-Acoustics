"""実験 A1c・A1d の seed をまとめる（平均 ± 標準誤差）。出力: results/a1cd_summary.txt"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def ms(xs, f="{:.1f}", scale=100):
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f.format(xs[0])
    return (f + "±" + f).format(st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)))


lines = []
for exp in ("a1c", "a1d", "a1d_f8b6"):
    for task in ("order", "freq"):
        runs = [json.load(open(p, encoding="utf-8"))
                for p in sorted(glob.glob(os.path.join(ROOT, "results", exp, task, "seed*", "route.json")))]
        if not runs:
            continue
        lines.append(f"## {exp} 課題 {task}  seed {[r['seed'] for r in runs]}（専門家 4 人・16 クラス、平均 ± 標準誤差）")
        if exp.startswith("a1d"):
            lines.append(f"後段単独の正解率 {ms([st.mean(r['solo']) for r in runs])}%、学習する数 {runs[0]['params']}")
        lines.append("| 方式 | 全体の正解率 % | 振り分け top1 % | FFT/入力 | 計算量 % | 実測 ms/入力 |")
        lines.append("|---|---|---|---|---|---|")
        for name in runs[0]["results"]:
            v = [r["results"][name] for r in runs]
            lines.append(f"| {name} | {ms([x['acc'] for x in v])} | {ms([x.get('route_top1') for x in v])} | {v[0]['ffts']} | "
                         f"{v[0]['ffts_pct']:.1f} | {ms([x.get('ms') for x in v], '{:.3f}', 1)} |")
        lines.append("")
txt = "\n".join(lines)
print(txt)
open(os.path.join(ROOT, "results", "a1cd_summary.txt"), "w", encoding="utf-8").write(txt + "\n")

"""実験 A1e の seed をまとめる（平均 ± 標準誤差）。出力: results/a1e_summary.txt"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def ms(xs):
    xs = [100 * x for x in xs]
    if len(xs) == 1:
        return f"{xs[0]:.1f}"
    return f"{st.mean(xs):.1f}±{st.stdev(xs) / math.sqrt(len(xs)):.1f}"


lines = []
for task in ("order", "freq"):
    runs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(ROOT, "results", "a1e", task, "seed*", "route.json")))]
    if not runs:
        continue
    R = lambda name: [r["results"][name] for r in runs]
    lines += [f"## 課題 {task}  seed {[r['seed'] for r in runs]}（平均 ± 標準誤差）",
              f"5 人目の後段だけの正解率（自分の 4 クラスの中）: {ms([v['acc'] for v in R('new-back-solo')])}%",
              "| 方式 | 全体 20 クラス % | 古いテスト % | 新しいテスト % | 振り分け top1 古い / 新しい % | 上位 m に正解 古い / 新しい % | 計算量（5 人全員=100%） | 取消で元どおり |",
              "|---|---|---|---|---|---|---|---|"]
    for m in (1, 2):
        v = R(f"before4-top{m}")
        lines.append(f"| 足す前（4 人）top{m} | - | {ms([x['acc'] for x in v])} | - | {ms([x['route_top1'] for x in v])} / - | "
                     f"{ms([x['route_topm'] for x in v])} / - | {v[0]['ffts']} 回（4 人全員の {v[0]['ffts_pct_of4']:.1f}%） | - |")
    for tag in ("replay100", "replay10"):
        for m in (1, 2):
            v = R(f"after5-{tag}-top{m}")
            lines.append(f"| 足した後 {tag} top{m}{' + jcal' if m == 2 else ''} | {ms([x['all']['acc'] for x in v])} | "
                         f"{ms([x['old']['acc'] for x in v])} | {ms([x['new']['acc'] for x in v])} | "
                         f"{ms([x['old']['route_top1'] for x in v])} / {ms([x['new']['route_top1'] for x in v])} | "
                         f"{ms([x['old']['route_topm'] for x in v])} / {ms([x['new']['route_topm'] for x in v])} | "
                         f"{v[0]['ffts']} 回（{v[0]['ffts_pct']:.1f}%） | {all(x['undo_identical'] for x in v)} |")
    for name in ("orig5-all-conf", "orig5-all-jcal"):
        v = R(name)
        lines.append(f"| 比べる相手: 元の専門家 5 人全員 {name[10:]} | {ms([x['acc'] for x in v])} | {ms([x['old'] for x in v])} | "
                     f"{ms([x['new'] for x in v])} | - | - | {v[0]['ffts']} 回（100%） | - |")
    lines.append("")
txt = "\n".join(lines)
print(txt)
open(os.path.join(ROOT, "results", "a1e_summary.txt"), "w", encoding="utf-8").write(txt + "\n")

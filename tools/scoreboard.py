"""SN2-Acoustics の到達度を「ポイント」で表す（合計 100 点）。

    python tools/scoreboard.py            # 表を表示
    python tools/scoreboard.py --md       # docs/scoreboard.md を書き出す

考え方（Sin-CNN2 の tools/scoreboard.py と同じ）
  各項目に「ゼロ点の基準（何もしない場合: 偶然・ランダムに送るなど）」と「目標（最低限の指標: 定番の手法・全員を動かす最良の値）」を置き、
  今の値がその間のどこにあるかで採点する（目標に届けば満点、越えても満点）。
    点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)
  小さいほど良い項目（計算量、時間など）も同じ式でそのまま扱える（基準と目標の大小が逆になるだけ）。
  時間のように桁で変わるものは、対数（log10）で採点する。
新しい結果が出たら、下の ITEMS の「今の値」と「出典」を書き換えて再実行する。未測定の項目は None（0 点）で残す。
"""
import math
import sys

# (軸, 項目, 配点, ゼロ点の基準, 目標, 今の値, 対数で採点, 目標の根拠, 今の値の出典)。今の値が None なら未測定（0 点）
ITEMS = [
    ("振り分け", "順序課題の全体の正解率（専門家 4 人・16 クラス、%）", 15, 24.5, 97.3, 98.3, False,
     "元の専門家を全員動かして自信で選ぶ（実験02）", "実験04 前段 8 + 後段 6、上位 2 人 + 較正"),
    ("振り分け", "頻度課題の全体の正解率（専門家 4 人・16 クラス、%）", 10, 23.9, 93.7, 94.2, False,
     "元の専門家を全員動かして較正（実験02）", "実験04 前段 8 + 後段 6、上位 2 人 + 較正"),
    ("振り分け", "計算量（FFT 回数、全員を動かす = 100%。小さいほど良い）", 15, 100.0, 35.0, 32.6, False,
     "全員の 35% 以下", "実験04 前段 8 + 後段 6、上位 2 人"),
    ("振り分け", "振り分けの正解率・順序課題（上位 1、%）", 10, 24.5, 95.0, 90.9, False,
     "95%", "実験04 共通の前段 8 枚"),
    ("振り分け", "振り分けの正解率・頻度課題（上位 1、%）", 5, 24.5, 95.0, 94.1, False,
     "95%", "実験04 共通の前段 8 枚"),
    ("組み立て", "後から 1 人足したときの全体の正解率・順序課題（5 人・20 クラス、%）", 10, 20.0, 96.1, 91.0, False,
     "元の専門家 5 人を全員動かして自信で選ぶ（実験05）", "実験05 古いデータ 10% を混ぜる、上位 2 人 + 較正"),
    ("組み立て", "後から 1 人足したときの全体の正解率・頻度課題（5 人・20 クラス、%）", 5, 20.0, 93.0, 92.6, False,
     "元の専門家 5 人を全員動かして較正（実験05）", "実験05 古いデータ全部を混ぜる、上位 2 人 + 較正"),
    ("組み立て", "取消の厳密さ（外して元の答えに 1 件残らず戻るか）", 5, 0.0, 1.0, 1.0, False,
     "完全に戻る", "実験05（全 seed・両課題）"),
    ("組み立て", "学習する数（元の専門家 4 人の合計 8260 に対する比。小さいほど良い）", 5, 1.0, 0.5, 4169 / 8260, False,
     "元の半分", "実験04 前段 8 + 後段 6（4169 個）"),
    ("効率", "推論の速さ（µs/入力、GPU、まとめて流す。小さいほど良い）", 5, 1000.0, 1.0, 84.0, True,
     "1µs（Sin-CNN2 と同じ目標）", "実験04 前段 8 + 後段 6、上位 2 人（0.084 ms）"),
    ("効率", "1 グレインの学習時間（秒、GPU。小さいほど良い）", 5, 3600.0, 10.0, 92.0, True,
     "10 秒", "実験01 専門家 1 人（板 16 枚、1500 ステップ、約 92 秒）"),
    ("2D", "2 次元の入力（画像など）のグレインで同じ組み立てができるか（1D の 3 つの目標を満たした課題の割合）", 10, 0.0, 1.0, 0.5, False,
     "2D 用の部品で、振り分け・共有・追加が 1D と同じ水準",
     "実験06・07: 合成 4 課題は 3 目標とも達成（R2 板 4 枚 87.1% 対 85.6%、計算量 31.8%、振り分け 95.2%）。"
     "Fashion は全員を動かす最良を計算量 18% で上回るが、振り分けの正解率 83〜89% と、1 本で全クラス（84.5%）に届かない → 0.5。実験08: Fashion で後から足すのは全データの学び直しに 1.3〜2 点差（取消は完全）、合成は全員を動かす方式に 3 点負け（据え置き 0.5）"),
]


def score(lo, hi, cur, use_log):
    if cur is None:
        return 0.0
    if use_log:
        lo, hi, cur = math.log10(lo), math.log10(hi), math.log10(cur)
    r = (cur - lo) / (hi - lo)
    return max(0.0, min(1.0, r))


def table():
    rows, axes = [], {}
    for axis, name, w, lo, hi, cur, lg, why, src in ITEMS:
        r = score(lo, hi, cur, lg)
        rows.append((axis, name, w, lo, hi, cur, r * w, why, src))
        a = axes.setdefault(axis, [0.0, 0.0, 0.0])
        a[0] += r * w
        a[1] += w
        a[2] += w if cur is not None else 0
    return rows, axes


def fmt(x):
    return "未測定" if x is None else (f"{x:.3g}" if isinstance(x, float) else f"{x:g}")


def markdown():
    rows, axes = table()
    total = sum(v[0] for v in axes.values())
    measured = sum(v[2] for v in axes.values())
    out = ["# SN2-Acoustics の到達度（ポイント）", "",
           "各項目に「ゼロ点の基準」（何もしない場合）と「目標」（最低限の指標＝定番の手法や、専門家を全員動かす一番良い方式の値）を置き、"
           "今の値がその間のどこにあるかで採点しています。目標に届けば満点です。",
           "計算式と数値は `tools/scoreboard.py` にあり、新しい結果が出たら書き換えて `python tools/scoreboard.py --md` で更新します。", "",
           "  点 = 配点 × clip((今の値 − ゼロ点の基準) / (目標 − ゼロ点の基準), 0, 1)　（時間は log10 で）", "",
           f"## 合計: **{total:.1f} / 100 点**（測定済みの項目だけなら {total:.1f} / {measured:.0f} 点）", "",
           "| 軸 | 点 | 配点 | 達成率 |", "|---|---|---|---|"]
    for axis, (got, w, mw) in axes.items():
        out.append(f"| {axis} | {got:.1f} | {w:g} | {100 * got / w:.0f}% |")
    out += ["", "## 項目ごと", "",
            "| 軸 | 項目 | 配点 | ゼロ点の基準 | 目標 | 今の値 | 点 | あと | 目標の根拠 | 出典 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for axis, name, w, lo, hi, cur, got, why, src in rows:
        out.append(f"| {axis} | {name} | {w:g} | {fmt(lo)} | {fmt(hi)} | {fmt(cur)} | {got:.1f} | {w - got:.1f} | {why} | {src} |")
    rest = sorted(rows, key=lambda r: -(r[2] - r[6]))[:5]
    out += ["", "## 伸びしろの大きい項目（あと何点取れるか）", ""]
    for axis, name, w, lo, hi, cur, got, why, src in rest:
        if w - got > 0.05:
            out.append(f"- **{name}**（{axis}）: あと {w - got:.1f} 点")
    out += ["", "目標・配点の決め方と過去の表は [legacy/scoreboard_until_exp05.md](legacy/scoreboard_until_exp05.md)（実験05 までの手書きの表）も参照。"]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md = markdown()
    if "--md" in sys.argv:
        import pathlib
        p = pathlib.Path(__file__).resolve().parents[1] / "docs" / "scoreboard.md"
        p.write_text(md, encoding="utf-8")
        print("書き出し:", p)
    else:
        print(md)

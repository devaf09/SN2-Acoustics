# experiments/legacy/ — 形式をそろえる前の集計スクリプト（記録として残す）

2026-10-17 夜に、記録を Sin-CNN2 と同じ形式（実験番号、results/expNN.txt の 1 行 JSON）に移した。
ここにあるのは、移行前の集計スクリプトで、移行前の結果の置き場所（results/a1/... など。今は results/legacy/ の下）を前提にしている。
そのままでは動かない。今の集計は `python tools/summarize.py`（移行前のまとめと平均値が一致することは `--check` で確かめられる）。

| ファイル | 対応する実験 |
|---|---|
| a1_summary.py | 実験01（旧 A1） |
| a1b_summary.py | 実験02（旧 A1b） |
| a1cd_summary.py | 実験03・04（旧 A1c・A1d） |
| a1e_summary.py | 実験05（旧 A1e） |

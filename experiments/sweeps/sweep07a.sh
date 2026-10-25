# リポジトリ直下で実行する: bash experiments/sweeps/sweep07a.sh（共通の GPU 待ち行列から）
# 実験07: 2D の共通の前段 + 後段（合成 4 課題）。前段 8 + 後段 6・8、前段 4 + 後段 12
# 見込み: 約 40 分。結果は results/exp07.txt（比べる相手は実験06）
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp07_2d_shared.py --task synth4 --seed $seed --configs 8:6,8:8,4:12 --device cuda
done

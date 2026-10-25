# リポジトリ直下で実行する: bash experiments/sweeps/sweep07b.sh（共通の GPU 待ち行列から）
# 実験07: 2D の共通の前段 + 後段（Fashion-MNIST、分け方 mix）。前段 8 + 後段 6・8
# 見込み: 約 50 分。結果は results/exp07.txt（比べる相手は実験06）
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp07_2d_shared.py --task fashion_mix --seed $seed --configs 8:6,8:8 --device cuda
done

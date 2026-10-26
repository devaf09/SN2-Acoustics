# リポジトリ直下で実行する: bash experiments/sweeps/sweep09a.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験09: 振り分け役（= 共通の前段）を強くする。板の途中に光の非線形（平方根）、板・区画を増やす。振り分け役だけを学習（課題 synth4）
# 見込み: 約 30 分（7 通り × 3 seed、実験07 の振り分け役の学習時間から）。結果は results/exp09.txt
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp09_2d_router.py --task synth4 --seed $seed --device cuda
done

# リポジトリ直下で実行する: bash experiments/sweeps/sweep06b.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験06: Fashion-MNIST を専門家 5 人 × 2 クラスに（分け方 mix）。データは SN2-Parts-2D/data（MIT、写さずに読む）
# 見込み: 約 1 時間（3000 ステップ × 専門家 5 人 + 1 本 + 振り分け役、3 seed）。結果は results/exp06.txt
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp06_2d_route.py --task fashion_mix --seed $seed --device cuda
done

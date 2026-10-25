# リポジトリ直下で実行する: bash experiments/sweeps/sweep06a.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験06: 2D の専門家を全員動かす方式と振り分け役（合成 4 課題 = 専門家 4 人・33 クラス）。担当なしの絵あり（0.3）と、なし（比較）
# 見込み: 約 1 時間（SN2-Parts-2D 実験00 の 1500 ステップ約 1 分から）。結果は results/exp06.txt
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp06_2d_route.py --task synth4 --seed $seed --device cuda
done
for seed in 0 1 2; do
  python experiments/exp06_2d_route.py --task synth4 --seed $seed --reject 0 --routers R2-d8 --flat 0 --device cuda
done

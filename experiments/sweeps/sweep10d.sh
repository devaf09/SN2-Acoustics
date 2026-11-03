# リポジトリ直下で実行する: bash experiments/sweeps/sweep10d.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験10 の追加（課題 synth4）: 10a・10b で前段を学び直すと古い後段が合わなくなった（古いクラスが下がる）ので、前段をつなぎ止める版を比べる。
#   K1f / K2f = 前段の光そのものもお手本に合わせる、K2b1 / K2b10 = 後段のお手本合わせを 10 倍 / 100 倍に
# 見込み: 約 1.5 時間（実験10a の実測 1 seed 19〜23 分 + 4 通り）。結果は results/exp10.txt（args.variants で区別）。終わった seed は飛ばす
export PYTHONIOENCODING=utf-8
V=K1f,K2f,K2b1,K2b10
for seed in 0 1 2; do
  if [ -f results/exp10.txt ] && grep -q "\"variants\": \"$V\".*\"task\": \"synth4\", \"seed\": $seed," results/exp10.txt; then
    echo "seed $seed は済み（飛ばす）"; continue
  fi
  python experiments/exp10_2d_consolidate.py --task synth4 --seed $seed --device cuda --variants $V
done

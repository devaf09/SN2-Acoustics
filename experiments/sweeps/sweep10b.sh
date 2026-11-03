# リポジトリ直下で実行する: bash experiments/sweeps/sweep10b.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験10: 記憶の固定化（課題 fashion_sim）。付け足した系（実験08 の thin・古いデータ 1 割 + 下駄）をお手本に、K1 前段だけ / K2 前段 + 後段 / K3 お手本なし で学び直す
# 見込み: 約 3〜3.5 時間（実験08 の実測 18〜20 分/seed + 固定化 3 通り 40 分前後/seed の見込み）。結果は results/exp10.txt。途中で止まっても、終わった seed は飛ばして続きから回る
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  if [ -f results/exp10.txt ] && grep -q "\"task\": \"fashion_sim\", \"seed\": $seed," results/exp10.txt; then
    echo "seed $seed は済み（飛ばす）"; continue
  fi
  python experiments/exp10_2d_consolidate.py --task fashion_sim --seed $seed --device cuda
done

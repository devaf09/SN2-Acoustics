# リポジトリ直下で実行する: bash experiments/sweeps/sweep10g.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験10 の追加 2: 固定化の後、受付の下駄（専門家ごと）を検証用データで合わせ直す（+bias）。10a〜e で、固定化すると受付が新しい人に偏り、
#   古い入力の受付が下がった（服 88 → 85%、合成 96 → 83〜90%）ため。K2 / K2f / K3 を服（別々）× 3 seed。+bias は評価だけ（学習は 10a〜e と同じ）
# 見込み: 約 2.3 時間（服（別々）、10f の続き、実験10a〜c の実測から）。結果は results/exp10.txt（args.variants で区別）。終わった分は飛ばす
export PYTHONIOENCODING=utf-8
V=K2,K2f,K3
for task in fashion_mix; do
  for seed in 0 1 2; do
    if [ -f results/exp10.txt ] && grep -q "\"variants\": \"$V\".*\"task\": \"$task\", \"seed\": $seed," results/exp10.txt; then
      echo "$task seed $seed は済み（飛ばす）"; continue
    fi
    python experiments/exp10_2d_consolidate.py --task $task --seed $seed --device cuda --variants $V
  done
done

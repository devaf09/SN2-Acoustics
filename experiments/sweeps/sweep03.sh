# リポジトリ直下で実行する: bash experiments/sweeps/sweep03.sh
# 実験03（旧 A1c）: 振り分け役（R2、板 8 枚・16 枚）だけ色の数 T を 8・4・2・1 に減らす。専門家 4 人
# 先に sweep01.sh・sweep02.sh で専門家を作っておく。結果は results/exp03.txt
export PYTHONIOENCODING=utf-8
for task in order freq; do
  for seed in 0 1 2; do
    python experiments/exp03_colors.py --task $task --seed $seed --device cuda --configs 8:8,8:4,8:2,8:1,16:4,16:2,16:1
  done
done

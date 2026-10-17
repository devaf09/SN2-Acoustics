# リポジトリ直下で実行する: bash experiments/sweeps/sweep04.sh
# 実験04（旧 A1d）: 振り分け役の板 8 枚を共通の前段にし、専門家は後段（8 枚・6 枚）だけ持つ
# 比べる相手（元の専門家を全員動かす）は実験02 の結果を使う。結果は results/exp04.txt
export PYTHONIOENCODING=utf-8
for back in 8 6; do
  for task in order freq; do
    for seed in 0 1 2; do
      python experiments/exp04_shared_front.py --task $task --seed $seed --front 8 --back $back --device cuda
    done
  done
done

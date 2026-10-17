# リポジトリ直下で実行する: bash experiments/sweeps/sweep05.sh
# 実験05（旧 A1e）: 実験04（前段 8・後段 6）の系に、前段を凍結したまま 5 人目の専門家を後から足す。取消も確かめる
# 比べる相手: 元の専門家（板 16 枚）5 人を全員動かす（5 人目はこのスクリプトの中で学習）。結果は results/exp05.txt
export PYTHONIOENCODING=utf-8
for task in order freq; do
  for seed in 0 1 2; do
    python experiments/exp05_add_expert.py --task $task --seed $seed --back 6 --device cuda
  done
done

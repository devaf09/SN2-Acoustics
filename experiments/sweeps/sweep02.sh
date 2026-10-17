# リポジトリ直下で実行する: bash experiments/sweeps/sweep02.sh
# 実験02（旧 A1b）: 順序だけで決まる課題での振り分け役と較正。頻度課題（実験01 の専門家）でも同じ比較をする
# 先に sweep01.sh で頻度課題の専門家を作っておく。結果は results/exp02.txt
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp01_experts.py --task order --seed $seed --device cuda
done
for task in order freq; do
  for seed in 0 1 2; do
    python experiments/exp02_order.py --task $task --seed $seed --device cuda
  done
done

# リポジトリ直下で実行する: bash experiments/sweeps/sweep01.sh
# 実験01（旧 A1）: 1 専門家の時間の計測、専門家 4 人の学習（頻度課題）、振り分け役の比較（専門家 4 人と 2 人）
# 比較 = 全員を動かして足す（avg）・gate、正解の専門家へ（上限）、ランダム（ゼロ点）、R1 水面の板、R2 薄い SinWave、文字頻度の回帰
# 2026-10-17 の実行: 時間は CPU（CUDA_VISIBLE_DEVICES=""）と GPU（RTX 3060）、ほかは GPU。結果は results/exp01.txt
export PYTHONIOENCODING=utf-8
CUDA_VISIBLE_DEVICES="" python experiments/exp01_time.py --device cpu
python experiments/exp01_time.py --device cuda --steps 10
for seed in 0 1 2; do
  python experiments/exp01_experts.py --task freq --seed $seed --device cuda
  python experiments/exp01_route.py --seed $seed --device cuda
done

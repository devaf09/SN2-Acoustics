# リポジトリ直下で実行する: bash experiments/sweeps/sweep08a.sh（共通の GPU 待ち行列から。GPU の番号は指定しない）
# 実験08: 2D で、共通の前段を凍結したまま専門家を後から 1 人足す（課題 synth4、前段 8 + 後段 6）。新しい出口 plate / thin、古いデータ 100% / 10%、下駄の補正
# 比べる相手に実験06 の元の専門家（results/models2d/）を使うので、sweep06 の後に回す。見込み: 約 45 分（実験07 の実測から）。結果は results/exp08.txt
export PYTHONIOENCODING=utf-8
for seed in 0 1 2; do
  python experiments/exp08_2d_add.py --task synth4 --seed $seed --device cuda
done

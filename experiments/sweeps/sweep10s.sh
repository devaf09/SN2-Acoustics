# リポジトリ直下で実行する: bash experiments/sweeps/sweep10s.sh（共通の GPU 待ち行列から）
# 実験10 の試運転: 合成課題・seed 0 をごく短いステップで回し、プログラムが最後まで動くかだけを確かめる（数分）。結果は results/exp10_smoke.txt（本番の記録には入れない）
export PYTHONIOENCODING=utf-8
python experiments/exp10_2d_consolidate.py --task synth4 --seed 0 --steps 60 --csteps 30 --n_train 400 --n_generic 300 --device cuda --out results/exp10_smoke.txt

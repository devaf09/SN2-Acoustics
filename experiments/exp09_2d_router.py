"""実験09: 2D の振り分け役（共通の前段にもなる）を強くする。板の途中に光の非線形（平方根）をはさむ、板・区画を増やす。

着想: 実験06〜08 で、Fashion-MNIST の正解率の天井は振り分けの正解率（83〜91%）だった（専門家は担当の中で 93〜99.6%）。
      SN2-Parts-2D では、段の間に「明るさを測って平方根の振幅で光を出し直す」非線形をはさむと見分けが強くなった（実験28・75〜77）。
      振り分け役（= 実験07・08 の共通の前段）に同じことをすると、振り分けが上がるか。
方法: 振り分け役だけを学習して、振り分けの正解率（上位 1・上位 2 に正解が入る割合）と計算量を測る（専門家・後段は学習しない。安い）。
      変える: 板の枚数（8 / 12 / 16）、非線形の位置（なし / 4 枚目の後 / 2・4・6 枚目の後 / 4・8 枚目の後 …）、区画（8 / 16）。
比べるもの: 実験06・07 の振り分け役（板 8 枚、区画 8、非線形なし。この中でも同じ条件を学習し直す）。参考: 1 本で全クラスの正解率（実験06）。
計算量: 2 次元 FFT の回数（板 1 枚 2 回、レンズ 1 回。非線形は FFT なし）。元の専門家（板 16 枚）1 人 = 33 回。

使い方: python experiments/exp09_2d_router.py --task fashion_mix --seed 0 --device cuda
出力: results/exp09.txt に 1 行の JSON を追記（振り分け役 1 つ = 1 行）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import sn2_2d as s2
from sn2_io import append_result, device_name

VARIANTS = {   # 名前: (板の数, 区画, 非線形を入れる板の番号)
    "d8": (8, 8, ()),
    "d8-nl3": (8, 8, (3,)),
    "d8-nl135": (8, 8, (1, 3, 5)),
    "d12-nl37": (12, 8, (3, 7)),
    "d16-nl3711": (16, 8, (3, 7, 11)),
    "d8p16": (8, 16, ()),
    "d8p16-nl3": (8, 16, (3,)),
}

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="fashion_mix", choices=["synth4", "fashion_mix", "fashion_sim"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=0, help="0 なら synth4 は 1500、fashion は 3000")
ap.add_argument("--variants", default=",".join(VARIANTS))
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--n_train", type=int, default=4000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
a = ap.parse_args()
DEV = a.device
steps = a.steps or (1500 if a.task == "synth4" else 3000)
D = s2.load_task(a.task, a.seed, L=a.L, n_train=a.n_train)
K = len(D["owners"])
e_te = D["ete"]
N = len(e_te)
FFT_E = s2.n_ffts(16)
for name in a.variants.split(","):
    t0 = time.time()
    d, pt, nl = VARIANTS[name]
    torch.manual_seed(a.seed)
    r = s2.NLNet2D(a.L, K, plates=["xy"] * d, patches=pt, seed=a.seed * 100 + 70 + d, nl_after=nl).to(DEV)
    if not nl:
        with torch.no_grad():
            u = s2.field(D["Xte"][:32].to(DEV))
            assert torch.allclose(r(u), p2_out := s2.p2.Net2D.forward(r, u), atol=1e-4)
    s2.train_cls(r, D["Xtr"], D["etr"], steps, a.seed, DEV)
    r.eval()
    t_train = time.time() - t0
    Z = s2.scores(r, D["Xte"], DEV)
    top = Z.argsort(1, descending=True)
    ffts = s2.n_ffts(d)
    U = s2.field(D["Xte"].to(DEV))
    ms = s2.timed(lambda: [r(U[i:i + 500]) for i in range(0, N, 500)], DEV) / N * 1000
    rec = dict(route_top1=(top[:, 0] == e_te).float().mean().item(), route_top2=(top[:, :2] == e_te[:, None]).any(1).float().mean().item(),
               ffts=ffts, ffts_pct_of_all=100.0 * ffts / (K * FFT_E), ms=ms, params=r.n_params(), train_s=round(t_train, 1))
    append_result("09", {"script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed, "steps": steps,
                         "variant": name, "plates": d, "patches": pt, "nl_after": list(nl), "K": K, "n_test": N,
                         "device_name": device_name(DEV), "results": rec, "sec": round(time.time() - t0, 1)}, a.out)

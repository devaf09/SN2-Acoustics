"""実験07: 2 次元のグレインで、共通の前段 + 専門家ごとの後段（1 次元の実験04 の 2D 版）。

着想: 1 次元の実験04 で、振り分け役の板を全専門家の共通の前段にして、専門家は後段だけを持つと、全員を動かす最良と同等以上を
      計算量 32.6% で出せた。2D でも同じことが成り立つか（再開の計画）。2D の前段の光は「位置」の情報も持つので、位置で決まる課題で困らないかも見る。
方法: 振り分け役（Net2D、板 ① f 枚、区画 8、窓 = 専門家の数）を全員のデータで「どの専門家か」学習 → 凍結して前段に。
      前段を出た光（レンズの前）を、レンズ → 振り分け役の窓（振り分け）と、選んだ専門家の後段（Net2D、板 ① b 枚）へ。
      後段は、自分のデータの前段の光と、担当なしの一般の絵の前段の光（重み 0.3）で学習。
比べるもの: 実験06 の「元の専門家（板 16 枚）を全員動かす最良」（集計で突き合わせる）。この中では、後段を全員動かして自信で選ぶ・較正。
答え方: 上位 1 人、上位 2 人 + 較正（生の点数、検証用データで α・τ_k・b_k）。
計算量: 前段 2f + 振り分けのレンズ 1 + 後段 1 人 (2b + 1)。元の専門家（板 16 枚、33 回）を全員動かす = 100%。

使い方: python experiments/exp07_2d_shared.py --task synth4 --seed 0 --configs 8:6,8:8 --device cuda
出力: results/exp07.txt に 1 行の JSON を追記（前段・後段の組ごとに 1 行）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import sn2_2d as s2
from sn2_2d import p2, t2
from sn2_io import append_result, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="synth4", choices=["synth4", "fashion_mix", "fashion_sim"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=0, help="0 なら synth4 は 1500、fashion は 3000")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--configs", default="8:6,8:8", help="前段の板の数:後段の板の数 の並び")
ap.add_argument("--reject", type=float, default=0.3)
ap.add_argument("--n_train", type=int, default=4000)
ap.add_argument("--full_plates", type=int, default=16, help="比べる元の専門家の板の数（計算量の 100%% の基準）")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
a = ap.parse_args()
DEV = a.device
steps = a.steps or (1500 if a.task == "synth4" else 3000)
torch.manual_seed(a.seed)
D = s2.load_task(a.task, a.seed, L=a.L, n_train=a.n_train)
K, C, owners, L = len(D["owners"]), D["C"], D["owners"], a.L
G = t2.make_generic(4000, L=L, seed=a.seed + 999) if a.reject > 0 else None
FFT_FULL = K * s2.n_ffts(a.full_plates)
cols, off = [], 0
for o in owners:
    cols.append(list(range(off, off + len(o))))
    off += len(o)
col2class = torch.tensor([c for o in owners for c in o])
class2col = torch.empty(C, dtype=torch.long)
class2col[col2class] = torch.arange(len(col2class))
y_te, e_te, y_va = D["yte"], D["ete"], D["yva"]
y_va_col = class2col[y_va]
N = len(y_te)
acc = lambda pc: (col2class[pc] == y_te).float().mean().item()
ones, zeros = torch.ones(K), torch.zeros(K)

routers = {}
for cfg in a.configs.split(","):
    T_START = time.time()
    f, b = map(int, cfg.split(":"))
    log = []
    if f not in routers:
        t0 = time.time()
        r = p2.Net2D(L, K, plates=["xy"] * f, patches=a.patches, seed=a.seed * 100 + 70 + f).to(DEV)
        s2.train_cls(r, D["Xtr"], D["etr"], steps, a.seed, DEV)
        r.eval()
        for p in r.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            u = s2.field(D["Xte"][:64].to(DEV))
            assert torch.allclose(r(u), s2.lens_scores(r, s2.front_field(r, u)), atol=1e-3), "前段 + レンズが forward と一致しない"

            def ff(X, bs=500):
                return torch.cat([s2.front_field(r, s2.field(X[i:i + bs].to(DEV))).cpu() for i in range(0, len(X), bs)])
            Ftr, Fte, Fva = ff(D["Xtr"]), ff(D["Xte"]), ff(D["Xva"])
            FG = ff(G) if G is not None else None
            Zte = torch.cat([s2.lens_scores(r, Fte[i:i + 500].to(DEV)).cpu() for i in range(0, N, 500)])
            Zva = torch.cat([s2.lens_scores(r, Fva[i:i + 500].to(DEV)).cpu() for i in range(0, len(Fva), 500)])
        routers[f] = (r, Ftr, Fte, Fva, FG, Zte, Zva, time.time() - t0)
    r, Ftr, Fte, Fva, FG, Zte, Zva, t_router = routers[f]
    log.append(f"前段 {f} 枚（振り分け役）の学習と前段の光の計算 {t_router:.0f} 秒、振り分けの正解率 {(Zte.argmax(1) == e_te).float().mean():.3f}")
    backs, solo = [], []
    for k in range(K):
        t0 = time.time()
        net = p2.Net2D(L, C, plates=["xy"] * b, patches=a.patches, seed=a.seed * 100 + 80 + k).to(DEV)
        m = D["etr"] == k
        p2.train(net, Ftr[m], D["ytr"][m], steps=steps, seed=a.seed * 100 + k, reject=FG, reject_weight=a.reject, schedule="cosine")
        net.eval()
        backs.append(net)
        mt = e_te == k
        Sk = s2.scores(net, Fte[mt], DEV, owners[k], prep=lambda v: v)
        solo.append((torch.tensor(owners[k])[Sk.argmax(1)] == y_te[mt]).float().mean().item())
        log.append(f"後段 {k}: 単独 {solo[-1]:.3f}、{time.time() - t0:.0f} 秒")
    S_te = [s2.scores(n_, Fte, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]
    S_va = [s2.scores(n_, Fva, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]
    fft_front, fft_lens, fft_back = 2 * f, 1, s2.n_ffts(b)
    res = {}
    pred = s2.combine(None, S_te, torch.tensor(1.0), ones, zeros, False).argmax(1)
    res["shared-all-conf"] = dict(acc=acc(pred), ffts=fft_front + K * fft_back)
    _, tj, bj = s2.fit_joint(None, S_va, y_va_col)
    res["shared-all-jcal"] = dict(acc=acc(s2.combine(None, S_te, torch.tensor(1.0), tj, bj, True).argmax(1)), ffts=fft_front + K * fft_back)
    al, tau, bb = s2.fit_joint(Zva, S_va, y_va_col)
    top_all = Zte.argsort(1, descending=True)
    Xte_dev = s2.field(D["Xte"].to(DEV))
    for m in (1, 2):
        top = top_all[:, :m]
        sel = torch.zeros(N, K, dtype=torch.bool)
        sel[torch.arange(N)[:, None], top] = True
        sc = s2.combine(Zte, S_te, al, tau, bb, True) if m == 2 else s2.combine(Zte, S_te, torch.tensor(1.0), ones, zeros, False)
        pred = s2.select(sc, cols, sel)
        seld = sel.to(DEV)

        def fn():
            Fx = torch.cat([s2.front_field(r, Xte_dev[i:i + 500]) for i in range(0, N, 500)])
            for i in range(0, N, 500):
                s2.lens_scores(r, Fx[i:i + 500])
            for k, n_ in enumerate(backs):
                idx = seld[:, k].nonzero()[:, 0]
                for i in range(0, len(idx), 500):
                    n_(Fx[idx[i:i + 500]])
        res[f"shared-top{m}" + ("-jcal" if m == 2 else "")] = dict(
            acc=acc(pred), route_top1=(top[:, 0] == e_te).float().mean().item(),
            route_topm=(top == e_te[:, None]).any(1).float().mean().item(),
            ffts=fft_front + fft_lens + m * fft_back, ms=s2.timed(fn, DEV) / N * 1000)
    for v in res.values():
        v["ffts_pct"] = 100.0 * v["ffts"] / FFT_FULL
    params = r.n_params() + sum(n_.n_params() for n_ in backs)
    append_result("07", {"script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed, "steps": steps,
                         "front": f, "back": b, "K": K, "C": C, "n_test": N, "solo": solo, "params": params, "fft_full": FFT_FULL,
                         "device_name": device_name(DEV), "log": log, "results": res,
                         "sec": round(time.time() - T_START + (t_router if cfg == a.configs.split(",")[0] else 0), 1)}, a.out)
    for k, v in res.items():
        print(f"{cfg} {k:18s} acc {v['acc']:.3f}  FFT {v['ffts']} ({v['ffts_pct']:.1f}%)  ms {v.get('ms')}", flush=True)

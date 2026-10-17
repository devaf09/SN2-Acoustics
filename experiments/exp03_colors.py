"""実験03（旧 A1c）: 振り分け役 R2（薄い SinWave）だけ色の数 T を減らし、計算量を下げる。専門家は T=8 のまま。

振り分け役は自分の T で文字列を光にする（同じ文字列、色の標本だけ少ない光源）。FFT 回数 = T_r·(2·板の数 + 1)。
T=1 の虹は、全部の文字を干渉させる coh と同じになる。
比べる相手（同じ専門家で計算し直す）: 全員を動かして「自信で選ぶ」（all-conf）と「生の点数で較正して最大」（all-jcal）。
答え方: 上位 1 人（較正は答えを変えない）と、上位 2 人 + 較正 jcal（A1b と同じ式）。

使い方: python experiments/a1c_route.py --task order --seed 0 --device cuda
着想: 実験02 で順序を見分けられたのは薄い SinWave（R2）だけだが、板 8 枚で計算量 37.9% と目標（35%）を少し超えた。色の数 T を減らして計算を減らせるか（実験02 の「次の案」）。
出力: results/exp03.txt に 1 行の JSON を追記
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
import torch.nn.functional as F

from vendor import sinwave_minimal_v1_1_0 as sw
from sn2_tasks import get_skills
from sn2_io import append_result, models_dir, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="order")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--device", default="cpu")
ap.add_argument("--configs", default="8:8,8:4,8:2,8:1,16:4,16:2,16:1", help="板の数:T の並び")
ap.add_argument("--out", default=None, help="結果の 1 行 JSON の追記先（既定は results/expNN.txt）")
a = ap.parse_args()
T_START = time.time()
DEV = a.device
torch.manual_seed(a.seed)
EXP_DIR = models_dir(a.task, a.seed)
K = 4
experts = [sw.load(os.path.join(EXP_DIR, f"expert{i}.pt"), device=DEV).eval() for i in range(K)]
skills = get_skills(a.task, K, a.seed)
E0 = experts[0]
FFT_EXPERT = E0.T * (2 * E0.A.shape[0] + 1)
C = 4 * K


def sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


def split(j):
    return ([x for sk in skills for x in sk[j][0]], [k for k, sk in enumerate(skills) for _ in sk[j][0]],
            [y for sk in skills for y in sk[j][1]])


xtr, etr, ytr = split(0)
xte, ete, yte = split(1)
xva, eva, yva = split(2)
ete, yte, eva, yva = map(torch.tensor, (ete, yte, eva, yva))
N = len(xte)


@torch.no_grad()
def expert_scores(U, bs=512):
    out = []
    for net in experts:
        own = torch.tensor(net.own, device=U.device)
        out.append(torch.cat([net(U[i:i + bs])[:, own] for i in range(0, len(U), bs)]).cpu())
    return torch.stack(out, 1)        # (N, K, 4)


@torch.no_grad()
def timed_pipeline(U, sel, bs=256, reps=3):
    best = float("inf")
    sel = sel.to(U.device)
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for k, net in enumerate(experts):
            idx = sel[:, k].nonzero()[:, 0]
            for i in range(0, len(idx), bs):
                net(U[idx[i:i + bs]])
        sync()
        best = min(best, time.perf_counter() - t0)
    return best / len(U) * 1000


def combine(Zr, S, alpha, tau, b, raw):
    lp = torch.log_softmax(alpha * Zr, 1) if Zr is not None else torch.zeros(S.shape[0], S.shape[1])
    ls = S / tau[None, :, None] if raw else torch.log_softmax(S / tau[None, :, None], 2)
    return (lp[:, :, None] + ls + b[None, :, None]).reshape(S.shape[0], -1)


def fit_joint(Zr, S, y, iters=500):
    la, lt, b = torch.zeros(1, requires_grad=True), torch.zeros(K, requires_grad=True), torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([la, lt, b], lr=0.05)
    for _ in range(iters):
        loss = F.cross_entropy(combine(Zr, S, la.exp(), lt.exp(), b, raw=True), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return la.exp().detach(), lt.exp().detach(), b.detach()


Ute, Uva = E0.encode(xte), E0.encode(xva)
S, Sva = expert_scores(Ute), expert_scores(Uva)
owner = torch.arange(C) // 4
acc = lambda sc: (sc.argmax(1) == yte).float().mean().item()
t_all = timed_pipeline(Ute, torch.ones(N, K, dtype=torch.bool))
ones, zeros = torch.ones(K), torch.zeros(K)
res = {}
res["all-conf"] = dict(acc=acc(combine(None, S, torch.tensor(1.0), ones, zeros, False)), route_top1=None,
                       ffts=K * FFT_EXPERT, ffts_pct=100.0, ms=t_all, extra_params=0)
_, tj, bj = fit_joint(None, Sva, yva)
res["all-jcal"] = dict(acc=acc(combine(None, S, torch.tensor(1.0), tj, bj, True)), route_top1=None,
                       ffts=K * FFT_EXPERT, ffts_pct=100.0, ms=t_all, extra_params=0)

for cfg in a.configs.split(","):
    d, Tr = map(int, cfg.split(":"))
    t0 = time.time()
    net = sw.SinWaveNet(n_slots=K, layers=d, patches=64, T=Tr, seed=a.seed * 100 + 50 + d).to(DEV)
    sw.train(net, xtr, etr, steps=a.steps, seed=a.seed)
    net.eval()
    t_train = time.time() - t0
    with torch.no_grad():
        Rte, Rva = net.encode(xte), net.encode(xva)
        z = lambda U: torch.cat([net(U[i:i + 256]) for i in range(0, len(U), 256)]).cpu()
        Zte, Zva = z(Rte), z(Rva)
        sync()
        t1 = time.perf_counter()
        z(Rte)
        sync()
        t_router = (time.perf_counter() - t1) / N * 1000
    ffts_r = Tr * (2 * d + 1)
    al, tau, b = fit_joint(Zva, Sva, yva)
    top_all = Zte.argsort(1, descending=True)
    for m in (1, 2):
        top = top_all[:, :m]
        sel = torch.zeros(N, K, dtype=torch.bool)
        sel[torch.arange(N)[:, None], top] = True
        sc = combine(Zte, S, al, tau, b, raw=True) if m == 2 else combine(Zte, S, torch.tensor(1.0), ones, zeros, False)
        pred = sc.masked_fill(~sel.repeat_interleave(4, 1), -1e9).argmax(1)
        ffts = ffts_r + m * FFT_EXPERT
        res[f"R2-d{d}T{Tr}-top{m}" + ("-jcal" if m == 2 else "")] = dict(
            acc=(pred == yte).float().mean().item(), route_top1=(top[:, 0] == ete).float().mean().item(),
            route_topm=(top == ete[:, None]).any(1).float().mean().item(), ffts=ffts, ffts_pct=100.0 * ffts / (K * FFT_EXPERT),
            ms=t_router + timed_pipeline(Ute, sel), router_ms=t_router, router_ffts=ffts_r,
            extra_params=sum(p.numel() for p in net.parameters() if p.requires_grad), train_s=t_train)
    print(f"  {a.task} seed {a.seed} d{d} T{Tr}: 振り分けの正解率 {res[f'R2-d{d}T{Tr}-top1']['route_top1']:.3f}, "
          f"学習 {t_train:.0f} s", flush=True)

out = {"task": a.task, "seed": a.seed, "steps": a.steps, "device_name": device_name(DEV),
       "K": K, "fft_per_expert": FFT_EXPERT, "results": res}
append_result("03", {"script": os.path.basename(__file__), "args": vars(a), **out, "sec": round(time.time() - T_START, 1)}, a.out)
lines = [f"{a.task} seed {a.seed} K={K}"]
for k, v in res.items():
    lines.append(f"{k:20s} acc {v['acc']:.3f}  route1 {v['route_top1'] if v['route_top1'] is None else round(v['route_top1'], 3)}  "
                 f"FFT {v['ffts']} ({v['ffts_pct']:.1f}%)  ms {v['ms']:.3f}  params+ {v['extra_params']}")
txt = "\n".join(lines)
print(txt)

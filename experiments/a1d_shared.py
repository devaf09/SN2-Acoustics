"""実験 A1d: 振り分け役と専門家で、前半の板を共有する。

1. 振り分け役 R（SinWaveNet、板 8 枚・区画 64、T=8、窓 = 専門家の数）を全員のデータで「どの専門家か」学習する。
2. R の板 8 枚を「共通の前段」として凍結する。前段を出た光は、
   - レンズに通して R の窓で見る → 振り分け（追加の FFT は T 回だけ）
   - そのまま専門家の「後段」（板 8 枚・区画 64）に入れる
3. 専門家の後段は、自分のデータの前段を出た光だけで学習する（1500 ステップ、バッチ 256、学習率 0.01。sw.train と同じ）。
FFT 回数（1 入力）: 前段 2·8·T = 128、振り分けのレンズ T = 8、後段 1 人 T·(2·8+1) = 136。
上位 1 人で 272 回 = 元の専門家 4 人を全員動かす（1056 回）の 25.8%。上位 2 人で 408 回 = 38.6%。

比べる相手: A1b・A1c と同じ、元の専門家（板 16 枚）を全員動かして「自信で選ぶ」「較正して最大」。

使い方: python experiments/a1d_shared.py --task order --seed 0 --device cuda
出力: results/a1d/{task}/seed{seed}/route.json と route.txt（前段 8・後段 8 以外は results/a1d_f{前段}b{後段}/...）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
import torch.nn.functional as F

from vendor import sinwave_minimal_v1_1_0 as sw
from sn2_tasks import get_skills

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="order")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--front", type=int, default=8)
ap.add_argument("--back", type=int, default=8)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
DEV = a.device
torch.manual_seed(a.seed)
OUT = os.path.join(ROOT, "results", "a1d" if (a.front, a.back) == (8, 8) else f"a1d_f{a.front}b{a.back}", a.task, f"seed{a.seed}")
os.makedirs(OUT, exist_ok=True)
K, C = 4, 16
skills = get_skills(a.task, K, a.seed)


def split(j):
    return ([x for sk in skills for x in sk[j][0]], [k for k, sk in enumerate(skills) for _ in sk[j][0]],
            [y for sk in skills for y in sk[j][1]])


xtr, etr, ytr = split(0)
xte, ete, yte = split(1)
xva, eva, yva = split(2)
etr_t, ytr_t = torch.tensor(etr), torch.tensor(ytr)
ete, yte, eva, yva = map(torch.tensor, (ete, yte, eva, yva))
N = len(xte)


def sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


def front_field(net, u):
    """net の板を全部通した光（レンズの前）。sw.SinWaveNet.intensity の板の部分と同じ計算。"""
    for d in range(net.A.shape[0]):
        p = net.patch
        phase = net.A[d, p] * torch.sin(net.w[d, p] * net.t + net.phi[d, p])
        u = u * torch.polar(torch.ones_like(phase), phase)
        u = torch.fft.ifft(torch.fft.fft(u) * net.H)
        u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    return u


def lens_scores(net, u):
    """レンズ → 窓。sw.SinWaveNet.intensity の最後と forward と同じ計算。"""
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    inten = inten / (inten.mean(-1, keepdim=True) + 1e-12)
    return net.gain * (inten @ net.det.T) + net.bias


def train_on_fields(net, X, Y, steps, seed, lr=0.01, bs=256):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(Y), (min(bs, len(Y)),), generator=gen)
        loss = F.cross_entropy(net(X[i.to(X.device)]), Y[i].to(X.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net.eval()


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


t0 = time.time()
router = sw.SinWaveNet(n_slots=K, layers=a.front, patches=64, seed=a.seed * 100 + 70).to(DEV)
sw.train(router, xtr, etr, steps=a.steps, seed=a.seed)
router.eval()
for p in router.parameters():
    p.requires_grad_(False)
T = router.T
with torch.no_grad():
    enc = lambda xs: torch.cat([front_field(router, router.encode(xs[i:i + 512])) for i in range(0, len(xs), 512)])
    U64 = router.encode(xte[:64])
    assert torch.allclose(router(U64), lens_scores(router, front_field(router, U64)), atol=1e-4), "前段 + レンズが forward と一致しない"
    Ftr, Fte, Fva = enc(xtr), enc(xte), enc(xva)
    Zte = torch.cat([lens_scores(router, Fte[i:i + 512]) for i in range(0, N, 512)]).cpu()
    Zva = torch.cat([lens_scores(router, Fva[i:i + 512]) for i in range(0, len(xva), 512)]).cpu()
log = [f"前段（振り分け役）の学習 {time.time() - t0:.0f} s, 振り分けの正解率 {(Zte.argmax(1) == ete).float().mean():.3f}"]
print(log[-1], flush=True)

backs = []
for k in range(K):
    t0 = time.time()
    idx = (etr_t == k).nonzero()[:, 0]
    net = sw.SinWaveNet(n_slots=C, own=list(range(4 * k, 4 * k + 4)), layers=a.back, patches=64, seed=a.seed * 100 + 80 + k).to(DEV)
    train_on_fields(net, Ftr[idx.to(DEV)], ytr_t[idx], a.steps, a.seed * 100 + k)
    backs.append(net)
    log.append(f"後段 {k} の学習 {time.time() - t0:.0f} s")
    print(log[-1], flush=True)


@torch.no_grad()
def back_scores(Fx):
    out = []
    for net in backs:
        own = torch.tensor(net.own, device=DEV)
        out.append(torch.cat([net(Fx[i:i + 512])[:, own] for i in range(0, len(Fx), 512)]).cpu())
    return torch.stack(out, 1)


S, Sva = back_scores(Fte), back_scores(Fva)
solo = [(S[ete == k, k].argmax(1) + 4 * k == yte[ete == k]).float().mean().item() for k in range(K)]
ones, zeros = torch.ones(K), torch.zeros(K)
FFT_FULL = 4 * T * (2 * 16 + 1)
fft_front, fft_lens, fft_back = 2 * a.front * T, T, T * (2 * a.back + 1)
res = {}
acc = lambda sc: (sc.argmax(1) == yte).float().mean().item()
res["shared-all-conf"] = dict(acc=acc(combine(None, S, torch.tensor(1.0), ones, zeros, False)),
                              ffts=fft_front + K * fft_back, route_top1=None)
_, tj, bj = fit_joint(None, Sva, yva)
res["shared-all-jcal"] = dict(acc=acc(combine(None, S, torch.tensor(1.0), tj, bj, True)), ffts=fft_front + K * fft_back,
                              route_top1=None)
al, tau, b = fit_joint(Zva, Sva, yva)
top_all = Zte.argsort(1, descending=True)


@torch.no_grad()
def timed(sel, reps=3):
    """前段 + 振り分けのレンズ + 選んだ後段だけを動かす時間（符号化は除く、1 入力あたり ms）"""
    U = router.encode(xte)
    best = float("inf")
    sel = sel.to(DEV)
    for _ in range(reps):
        sync()
        t1 = time.perf_counter()
        Fx = torch.cat([front_field(router, U[i:i + 256]) for i in range(0, N, 256)])
        for i in range(0, N, 256):
            lens_scores(router, Fx[i:i + 256])
        for k, net in enumerate(backs):
            idx = sel[:, k].nonzero()[:, 0]
            for i in range(0, len(idx), 256):
                net(Fx[idx[i:i + 256]])
        sync()
        best = min(best, time.perf_counter() - t1)
    return best / N * 1000


for m in (1, 2):
    top = top_all[:, :m]
    sel = torch.zeros(N, K, dtype=torch.bool)
    sel[torch.arange(N)[:, None], top] = True
    sc = combine(Zte, S, al, tau, b, raw=True) if m == 2 else combine(Zte, S, torch.tensor(1.0), ones, zeros, False)
    pred = sc.masked_fill(~sel.repeat_interleave(4, 1), -1e9).argmax(1)
    res[f"shared-top{m}" + ("-jcal" if m == 2 else "")] = dict(
        acc=(pred == yte).float().mean().item(), route_top1=(top[:, 0] == ete).float().mean().item(),
        route_topm=(top == ete[:, None]).any(1).float().mean().item(), ffts=fft_front + fft_lens + m * fft_back, ms=timed(sel))
for v in res.values():
    v["ffts_pct"] = 100.0 * v["ffts"] / FFT_FULL
params = sum(p.numel() for p in router.parameters()) + sum(p.numel() for n in backs for p in n.parameters())
out = {"task": a.task, "seed": a.seed, "steps": a.steps, "front": a.front, "back": a.back, "solo": solo, "params": params,
       "fft_full": FFT_FULL, "log": log, "results": res}
json.dump(out, open(os.path.join(OUT, "route.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
lines = [f"{a.task} seed {a.seed}  前段 {a.front} 枚 + 後段 {a.back} 枚  後段単独 {['%.3f' % s for s in solo]}  学習する数 {params}"]
for k, v in res.items():
    lines.append(f"{k:18s} acc {v['acc']:.3f}  route1 {v['route_top1']}  FFT {v['ffts']} ({v['ffts_pct']:.1f}%)  ms {v.get('ms')}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(OUT, "route.txt"), "w", encoding="utf-8").write(txt + "\n")

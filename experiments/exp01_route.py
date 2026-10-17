"""実験01（旧 A1）の 2 段目: 学習済みの専門家（a1_experts.py）に、振り分け役を付けて比べる。

比べるもの（K = 4 人と 2 人）
  all-avg   : 全員を動かして足す（sw.merge。スコアの平均で全クラスから最大）
  all-gate  : 全員を動かし、担当の窓に入った光の割合 r_k で選ぶ（Sin-CNN2 sinwave/model.py の Parallel gate と同じ式）
  oracle    : 正解の専門家に必ず送る（上限）
  random    : ランダムな専門家に送る（ゼロ点）
  R1        : 水面の板。入力の光をレンズ（FFT）に通した面の明るさ I(f) に、専門家ごとの透過率 M_k(f)=sigmoid(θ_k(f)) を掛けた和
  R2-dN     : 薄い SinWave（板 N 枚・区画 16、窓 = 専門家の数）
  LR        : 文字頻度のロジスティック回帰（振り分け役として）
  LR-flat   : 文字頻度のロジスティック回帰（全クラスを直接分類する。参考）
振り分け役のあとは、選んだ上位 m 人（m=1,2）だけを動かし、
  最終の点数 = log(振り分けの確からしさ p_k) + 専門家 k の担当窓の中での log-softmax
で全クラスから最大のものを答える。

使い方: python experiments/a1_route.py --seed 0 [--steps 1500] [--threads 4]
着想: Sin-CNN2 の融合（全員を動かして足す）は計算が専門家の数だけ増える。担当だけを動かす「振り分け役」で計算を減らせるか（コーディネーターの到達点 A の案）。
専門家は exp01_experts.py で先に学習しておく（results/models/freq/seed*/）。
出力: results/exp01.txt に 1 行の JSON を追記（part="route"）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

from vendor import sinwave_minimal_v1_1_0 as sw
from vendor.sinwave_tasks import task_skills
from sn2_io import append_result, models_dir, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--Ks", default="4,2")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None, help="結果の 1 行 JSON の追記先（既定は results/expNN.txt）")
a = ap.parse_args()
T_START = time.time()
if a.threads:
    torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
D = models_dir("freq", a.seed)
DEV = a.device
experts_all = [sw.load(os.path.join(D, f"expert{i}.pt"), device=DEV) for i in range(4)]
for e in experts_all:
    e.eval()
skills_all = task_skills(4, seed=a.seed)
E0 = experts_all[0]
T, LAYERS = E0.T, E0.A.shape[0]
FFT_EXPERT = T * (2 * LAYERS + 1)        # 1 入力あたり: 板ごとに fft+ifft、最後にレンズ 1 回。T 色それぞれ


# ---------------------------------------------------------------------------------------------
# 振り分け役
# ---------------------------------------------------------------------------------------------
def lens_intensity(u):
    """入力の光 (N, T, L) をレンズに通した面の明るさ (N, L)（平均 1）。FFT は T 回。"""
    I = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1).abs().pow(2).mean(1)
    return I / (I.mean(-1, keepdim=True) + 1e-12)


class R1(nn.Module):
    """水面の板: r_k = mean_f sigmoid(θ_k(f))·I(f)。点数 = gain·r_k + b_k。"""

    def __init__(self, K, L, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.01 * torch.randn(K, L, generator=g))
        self.gain = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(K))

    def forward(self, u):
        I = lens_intensity(u)
        r = I @ torch.sigmoid(self.theta).T / I.shape[-1]
        return self.gain * r + self.bias


def char_feats(texts, vocab):
    X = torch.zeros(len(texts), len(vocab))
    for b, t in enumerate(texts):
        for c in t:
            if c in vocab:
                X[b, vocab[c]] += 1.0 / len(t)
    return X


def train_router(model, X, Y, steps, lr, seed, bs=256):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(Y), (min(bs, len(Y)),), generator=gen).to(X.device)
        loss = F.cross_entropy(model(X[i]), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------
# 専門家の出力（全員分をまとめて先に計算。時間は別に測る）
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def expert_outputs(net, U, bs=512):
    """戻り値: 担当窓の log-softmax (N, 4)、生のスコア (N, C)、担当窓の光の割合 r (N,)"""
    ls, sc, rr = [], [], []
    own = torch.tensor(net.own, device=U.device)
    for i in range(0, len(U), bs):
        inten = net.intensity(U[i:i + bs])
        f = inten @ net.det.T
        s = net.gain * f + net.bias
        sc.append(s)
        ls.append(torch.log_softmax(s[:, own], 1))
        rr.append(f[:, own].sum(1) / (f.sum(1) + 1e-9))
    return torch.cat(ls).cpu(), torch.cat(sc).cpu(), torch.cat(rr).cpu()


@torch.no_grad()
def sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


def timed_pipeline(experts, U, sel, router_fn=None, bs=256, reps=1):
    """選んだ専門家だけを実際に動かしたときの壁時計時間（符号化を除く、1 入力あたり ms）。
    sel: (N, K) の bool。router_fn: 振り分け役の計算（時間に含める）。"""
    best = float("inf")
    sel = sel.to(U.device)
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        if router_fn is not None:
            router_fn()
        for k, net in enumerate(experts):
            idx = sel[:, k].nonzero()[:, 0]
            for i in range(0, len(idx), bs):
                net(U[idx[i:i + bs]])
        sync()
        best = min(best, time.perf_counter() - t0)
    return best / len(U) * 1000


def run(K):
    experts = experts_all[:K]
    skills = skills_all[:K]
    C = 4 * K
    xtr = [x for (tr, _) in skills for x in tr[0]]
    etr = [k for k, (tr, _) in enumerate(skills) for _ in tr[0]]
    ytr = [y for (tr, _) in skills for y in tr[1]]
    xte = [x for (_, te) in skills for x in te[0]]
    ete = torch.tensor([k for k, (_, te) in enumerate(skills) for _ in te[0]])
    yte = torch.tensor([y for (_, te) in skills for y in te[1]])
    N = len(xte)
    Utr = E0.encode(xtr)
    Ute = E0.encode(xte)

    outs = [expert_outputs(e, Ute) for e in experts]
    LS = torch.stack([o[0] for o in outs], 1)          # (N, K, 4)
    SC = torch.stack([o[1] for o in outs], 1)          # (N, K, 16)
    Rg = torch.stack([o[2] for o in outs], 1)          # (N, K)
    owner = torch.arange(C) // 4                       # クラス → 専門家
    solo = [(LS[ete == k, k].argmax(1) + 4 * k == yte[ete == k]).float().mean().item() for k in range(K)]

    def final_from_router(logp, m):
        """logp: (N, K) 振り分けの log 確率。上位 m 人だけを動かし、log p + log-softmax で答える。"""
        top = logp.topk(m, 1).indices
        sel = torch.zeros(N, K, dtype=torch.bool)
        sel[torch.arange(N)[:, None], top] = True
        score = torch.full((N, C), -1e9)
        for k in range(K):
            s = logp[:, k:k + 1] + LS[:, k]
            score[:, 4 * k:4 * k + 4] = torch.where(sel[:, k:k + 1], s, torch.full_like(s, -1e9))
        pred = score.argmax(1)
        return dict(acc=(pred == yte).float().mean().item(),
                    route_top1=(top[:, 0] == ete).float().mean().item(),
                    route_topm=(top == ete[:, None]).any(1).float().mean().item()), sel

    res = {}
    ffts_all = K * FFT_EXPERT
    t_all = timed_pipeline(experts, Ute, torch.ones(N, K, dtype=torch.bool), reps=2)
    base_params = sum(n_params(e) for e in experts)

    # 全員を動かして足す（sw.merge）
    merged = sw.merge(experts)
    with torch.no_grad():
        avg = SC.mean(1)                                   # Merged.forward と同じ（スコアの平均）
        mask = torch.full((C,), float("-inf"))
        mask[merged.own] = 0
        pred = (avg[:, :C] + mask).argmax(1)
        chk = sw.predict(merged, xte[:64])
        assert chk == pred[:64].tolist(), "merge の再計算が sw.predict と一致しない"
    res["all-avg"] = dict(acc=(pred == yte).float().mean().item(),
                          route_top1=(owner[pred] == ete).float().mean().item(), route_topm=None,
                          ffts=ffts_all, ffts_pct=100.0, ms=t_all, params=base_params, extra_params=0)

    # 全員を動かして gate
    pg = Rg / Rg.sum(1, keepdim=True)
    score = (torch.log(Rg + 1e-9)[:, :, None] + LS).reshape(N, C)
    pred = score.argmax(1)
    res["all-gate"] = dict(acc=(pred == yte).float().mean().item(),
                           route_top1=(pg.argmax(1) == ete).float().mean().item(),
                           route_topm=(pg.topk(min(2, K), 1).indices == ete[:, None]).any(1).float().mean().item(),
                           ffts=ffts_all, ffts_pct=100.0, ms=t_all, params=base_params, extra_params=0)

    # 上限（正解の専門家へ）とゼロ点（ランダム）
    logp = torch.log(F.one_hot(ete, K).float() + 1e-12)
    r, sel = final_from_router(logp, 1)
    res["oracle-top1"] = dict(r, ffts=FFT_EXPERT, ffts_pct=100.0 / K, ms=timed_pipeline(experts, Ute, sel, reps=2),
                              params=base_params, extra_params=0)
    g = torch.Generator().manual_seed(a.seed)
    rnd = torch.randint(K, (N,), generator=g)
    logp = torch.log(F.one_hot(rnd, K).float() + 1e-12)
    r, sel = final_from_router(logp, 1)
    res["random-top1"] = dict(r, ffts=FFT_EXPERT, ffts_pct=100.0 / K, ms=timed_pipeline(experts, Ute, sel, reps=2),
                              params=base_params, extra_params=0)

    # 振り分け役を学習して比べる
    Etr = torch.tensor(etr)
    EtrD = Etr.to(DEV)
    routers = {}
    t0 = time.time()
    routers["R1"] = (train_router(R1(K, E0.L, a.seed).to(DEV), Utr, EtrD, a.steps, 0.05, a.seed), T)
    for d in (2, 4):
        net = sw.SinWaveNet(n_slots=K, layers=d, patches=16, seed=a.seed * 100 + 50 + d).to(DEV)
        sw.train(net, xtr, etr, steps=a.steps, seed=a.seed)
        routers[f"R2-d{d}"] = (net, T * (2 * d + 1))
    vocab = {c: i for i, c in enumerate(sorted(set("".join(xtr))))}
    Ftr, Fte = char_feats(xtr, vocab), char_feats(xte, vocab)
    lr_model = train_router(nn.Linear(len(vocab), K), Ftr, Etr, a.steps, 0.05, a.seed)
    print(f"  K={K}: 振り分け役の学習 {time.time() - t0:.0f} s", flush=True)

    for name, (model, ffts_r) in list(routers.items()) + [("LR", (lr_model, 0))]:
        with torch.no_grad():
            if name == "LR":
                fn = lambda: torch.log_softmax(lr_model(char_feats(xte, vocab)), 1)
            else:
                fn = lambda model=model: torch.cat([torch.log_softmax(model(Ute[i:i + 256]), 1)
                                                    for i in range(0, N, 256)])
            logp = fn().cpu()
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            t_router = (time.perf_counter() - t0) / N * 1000
        for m in (1, 2):
            if m > K:
                continue
            r, sel = final_from_router(logp, m)
            ffts = ffts_r + m * FFT_EXPERT
            res[f"{name}-top{m}"] = dict(r, ffts=ffts, ffts_pct=100.0 * ffts / ffts_all,
                                         ms=t_router + timed_pipeline(experts, Ute, sel, reps=2),
                                         router_ms=t_router, params=base_params + n_params(model),
                                         extra_params=n_params(model))

    # 参考: 文字頻度のロジスティック回帰で全クラスを直接分類
    Ytr = torch.tensor(ytr)
    flat = train_router(nn.Linear(len(vocab), C), Ftr, Ytr, a.steps, 0.05, a.seed)
    with torch.no_grad():
        pred = flat(Fte).argmax(1)
    res["LR-flat"] = dict(acc=(pred == yte).float().mean().item(), route_top1=(owner[pred] == ete).float().mean().item(),
                          route_topm=None, ffts=0, ffts_pct=0.0, ms=None, params=n_params(flat), extra_params=None)
    return dict(K=K, n_test=N, solo=solo, fft_per_expert=FFT_EXPERT, ms_all=t_all, results=res)


out = {"seed": a.seed, "steps": a.steps, "threads": torch.get_num_threads(), "device": DEV,
       "device_name": torch.cuda.get_device_name() if DEV != "cpu" else "cpu",
       "runs": [run(int(k)) for k in a.Ks.split(",")]}
append_result("01", {"part": "route", "script": os.path.basename(__file__), "args": vars(a), "task": "freq",
                     "seed": a.seed, "device_name": device_name(DEV), **{k: v for k, v in out.items() if k not in ("seed", "device", "device_name")},
                     "sec": round(time.time() - T_START, 1)}, a.out)
lines = []
for rr in out["runs"]:
    lines.append(f"seed {a.seed}  K={rr['K']}  test {rr['n_test']}  専門家単独 {['%.3f' % s for s in rr['solo']]}  "
                 f"FFT/専門家 {rr['fft_per_expert']}")
    lines.append(f"{'方式':14s} {'正解率':>7s} {'振分top1':>8s} {'振分topm':>8s} {'FFT':>5s} {'FFT%':>6s} {'ms/入力':>8s} {'学習数(追加)':>14s}")
    for k, v in rr["results"].items():
        f3 = lambda x: "   -  " if x is None else f"{x:.3f}"
        lines.append(f"{k:14s} {v['acc']:7.3f} {f3(v['route_top1']):>8s} {f3(v['route_topm']):>8s} {v['ffts']:5d} "
                     f"{v['ffts_pct']:6.1f} {('%.3f' % v['ms']) if v['ms'] is not None else '   -':>8s} "
                     f"{v['params']:>7d}({v['extra_params'] if v['extra_params'] is not None else '-'})")
    lines.append("")
txt = "\n".join(lines)
print(txt)

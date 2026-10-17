"""実験02（旧 A1b）: 振り分け役の比較を「順序だけで決まる課題」で行う。あわせて、上位 2 人を動かしても伸びない問題（A1 の問題点 3）を
検証用データでの較正で確かめる。task=freq（A1 と同じ課題・同じ専門家）でも動く。

比べるもの（K = 4 人と 2 人）
  all-avg / all-gate / oracle / random  : A1 と同じ
  all-jcal   : 全員を動かし、専門家ごとの温度 τ_k と下駄 b_k を検証用データで合わせてから全クラスで最大（融合の較正。生の点数で）
  all-conf   : 全員を動かし、担当窓での softmax の最大値（自信）がいちばん大きい専門家の答えを採る
  R1         : 水面の板（レンズ面の明るさ I(f) × 専門家ごとの透過率の板、L=1024 区画）
  R1-c128    : 粗い水面の板（隣り合う 8 区画の明るさを束ねて 128 区画。学習する数は 1/8）
  R2-dDpP    : 薄い SinWave（板 D 枚・区画 P）
  LR-uni     : 文字頻度のロジスティック回帰（振り分け役として）
  LR-bi      : 巡回バイグラム（隣り合う 2 文字の組、最後と最初もつなぐ）の頻度のロジスティック回帰（振り分け役として。強い参考）
  *-flat     : それぞれのロジスティック回帰で全クラスを直接分類（参考）
振り分けのあとの答え方（上位 m 人だけを動かす。m = 1, 2）
  raw  : log p_k + 専門家 k の担当窓での log-softmax（A1 と同じ）
  cal  : 振り分け役の温度 α と専門家ごとの温度 τ_k を、それぞれ検証用データで別々に合わせる
  jcal : α、τ_k、b_k を検証用データでまとめて合わせる（専門家は生の点数 S_k/τ_k + b_k、全クラスでの交差エントロピー）

使い方: python experiments/a1b_route.py --task order --seed 0 --device cuda
着想: 実験01 の課題は文字の頻度だけで解けた。順序だけで決まる課題で振り分け役の本当の力を測り、上位 2 人で伸びない問題を較正で確かめる（コーディネーターの案）。
専門家: results/models/{task}/seed*/（order は exp01_experts.py --task order で学習）。
出力: results/exp02.txt に 1 行の JSON を追記
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
import torch.nn as nn
import torch.nn.functional as F

from vendor import sinwave_minimal_v1_1_0 as sw
from sn2_tasks import get_skills
from sn2_io import append_result, models_dir, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="order")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--Ks", default="4,2")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None, help="結果の 1 行 JSON の追記先（既定は results/expNN.txt）")
a = ap.parse_args()
T_START = time.time()
DEV = a.device
torch.manual_seed(a.seed)
EXP_DIR = models_dir(a.task, a.seed)
experts_all = [sw.load(os.path.join(EXP_DIR, f"expert{i}.pt"), device=DEV).eval() for i in range(4)]
skills_all = get_skills(a.task, 4, a.seed)
E0 = experts_all[0]
T, LAYERS = E0.T, E0.A.shape[0]
FFT_EXPERT = T * (2 * LAYERS + 1)


def sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# 振り分け役
# ---------------------------------------------------------------------------------------------
def lens_intensity(u):
    I = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1).abs().pow(2).mean(1)
    return I / (I.mean(-1, keepdim=True) + 1e-12)


class R1(nn.Module):
    """水面の板。bins < L なら、隣り合う L/bins 区画の明るさを束ねてから透過率の板を掛ける（粗い板）。"""

    def __init__(self, K, L, seed, bins=None):
        super().__init__()
        self.bins = bins or L
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.01 * torch.randn(K, self.bins, generator=g))
        self.gain = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(K))

    def forward(self, u):
        I = lens_intensity(u)
        I = I.reshape(len(I), self.bins, -1).mean(-1)
        return self.gain * (I @ torch.sigmoid(self.theta).T / self.bins) + self.bias


def feats(texts, vocab, kind):
    X = torch.zeros(len(texts), len(vocab))
    for b, t in enumerate(texts):
        grams = list(t) if kind == "uni" else [t[i] + t[(i + 1) % len(t)] for i in range(len(t))]
        for g in grams:
            if g in vocab:
                X[b, vocab[g]] += 1.0 / len(grams)
    return X


def make_vocab(texts, kind):
    gs = set()
    for t in texts:
        gs.update(list(t) if kind == "uni" else [t[i] + t[(i + 1) % len(t)] for i in range(len(t))])
    return {g: i for i, g in enumerate(sorted(gs))}


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
    return model.eval()


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def expert_outputs(net, U, bs=512):
    """担当窓のスコア (N, 4)、全窓のスコア (N, C)、担当窓の光の割合 r (N,)"""
    own_s, sc, rr = [], [], []
    own = torch.tensor(net.own, device=U.device)
    for i in range(0, len(U), bs):
        f = net.intensity(U[i:i + bs]) @ net.det.T
        s = net.gain * f + net.bias
        sc.append(s)
        own_s.append(s[:, own])
        rr.append(f[:, own].sum(1) / (f.sum(1) + 1e-9))
    return torch.cat(own_s).cpu(), torch.cat(sc).cpu(), torch.cat(rr).cpu()


@torch.no_grad()
def timed_pipeline(experts, U, sel, bs=256, reps=2):
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


# ---------------------------------------------------------------------------------------------
# 較正
# ---------------------------------------------------------------------------------------------
def fit_temp(logits, y, iters=300):
    """1 つの温度 τ（log でまとめて学習）。logits (N, M)、y (N,)。"""
    lt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([lt], lr=0.05)
    for _ in range(iters):
        loss = F.cross_entropy(logits / lt.exp(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lt.exp().item()


def fit_joint(Zr, S, y, K, iters=500):
    """α、τ_k、b_k をまとめて合わせる（専門家の点数は raw のまま、全クラスの交差エントロピー）。Zr: 振り分け役の点数 (N, K)（None なら振り分け役なし）、S: 担当窓のスコア (N, K, 4)。"""
    la = torch.zeros(1, requires_grad=True)
    lt = torch.zeros(K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([la, lt, b], lr=0.05)
    for _ in range(iters):
        sc = combine(Zr, S, la.exp(), lt.exp(), b, raw=True)
        loss = F.cross_entropy(sc, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return la.exp().detach(), lt.exp().detach(), b.detach()


def combine(Zr, S, alpha, tau, b, raw=False):
    """全クラスの点数 (N, C) = log_softmax(α·Zr)_k + log_softmax(S_k / τ_k) + b_k
    raw=True: log-softmax を取らず S_k / τ_k + b_k をそのまま使う（専門家どうしの点数の目盛りを直接そろえる）。
    注意: raw=False のまま τ_k, b_k を全クラスの交差エントロピーで合わせると、各専門家の確率の和が 1 に固定されるため
    専門家どうしの比べ合いが損失に入らず、較正にならない（最初の版の jcal の誤り。NOTES の A1b を参照）。"""
    lp = torch.log_softmax(alpha * Zr, 1) if Zr is not None else torch.zeros(S.shape[0], S.shape[1])
    ls = S / tau[None, :, None] if raw else torch.log_softmax(S / tau[None, :, None], 2)
    return (lp[:, :, None] + ls + b[None, :, None]).reshape(S.shape[0], -1)


# ---------------------------------------------------------------------------------------------
def run(K):
    experts = experts_all[:K]
    skills = skills_all[:K]
    C = 4 * K
    split = lambda j: ([x for sk in skills for x in sk[j][0]], [k for k, sk in enumerate(skills) for _ in sk[j][0]],
                       [y for sk in skills for y in sk[j][1]])
    xtr, etr, ytr = split(0)
    xte, ete, yte = split(1)
    xva, eva, yva = split(2)
    ete, yte, eva, yva = map(torch.tensor, (ete, yte, eva, yva))
    Etr = torch.tensor(etr)
    N = len(xte)
    Utr, Ute, Uva = E0.encode(xtr), E0.encode(xte), E0.encode(xva)

    def expert_block(U):
        outs = [expert_outputs(e, U) for e in experts]
        return (torch.stack([o[0] for o in outs], 1), torch.stack([o[1] for o in outs], 1),
                torch.stack([o[2] for o in outs], 1))

    S, SC, Rg = expert_block(Ute)
    Sva, _, _ = expert_block(Uva)
    owner = torch.arange(C) // 4
    solo = [(S[ete == k, k].argmax(1) + 4 * k == yte[ete == k]).float().mean().item() for k in range(K)]
    ones = torch.ones(K)

    # 専門家ごとの温度（自分の検証用データで）
    tau_own = torch.tensor([fit_temp(Sva[eva == k, k], yva[eva == k] - 4 * k) for k in range(K)])

    def evaluate(score_all, sel, top):
        pred = score_all.masked_fill(~sel.repeat_interleave(4, 1), -1e9).argmax(1)
        return dict(acc=(pred == yte).float().mean().item(),
                    route_top1=(top[:, 0] == ete).float().mean().item(),
                    route_topm=(top == ete[:, None]).any(1).float().mean().item())

    res = {}
    ffts_all = K * FFT_EXPERT
    allsel = torch.ones(N, K, dtype=torch.bool)
    t_all = timed_pipeline(experts, Ute, allsel)
    base_params = sum(n_params(e) for e in experts)
    common_all = dict(ffts=ffts_all, ffts_pct=100.0, ms=t_all, params=base_params, extra_params=0)

    avg = SC.mean(1)[:, :C]
    pred = avg.argmax(1)
    assert sw.predict(sw.merge(experts), xte[:64]) == pred[:64].tolist()
    res["all-avg"] = dict(acc=(pred == yte).float().mean().item(), route_top1=(owner[pred] == ete).float().mean().item(),
                          route_topm=None, **common_all)
    pg = Rg / Rg.sum(1, keepdim=True)
    pred = (torch.log(Rg + 1e-9)[:, :, None] + torch.log_softmax(S, 2)).reshape(N, C).argmax(1)
    res["all-gate"] = dict(acc=(pred == yte).float().mean().item(), route_top1=(pg.argmax(1) == ete).float().mean().item(),
                           route_topm=(pg.topk(min(2, K), 1).indices == ete[:, None]).any(1).float().mean().item(),
                           **common_all)
    _, tau_j, b_j = fit_joint(None, Sva, yva, K)
    sc = combine(None, S, torch.tensor(1.0), tau_j, b_j, raw=True)
    pred = sc.argmax(1)
    res["all-jcal"] = dict(acc=(pred == yte).float().mean().item(), route_top1=(owner[pred] == ete).float().mean().item(),
                           route_topm=None, **common_all, tau=tau_j.tolist(), b=b_j.tolist())
    # 全員を動かし、自分の担当窓での softmax の最大値（自信）がいちばん大きい専門家の答えを採る
    pred = combine(None, S, torch.tensor(1.0), ones, torch.zeros(K)).argmax(1)
    res["all-conf"] = dict(acc=(pred == yte).float().mean().item(), route_top1=(owner[pred] == ete).float().mean().item(),
                           route_topm=None, **common_all)

    for name, choice in (("oracle", ete), ("random", torch.randint(K, (N,), generator=torch.Generator().manual_seed(a.seed)))):
        sel = F.one_hot(choice, K).bool()
        sc = combine(None, S, torch.tensor(1.0), ones, torch.zeros(K))
        res[f"{name}-top1"] = dict(evaluate(sc, sel, choice[:, None]), ffts=FFT_EXPERT, ffts_pct=100.0 / K,
                                   ms=timed_pipeline(experts, Ute, sel), params=base_params, extra_params=0)

    # 振り分け役の学習
    t0 = time.time()
    routers = {}
    EtrD = Etr.to(DEV)
    routers["R1"] = (train_router(R1(K, E0.L, a.seed).to(DEV), Utr, EtrD, a.steps, 0.05, a.seed), T)
    routers["R1-c128"] = (train_router(R1(K, E0.L, a.seed, bins=128).to(DEV), Utr, EtrD, a.steps, 0.05, a.seed), T)
    for d, p in ((4, 16), (4, 64), (8, 64)):
        net = sw.SinWaveNet(n_slots=K, layers=d, patches=p, seed=a.seed * 100 + 50 + d).to(DEV)
        sw.train(net, xtr, etr, steps=a.steps, seed=a.seed)
        routers[f"R2-d{d}p{p}"] = (net.eval(), T * (2 * d + 1))
    lrs = {}
    for kind in ("uni", "bi"):
        vocab = make_vocab(xtr, kind)
        Ftr, Fte, Fva = feats(xtr, vocab, kind), feats(xte, vocab, kind), feats(xva, vocab, kind)
        lrs[kind] = (vocab, Ftr, Fte, Fva)
        routers[f"LR-{kind}"] = (train_router(nn.Linear(len(vocab), K), Ftr, Etr, a.steps, 0.05, a.seed), 0)
    print(f"  {a.task} K={K}: 振り分け役の学習 {time.time() - t0:.0f} s", flush=True)

    for name, (model, ffts_r) in routers.items():
        with torch.no_grad():
            if name.startswith("LR-"):
                kind = name[3:]
                vocab = lrs[kind][0]
                fn = lambda X: model(X)
                Zte, Zva = fn(lrs[kind][2]), fn(lrs[kind][3])
                tfn = lambda: model(feats(xte, vocab, kind))
            else:
                fn = lambda U: torch.cat([model(U[i:i + 256]) for i in range(0, len(U), 256)]).cpu()
                Zte, Zva = fn(Ute), fn(Uva)
                tfn = lambda: fn(Ute)
            sync()
            t0 = time.perf_counter()
            tfn()
            sync()
            t_router = (time.perf_counter() - t0) / N * 1000
        alpha_c = fit_temp(Zva, eva)
        al_j, tau_jj, b_jj = fit_joint(Zva, Sva, yva, K)
        variants = {"raw": (torch.tensor(1.0), ones, torch.zeros(K)),
                    "cal": (torch.tensor(1.0 / alpha_c), tau_own, torch.zeros(K)),
                    "jcal": (al_j, tau_jj, b_jj)}
        top_all = Zte.argsort(1, descending=True)
        for m in (1, 2):
            if m > K:
                continue
            top = top_all[:, :m]
            sel = torch.zeros(N, K, dtype=torch.bool)
            sel[torch.arange(N)[:, None], top] = True
            ffts = ffts_r + m * FFT_EXPERT
            ms = t_router + timed_pipeline(experts, Ute, sel)
            for vn, (al, tau, b) in variants.items():
                if m == 1 and vn != "raw":
                    # 上位 1 人では、振り分けの温度は答えを変えない。専門家の温度・下駄も 1 人の中では順位を変えない
                    continue
                sc = combine(Zte, S, al, tau, b, raw=(vn == "jcal"))
                res[f"{name}-top{m}" + ("" if vn == "raw" else f"-{vn}")] = dict(
                    evaluate(sc, sel, top), ffts=ffts, ffts_pct=100.0 * ffts / ffts_all, ms=ms, router_ms=t_router,
                    params=base_params + n_params(model), extra_params=n_params(model))

    for kind in ("uni", "bi"):
        vocab, Ftr, Fte, _ = lrs[kind]
        flat = train_router(nn.Linear(len(vocab), C), Ftr, torch.tensor(ytr), a.steps, 0.05, a.seed)
        with torch.no_grad():
            pred = flat(Fte).argmax(1)
        res[f"LR-{kind}-flat"] = dict(acc=(pred == yte).float().mean().item(),
                                      route_top1=(owner[pred] == ete).float().mean().item(), route_topm=None,
                                      ffts=0, ffts_pct=0.0, ms=None, params=n_params(flat), extra_params=None)
    return dict(K=K, n_test=N, solo=solo, tau_own=tau_own.tolist(), fft_per_expert=FFT_EXPERT, ms_all=t_all, results=res)


out = {"task": a.task, "seed": a.seed, "steps": a.steps, "device": DEV,
       "device_name": torch.cuda.get_device_name() if DEV != "cpu" else "cpu",
       "runs": [run(int(k)) for k in a.Ks.split(",")]}
append_result("02", {"script": os.path.basename(__file__), "args": vars(a), "device_name": device_name(DEV),
                     **{k: v for k, v in out.items() if k not in ("device", "device_name")}, "sec": round(time.time() - T_START, 1)}, a.out)
lines = []
for rr in out["runs"]:
    lines.append(f"{a.task} seed {a.seed}  K={rr['K']}  test {rr['n_test']}  専門家単独 {['%.3f' % s for s in rr['solo']]}  "
                 f"τ_k {['%.2f' % t for t in rr['tau_own']]}")
    for k, v in rr["results"].items():
        f3 = lambda x: "  -  " if x is None else f"{x:.3f}"
        lines.append(f"{k:22s} acc {v['acc']:.3f}  route1 {f3(v['route_top1'])}  routem {f3(v['route_topm'])}  "
                     f"FFT {v['ffts']:5d} ({v['ffts_pct']:5.1f}%)  ms {f3(v['ms'])}  params {v['params']}({v['extra_params']})")
    lines.append("")
txt = "\n".join(lines)
print(txt)

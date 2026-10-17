"""実験05（旧 A1e）: 共通の前段（A1d）を凍結したまま、後から 5 人目の専門家（新しい 4 クラス）を足す。取消も確かめる。

手順（seed ごと）
1. A1d と同じに作る: 振り分け役（板 8 枚・区画 64、T=8、窓 4）を専門家 0〜3 のデータだけで学習 → 凍結して共通の前段に。
   専門家 0〜3 の後段（板 6 枚）を、それぞれのデータの前段の光で学習。
2. 5 人目を足す（前段は凍結のまま）:
   - 5 人目の後段（板 6 枚、クラス 16〜19）を、新しいデータの前段の光だけで学習。
   - 振り分けの 5 つ目の出口: 前段の光のレンズ面の明るさ I(f) に、透過率の板 sigmoid(θ5(f)) を掛けた和（R1 と同じ形、1026 個）。
     既存 4 つの出口（窓）は凍結。5 つ目の出口だけを、5 択の交差エントロピーで学習する。
     「5 人目ではない」例が要るので、古いデータを混ぜる。replay=1.0（古い学習データを全部）と replay=0.1（各専門家 50 件）の 2 通り。
     新しいデータだけでは、選ぶ側の学習に「違う」例がないので、この形では学習できない。
   - 較正 jcal（α、τ_k、b_k）は 5 人分を検証用データで合わせ直す（足す前の 4 人分の値は取っておく）。
3. 取消: 5 つ目の出口と 5 人目の後段を外し、足す前の較正の値に戻す → 古いテストの答えが足す前と完全に同じかを確かめる。
4. 比べる相手: 元の専門家（板 16 枚）5 人を全員動かす（自信で選ぶ・較正して最大）。5 人目の元の専門家は、ここで学習する。
計算量は、元の専門家 5 人を全員動かす（5 × 264 = 1320 回）を 100% とする。

使い方: python experiments/a1e_add.py --task order --seed 0 --device cuda
着想: 実験04 で前段を共有すると、前段は「全専門家に共通の言葉」になる。前段の振り分けを学習したときに居なかった専門家を、後から足せるか（コーディネーターの案）。
出力: results/exp05.txt に 1 行の JSON を追記
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
ap.add_argument("--back", type=int, default=6)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default=None, help="結果の 1 行 JSON の追記先（既定は results/expNN.txt）")
a = ap.parse_args()
T_START = time.time()
DEV = a.device
torch.manual_seed(a.seed)
EXP_DIR = models_dir(a.task, a.seed)
K5 = 5
skills = get_skills(a.task, K5, a.seed)


def split(j, ks):
    return ([x for k in ks for x in skills[k][j][0]], torch.tensor([k for k in ks for _ in skills[k][j][0]]),
            torch.tensor([y for k in ks for y in skills[k][j][1]]))


OLD, ALL = [0, 1, 2, 3], [0, 1, 2, 3, 4]
log = []


def say(msg):
    log.append(msg)
    print(msg, flush=True)


def sync():
    if DEV != "cpu":
        torch.cuda.synchronize()


def front_field(net, u):
    for d in range(net.A.shape[0]):
        p = net.patch
        phase = net.A[d, p] * torch.sin(net.w[d, p] * net.t + net.phi[d, p])
        u = u * torch.polar(torch.ones_like(phase), phase)
        u = torch.fft.ifft(torch.fft.fft(u) * net.H)
        u = u / (u.abs().pow(2).mean(-1, keepdim=True) + 1e-24).sqrt()
    return u


def lens_I(u):
    u = torch.fft.fftshift(torch.fft.fft(u, norm="ortho"), dim=-1)
    inten = u.abs().pow(2).mean(1)
    return inten / (inten.mean(-1, keepdim=True) + 1e-12)


def train_loop(model, X, Y, steps, seed, lr, bs=256, params=None):
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(len(Y), (min(bs, len(Y)),), generator=gen)
        loss = F.cross_entropy(model(X[i.to(X.device)]), Y[i].to(X.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


def combine(Zr, S, alpha, tau, b, raw):
    K = S.shape[1]
    lp = torch.log_softmax(alpha * Zr, 1) if Zr is not None else torch.zeros(S.shape[0], K)
    ls = S / tau[None, :, None] if raw else torch.log_softmax(S / tau[None, :, None], 2)
    return (lp[:, :, None] + ls + b[None, :, None]).reshape(S.shape[0], -1)


def fit_joint(Zr, S, y, iters=500):
    K = S.shape[1]
    la, lt, b = torch.zeros(1, requires_grad=True), torch.zeros(K, requires_grad=True), torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([la, lt, b], lr=0.05)
    for _ in range(iters):
        loss = F.cross_entropy(combine(Zr, S, la.exp(), lt.exp(), b, raw=True), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return la.exp().detach(), lt.exp().detach(), b.detach()


# ---- 1. A1d と同じ 4 人の系 -------------------------------------------------------------------
t0 = time.time()
xtr_o, etr_o, _ = split(0, OLD)
router = sw.SinWaveNet(n_slots=4, layers=8, patches=64, seed=a.seed * 100 + 70).to(DEV)
sw.train(router, xtr_o, etr_o.tolist(), steps=a.steps, seed=a.seed)
router.eval()
for p in router.parameters():
    p.requires_grad_(False)
T = router.T


@torch.no_grad()
def fields(xs):
    return torch.cat([front_field(router, router.encode(xs[i:i + 512])) for i in range(0, len(xs), 512)])


@torch.no_grad()
def old_exits(Fx):
    return torch.cat([router.gain * (lens_I(Fx[i:i + 512]) @ router.det.T) + router.bias for i in range(0, len(Fx), 512)]).cpu()


data = {j: split(j, ALL) for j in range(3)}            # 0 学習 / 1 テスト / 2 検証、5 人分
Fd = {j: fields(data[j][0]) for j in range(3)}
say(f"前段の学習 {time.time() - t0:.0f} s")

backs = []
for k in ALL:
    t0 = time.time()
    idx = (data[0][1] == k).nonzero()[:, 0]
    net = sw.SinWaveNet(n_slots=20, own=list(range(4 * k, 4 * k + 4)), layers=a.back, patches=64,
                        seed=a.seed * 100 + 80 + k).to(DEV)
    train_loop(net, Fd[0][idx.to(DEV)], data[0][2][idx], a.steps, a.seed * 100 + k, 0.01)
    backs.append(net.eval())
    say(f"後段 {k} の学習 {time.time() - t0:.0f} s")


@torch.no_grad()
def back_scores(Fx, ks):
    out = []
    for k in ks:
        own = torch.tensor(backs[k].own, device=DEV)
        out.append(torch.cat([backs[k](Fx[i:i + 512])[:, own] for i in range(0, len(Fx), 512)]).cpu())
    return torch.stack(out, 1)


# ---- 2. 5 つ目の出口 ----------------------------------------------------------------------------
class NewExit(nn.Module):
    def __init__(self, L, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.01 * torch.randn(L, generator=g))
        self.gain = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, I):
        return self.gain * (I @ torch.sigmoid(self.theta)) / I.shape[-1] + self.bias


@torch.no_grad()
def Ifeat(Fx):
    return torch.cat([lens_I(Fx[i:i + 512]) for i in range(0, len(Fx), 512)])


Itr, Ite, Iva = Ifeat(Fd[0]), Ifeat(Fd[1]), Ifeat(Fd[2])
Z4 = {j: old_exits(Fd[j]) for j in range(3)}
S5 = {j: back_scores(Fd[j], ALL) for j in (1, 2)}
e_te, y_te = data[1][1], data[1][2]
e_va, y_va = data[2][1], data[2][2]
old_te, new_te = e_te < 4, e_te == 4
old_va = e_va < 4
N = len(e_te)
FFT_EXPERT = 8 * 33
FFT_FULL5 = K5 * FFT_EXPERT
fft_front, fft_lens, fft_back = 2 * 8 * T, T, T * (2 * a.back + 1)


def run_system(Zr, S, cal, m, mask_rows):
    """Zr (N, K)、S (N, K, 4)。上位 m 人を動かし、cal = (α, τ, b) で答える。戻り値: 予測 (N,)、上位の専門家 (N, m)"""
    K = S.shape[1]
    top = Zr.argsort(1, descending=True)[:, :m]
    sel = torch.zeros(len(Zr), K, dtype=torch.bool)
    sel[torch.arange(len(Zr))[:, None], top] = True
    if m == 1:
        sc = combine(Zr, S, torch.tensor(1.0), torch.ones(K), torch.zeros(K), False)
    else:
        sc = combine(Zr, S, *cal, raw=True)
    return sc.masked_fill(~sel.repeat_interleave(4, 1), -1e9).argmax(1), top


def metrics(pred, top, rows, e, y):
    return dict(acc=(pred[rows] == y[rows]).float().mean().item(),
                route_top1=(top[rows, 0] == e[rows]).float().mean().item(),
                route_topm=(top[rows] == e[rows][:, None]).any(1).float().mean().item())


res = {}
# 足す前（4 人、古いテストだけ）
cal4 = fit_joint(Z4[2][old_va], S5[2][old_va][:, :4], y_va[old_va])
before = {}
for m in (1, 2):
    pred, top = run_system(Z4[1][old_te], S5[1][old_te][:, :4], cal4, m, None)
    before[m] = pred
    res[f"before4-top{m}"] = dict(metrics(pred, top, torch.ones(len(pred), dtype=torch.bool), e_te[old_te], y_te[old_te]),
                                  ffts=fft_front + fft_lens + m * fft_back)

# 5 人目を足す
for replay in (1.0, 0.1):
    t0 = time.time()
    xtr, etr = data[0][0], data[0][1]
    keep = torch.zeros(len(etr), dtype=torch.bool)
    keep[etr == 4] = True
    g = torch.Generator().manual_seed(a.seed)
    for k in OLD:
        idx = (etr == k).nonzero()[:, 0]
        n = max(1, int(round(len(idx) * replay)))
        keep[idx[torch.randperm(len(idx), generator=g)[:n]]] = True
    ex = NewExit(router.L, a.seed).to(DEV)
    Z4tr = Z4[0].to(DEV)
    model = lambda i: torch.cat([Z4tr[i], ex(Itr[i])[:, None]], 1)
    sub = keep.nonzero()[:, 0].to(DEV)
    Ysub = etr[keep].to(DEV)

    class _M(nn.Module):
        def forward(self, j):
            return model(sub[j])

    train_loop(_M(), torch.arange(len(sub), device=DEV), Ysub.cpu(), a.steps, a.seed, 0.05, params=list(ex.parameters()))
    with torch.no_grad():
        Z5 = {j: torch.cat([Z4[j], ex(I).cpu()[:, None]], 1) for j, I in ((1, Ite), (2, Iva))}
    cal5 = fit_joint(Z5[2], S5[2], y_va)
    tag = f"replay{int(replay * 100)}"
    say(f"5 つ目の出口（{tag}、学習に使った古いデータ {int(keep.sum()) - int((etr == 4).sum())} 件）の学習 {time.time() - t0:.0f} s")
    for m in (1, 2):
        pred, top = run_system(Z5[1], S5[1], cal5, m, None)
        allr = torch.ones(N, dtype=torch.bool)
        res[f"after5-{tag}-top{m}"] = dict(all=metrics(pred, top, allr, e_te, y_te), old=metrics(pred, top, old_te, e_te, y_te),
                                          new=metrics(pred, top, new_te, e_te, y_te),
                                          ffts=fft_front + fft_lens + m * fft_back)
    # 取消: 5 つ目の出口と 5 人目の後段を外し、足す前の較正に戻す
    for m in (1, 2):
        pred, _ = run_system(Z5[1][old_te][:, :4], S5[1][old_te][:, :4], cal4, m, None)
        res[f"after5-{tag}-top{m}"]["undo_identical"] = bool(torch.equal(pred, before[m]))

# 5 人目の後段だけの正解率
S_new = S5[1][new_te][:, 4]
res["new-back-solo"] = dict(acc=(S_new.argmax(1) + 16 == y_te[new_te]).float().mean().item())

# ---- 比べる相手: 元の専門家（板 16 枚）5 人を全員動かす ----------------------------------------------
t0 = time.time()
orig = [sw.load(os.path.join(EXP_DIR, f"expert{i}.pt"), device=DEV).eval() for i in OLD]
xtr4, _, ytr4 = split(0, [4])
e5 = sw.SinWaveNet(n_slots=20, own=[16, 17, 18, 19], seed=a.seed * 100 + 4).to(DEV)
sw.train(e5, xtr4, ytr4.tolist(), steps=a.steps, seed=a.seed * 100 + 4)
orig.append(e5.eval())
say(f"元の形の 5 人目の専門家（板 16 枚）の学習 {time.time() - t0:.0f} s")


@torch.no_grad()
def orig_scores(xs):
    U = orig[0].encode(xs)
    out = []
    for net in orig:
        own = torch.tensor(net.own, device=DEV)
        out.append(torch.cat([net(U[i:i + 512])[:, own] for i in range(0, len(U), 512)]).cpu())
    return torch.stack(out, 1)


So_te, So_va = orig_scores(data[1][0]), orig_scores(data[2][0])
ones, zeros = torch.ones(K5), torch.zeros(K5)
pred = combine(None, So_te, torch.tensor(1.0), ones, zeros, False).argmax(1)
res["orig5-all-conf"] = dict(acc=(pred == y_te).float().mean().item(), old=(pred[old_te] == y_te[old_te]).float().mean().item(),
                             new=(pred[new_te] == y_te[new_te]).float().mean().item(), ffts=FFT_FULL5)
_, tj, bj = fit_joint(None, So_va, y_va)
pred = combine(None, So_te, torch.tensor(1.0), tj, bj, True).argmax(1)
res["orig5-all-jcal"] = dict(acc=(pred == y_te).float().mean().item(), old=(pred[old_te] == y_te[old_te]).float().mean().item(),
                             new=(pred[new_te] == y_te[new_te]).float().mean().item(), ffts=FFT_FULL5)
for v in res.values():
    if "ffts" in v:
        v["ffts_pct"] = 100.0 * v["ffts"] / FFT_FULL5
        v["ffts_pct_of4"] = 100.0 * v["ffts"] / (4 * FFT_EXPERT)

out = {"task": a.task, "seed": a.seed, "steps": a.steps, "back": a.back, "log": log,
       "params_new": {"back5": sum(p.numel() for p in backs[4].parameters()), "exit5": router.L + 2}, "results": res}
append_result("05", {"script": os.path.basename(__file__), "args": vars(a), "device_name": device_name(DEV), **out,
                     "sec": round(time.time() - T_START, 1)}, a.out)
txt = json.dumps(res, ensure_ascii=False, indent=1)
print(txt)

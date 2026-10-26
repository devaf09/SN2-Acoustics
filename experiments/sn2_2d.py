"""2 次元のグレイン（vendor/sn2_parts_2d_v0_11_0.py）で振り分けを試すための共通部品（実験06〜）。

- 課題: synth4（合成 4 課題を専門家 4 人に、全 33 クラス）、fashion_mix / fashion_sim（Fashion-MNIST を専門家 5 人 × 2 クラスに）
- 専門家・振り分け役・前段の作り方、FFT 回数の数え方、較正（生の点数）、実測時間

Fashion-MNIST は SN2-Parts-2D が取得したもの（Zalando Research、MIT）を、写さずにその場所から読む（持ち主の許可、2026-10-25）。
"""
from __future__ import annotations

import math
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

from vendor import sn2_parts_2d_v0_11_0 as p2
from vendor import sn2_tasks2d as t2

FASHION_ROOT = os.environ.get("SN2_FASHION_ROOT", r"D:\AI_git\SN2-Parts-2D\data")
FASHION_NAMES = ["T シャツ", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "シャツ", "スニーカー", "バッグ", "ブーツ"]
# 専門家 5 人 × 2 クラスの分け方
#   sim: 似た服どうしを同じ専門家に（専門家の中が難しく、振り分けは易しい）
#   mix: 似た服どうしを別の専門家に（振り分けが難しい）
FASHION_GROUPS = {"sim": [[0, 6], [2, 4], [5, 7], [1, 3], [8, 9]],
                  "mix": [[0, 1], [2, 5], [4, 8], [6, 9], [3, 7]]}
SYNTH_TASKS = ["orient", "position", "conj", "shape"]


def load_task(task: str, seed: int, L: int = 64, n_train: int = 4000, n_test: int = 2000, n_val: int = 1000):
    """戻り値: dict(Xtr, ytr, etr, Xte, yte, ete, Xva, yva, eva, C, owners)。
    y は全体のクラス番号、e は担当の専門家の番号、owners[k] は専門家 k のクラスの並び。"""
    if task == "synth4":
        parts = {s: [] for s in ("tr", "te", "va")}
        owners, off = [], 0
        for k, tk in enumerate(SYNTH_TASKS):
            n_c = t2.TASKS[tk]
            for s, n, sd in (("tr", n_train, seed * 100 + k), ("te", n_test, seed * 100 + 50 + k), ("va", n_val, seed * 100 + 80 + k)):
                X, y = t2.make(tk, n, L=L, seed=sd)
                parts[s].append((X, y + off, torch.full_like(y, k)))
            owners.append(list(range(off, off + n_c)))
            off += n_c
        cat = lambda s, i: torch.cat([p[i] for p in parts[s]])
        return dict(Xtr=cat("tr", 0), ytr=cat("tr", 1), etr=cat("tr", 2), Xte=cat("te", 0), yte=cat("te", 1), ete=cat("te", 2),
                    Xva=cat("va", 0), yva=cat("va", 1), eva=cat("va", 2), C=off, owners=owners)
    if task.startswith("fashion_"):
        from torchvision import datasets
        groups = FASHION_GROUPS[task.split("_", 1)[1]]
        # クラス番号を、専門家 k の担当が 2k, 2k+1 になるように並べ替える
        order = [c for g in groups for c in g]
        remap = torch.empty(10, dtype=torch.long)
        remap[torch.tensor(order)] = torch.arange(10)
        out = []
        for train in (True, False):
            d = datasets.FashionMNIST(FASHION_ROOT, train=train, download=False)
            X = d.data.float() / 255.0
            p = (L - X.shape[-1]) // 2
            X = F.pad(X, (p, L - X.shape[-1] - p, p, L - X.shape[-1] - p))
            out.append((X, remap[d.targets]))
        (Xa, ya), (Xte, yte) = out
        g = torch.Generator().manual_seed(seed)                # 学習データから検証用を 1 クラス 500 枚ずつ分ける
        perm = torch.randperm(len(ya), generator=g)
        va_idx = torch.cat([perm[ya[perm] == c][:500] for c in range(10)])
        mask = torch.ones(len(ya), dtype=torch.bool)
        mask[va_idx] = False
        Xtr, ytr, Xva, yva = Xa[mask], ya[mask], Xa[va_idx], ya[va_idx]
        return dict(Xtr=Xtr, ytr=ytr, etr=ytr // 2, Xte=Xte, yte=yte, ete=yte // 2, Xva=Xva, yva=yva, eva=yva // 2,
                    C=10, owners=[[2 * k, 2 * k + 1] for k in range(5)],
                    class_names=[FASHION_NAMES[c] for c in order])
    raise ValueError(task)


def field(X):
    return p2.image_to_field(X)


def n_ffts(n_plates: int, lens: int = 1) -> int:
    """2 次元 FFT の回数（板 1 枚 = 回折の fft2 + ifft2 で 2 回、レンズ 1 回）。"""
    return 2 * n_plates + lens


def front_field(net, u):
    """net の板を全部通した光（レンズの前）。Net2D.screen の板の部分と同じ計算（照明なし）。
    net が NLNet2D なら、指定の板の後で光の非線形（明るさの平方根で光を出し直す）をはさむ。"""
    nl = getattr(net, "nl_after", ())
    for d in range(len(net.kinds)):
        u = p2.normalize(p2.propagate(p2.apply_plate(u, net.plate_phase(d)), net.H))
        if d in nl:
            u = p2.normalize(u.abs().to(torch.complex64))       # √(明るさ) = |u|、位相は捨てる（実験09）
    return u


class NLNet2D(p2.Net2D):
    """Net2D の板の途中に光の非線形をはさんだもの（実験09）。nl_after = 非線形を入れる板の番号の並び（0 始まり）。
    非線形は「明るさを測って、その平方根の振幅で光を出し直す」（SN2-Parts-2D の実験28・75〜77 で効いた sqrt と同じ）。"""

    def __init__(self, *args, nl_after=(), **kw):
        super().__init__(*args, **kw)
        self.nl_after = tuple(nl_after)

    def screen(self, u):
        I = p2.intensity(p2.lens(front_field(self, u)))
        return I / (I.mean((-2, -1), keepdim=True) + 1e-12)


def lens_scores(net, u):
    """光（レンズの前）→ レンズ → 窓の点数。Net2D.screen の最後 + forward と同じ計算。"""
    I = p2.intensity(p2.lens(u))
    I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
    return net.gain * p2.read_windows(I, net.det) + net.bias


class R1(nn.Module):
    """水面の板（1 次元の実験01 の R1 の 2 次元版）: 入力の光をレンズに通した明るさ I に、専門家ごとの透過率 sigmoid(θ_k) を掛けた和。"""

    def __init__(self, K, L, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.01 * torch.randn(K, L, L, generator=g))
        self.gain = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.zeros(K))

    def forward(self, u):
        I = p2.intensity(p2.lens(u))
        I = I / (I.mean((-2, -1), keepdim=True) + 1e-12)
        return self.gain * torch.einsum("nyx,kyx->nk", I, torch.sigmoid(self.theta)) / I[0].numel() + self.bias


@torch.no_grad()
def scores(net, X, dev, own=None, bs=500, prep=field):
    """net の点数 (N, C)（own を与えればその列だけ）。X は CPU、まとめて dev へ送る。"""
    out = []
    for i in range(0, len(X), bs):
        s = net(prep(X[i:i + bs].to(dev)))
        out.append((s[:, own] if own is not None else s).cpu())
    return torch.cat(out)


def combine(Zr, S, alpha, tau, b, raw):
    """全クラスの点数 (N, Σ担当) = log_softmax(α·Zr)_k + （raw なら S_k/τ_k + b_k、でなければ log_softmax(S_k/τ_k)）。
    S は専門家ごとの担当の窓の点数のリスト（担当の数が専門家ごとに違ってよい）。"""
    K = len(S)
    N = S[0].shape[0]
    lp = torch.log_softmax(alpha * Zr, 1) if Zr is not None else torch.zeros(N, K)
    cols = []
    for k in range(K):
        s = S[k] / tau[k]
        cols.append(lp[:, k:k + 1] + (s + b[k] if raw else torch.log_softmax(s, 1)))
    return torch.cat(cols, 1)


def fit_joint(Zr, S, y, iters=500):
    """α、τ_k、b_k を検証用データで合わせる（生の点数、全クラスの交差エントロピー）。"""
    K = len(S)
    la, lt, b = torch.zeros(1, requires_grad=True), torch.zeros(K, requires_grad=True), torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([la, lt, b], lr=0.05)
    for _ in range(iters):
        loss = F.cross_entropy(combine(Zr, S, la.exp(), lt.exp(), b, raw=True), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return la.exp().detach(), lt.exp().detach(), b.detach()


def select(score_all, owners_cols, sel):
    """sel (N, K) で選んだ専門家の列だけを残して最大を取る。owners_cols[k] は combine の出力での専門家 k の列。"""
    mask = torch.zeros_like(score_all, dtype=torch.bool)
    for k, cols in enumerate(owners_cols):
        mask[:, cols] = sel[:, k:k + 1]
    return score_all.masked_fill(~mask, -1e9).argmax(1)


def sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


@torch.no_grad()
def timed(fn, dev, reps=3):
    """fn() の壁時計時間の最小（秒）。"""
    best = float("inf")
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def train_cls(net, X, y, steps, seed, dev, reject=None, reject_weight=0.3, prep=field, lr=0.01, bs=256):
    """p2.train と同じ学習（Adam、cosine）。X は CPU のまま、バッチごとに dev へ送る（GPU のメモリを節約）。"""
    return p2.train(net, X, y, steps=steps, lr=lr, batch_size=bs, seed=seed, to_input=prep, reject=reject,
                    reject_weight=reject_weight, schedule="cosine")

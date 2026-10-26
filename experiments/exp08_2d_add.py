"""実験08: 2 次元のグレインで、共通の前段を凍結したまま専門家を後から 1 人足す（1 次元の実験05 の 2D 版と、その次の一手）。

着想: 実験06 で、Fashion-MNIST では「専門家に分ける」より「1 本で全クラス」の方が正解率が高かった（84.6% 対 全員を動かす最良 69.8%）。
      専門家に分ける利点は正解率ではなく、後から足す・外すことにあるはず（1 本の光路は、新しいクラスを足すと全データで学び直しになる）。
      あわせて、1 次元の実験05 で弱かった「新しい出口」を 2 つの工夫で直せるか: 下駄（起こりやすさ）を検証用データで合わせる、薄い板の出口。
方法: 最後の専門家を「後から来る専門家」とし、残りのデータだけで振り分け役（板 ① f 枚）を学習 → 凍結して前段に。
      古い専門家の後段（板 b 枚）を学習して「足す前」の系を作る。
      足すときは、前段・古い出口・古い後段は凍結のまま、新しい後段（新しいデータだけ）と新しい出口だけを学習する。
      新しい出口は 2 種類:
        plate: 前段の光のレンズの後の明るさ × 透過率の板（1 次元の実験05 と同じ形。追加の FFT なし）
        thin : 前段の光 → 薄い板 2 枚 → レンズ → 窓 1 つ（追加の FFT 5 回）
      新しい出口の学習には「新しい専門家ではない」例として古いデータを混ぜる（全部 / 10%）。さらに、学習の後で
      新しい出口の下駄を検証用データで合わせ直す版（+bias）も測る。
比べるもの: 元の専門家（板 16 枚、実験06 で保存）を全員動かして較正（足した後の人数で）/ 1 本で全クラス（実験06、全データで学習し直す場合）/
            足す前の系。取消（新しい出口と後段を外すと、古いテストの答えが 1 件残らず足す前と同じか）。
計算量: 元の専門家を全員動かす（K × 33 回）= 100%。

使い方: python experiments/exp08_2d_add.py --task synth4 --seed 0 --device cuda
出力: results/exp08.txt に 1 行の JSON を追記
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F

import sn2_2d as s2
from sn2_2d import p2, t2
from sn2_io import append_result, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="synth4", choices=["synth4", "fashion_mix", "fashion_sim"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=0, help="0 なら synth4 は 1500、fashion は 3000")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--front", type=int, default=8)
ap.add_argument("--back", type=int, default=6)
ap.add_argument("--reject", type=float, default=0.3)
ap.add_argument("--n_train", type=int, default=4000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
a = ap.parse_args()
T_START = time.time()
DEV = a.device
steps = a.steps or (1500 if a.task == "synth4" else 3000)
torch.manual_seed(a.seed)
D = s2.load_task(a.task, a.seed, L=a.L, n_train=a.n_train)
K, C, owners, L = len(D["owners"]), D["C"], D["owners"], a.L
NEW, OLD = K - 1, list(range(K - 1))
G = t2.make_generic(4000, L=L, seed=a.seed + 999) if a.reject > 0 else None
log = []


def say(m):
    log.append(m)
    print(m, flush=True)


cols, off = [], 0
for o in owners:
    cols.append(list(range(off, off + len(o))))
    off += len(o)
col2class = torch.tensor([c for o in owners for c in o])
class2col = torch.empty(C, dtype=torch.long)
class2col[col2class] = torch.arange(len(col2class))
y_te, e_te, y_va, e_va = D["yte"], D["ete"], D["yva"], D["eva"]
old_te, new_te, old_va = e_te < NEW, e_te == NEW, e_va < NEW
y_te_col, y_va_col = class2col[y_te], class2col[y_va]
n_old_cols = sum(len(c) for c in cols[:NEW])

# ---- 1. 足す前の系（古い専門家だけ）------------------------------------------------------------
t0 = time.time()
old_tr = D["etr"] < NEW
router = p2.Net2D(L, NEW, plates=["xy"] * a.front, patches=a.patches, seed=a.seed * 100 + 70 + a.front).to(DEV)
s2.train_cls(router, D["Xtr"][old_tr], D["etr"][old_tr], steps, a.seed, DEV)
router.eval()
for p in router.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def ff(X, bs=500):
    return torch.cat([s2.front_field(router, s2.field(X[i:i + bs].to(DEV))).cpu() for i in range(0, len(X), bs)])


Ftr, Fte, Fva = ff(D["Xtr"]), ff(D["Xte"]), ff(D["Xva"])
FG = ff(G) if G is not None else None


@torch.no_grad()
def lens_I(Fx, bs=500):
    out = []
    for i in range(0, len(Fx), bs):
        I = p2.intensity(p2.lens(Fx[i:i + bs].to(DEV)))
        out.append((I / (I.mean((-2, -1), keepdim=True) + 1e-12)).cpu())
    return torch.cat(out)


@torch.no_grad()
def old_exits(Fx, bs=500):
    return torch.cat([s2.lens_scores(router, Fx[i:i + bs].to(DEV)).cpu() for i in range(0, len(Fx), bs)])


Z = {"tr": old_exits(Ftr), "te": old_exits(Fte), "va": old_exits(Fva)}
say(f"前段（古い {NEW} 人の振り分け）{time.time() - t0:.0f} 秒")
backs = []
for k in range(K):
    t0 = time.time()
    net = p2.Net2D(L, C, plates=["xy"] * a.back, patches=a.patches, seed=a.seed * 100 + 80 + k).to(DEV)
    m = D["etr"] == k
    p2.train(net, Ftr[m], D["ytr"][m], steps=steps, seed=a.seed * 100 + k, reject=FG, reject_weight=a.reject, schedule="cosine")
    net.eval()
    backs.append(net)
    say(f"後段 {k}{'（新しい）' if k == NEW else ''} {time.time() - t0:.0f} 秒")
S_te = [s2.scores(n_, Fte, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]
S_va = [s2.scores(n_, Fva, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]
new_solo = (torch.tensor(owners[NEW])[S_te[NEW][new_te].argmax(1)] == y_te[new_te]).float().mean().item()

FFT_E = s2.n_ffts(16)
FFT_FULL = K * FFT_E
fft_front, fft_back = 2 * a.front, s2.n_ffts(a.back)


def run(Zr, S, cal, m, rows=None):
    Kx = len(S)
    top = Zr.argsort(1, descending=True)[:, :m]
    sel = torch.zeros(len(Zr), Kx, dtype=torch.bool)
    sel[torch.arange(len(Zr))[:, None], top] = True
    sc = s2.combine(Zr, S, *cal, True) if m == 2 else s2.combine(Zr, S, torch.tensor(1.0), torch.ones(Kx), torch.zeros(Kx), False)
    return s2.select(sc, cols[:Kx], sel), top


def metr(pred_col, top, rows, e, ycol=None):
    ycol = y_te_col if ycol is None else ycol
    return dict(acc=(pred_col[rows] == ycol[rows]).float().mean().item(),
                route_top1=(top[rows, 0] == e[rows]).float().mean().item(),
                route_topm=(top[rows] == e[rows][:, None]).any(1).float().mean().item())


res = {}
cal_old = s2.fit_joint(Z["va"][old_va], [s[old_va] for s in S_va[:NEW]], y_va_col[old_va])
before = {}
for m in (1, 2):
    pc, top = run(Z["te"][old_te], [s[old_te] for s in S_te[:NEW]], cal_old, m)
    before[m] = pc
    res[f"before-top{m}"] = dict(old=metr(pc, top, torch.ones(len(pc), dtype=torch.bool), e_te[old_te], y_te_col[old_te]),
                                 ffts=fft_front + 1 + m * fft_back)


# ---- 2. 新しい出口 ------------------------------------------------------------------------------
class PlateExit(nn.Module):
    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(0.01 * torch.randn(L, L, generator=g))
        self.gain = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, I):
        return self.gain * (I * torch.sigmoid(self.theta)).mean((-2, -1)) + self.bias


class ThinExit(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.net = p2.Net2D(L, 1, plates=["xy"] * 2, patches=a.patches, seed=seed)

    def forward(self, Fx):
        return self.net(Fx)[:, 0]


I_all = {"tr": lens_I(Ftr), "te": lens_I(Fte), "va": lens_I(Fva)}
Fd = {"tr": Ftr, "te": Fte, "va": Fva}
for kind in ("plate", "thin"):
    for replay in (1.0, 0.1):
        t0 = time.time()
        etr = D["etr"]
        keep = etr == NEW
        g = torch.Generator().manual_seed(a.seed)
        for k in OLD:
            idx = (etr == k).nonzero()[:, 0]
            keep[idx[torch.randperm(len(idx), generator=g)[:max(1, int(round(len(idx) * replay)))]]] = True
        sub = keep.nonzero()[:, 0]
        ex = (PlateExit(a.seed) if kind == "plate" else ThinExit(a.seed * 100 + 60)).to(DEV)
        X_in = I_all["tr"] if kind == "plate" else Fd["tr"]
        opt = torch.optim.Adam(ex.parameters(), lr=0.05 if kind == "plate" else 0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        gen = torch.Generator().manual_seed(a.seed)
        for _ in range(steps):
            j = sub[torch.randint(len(sub), (256,), generator=gen)]
            logits = torch.cat([Z["tr"][j].to(DEV), ex(X_in[j].to(DEV))[:, None]], 1)
            loss = F.cross_entropy(logits, etr[j].to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        ex.eval()
        with torch.no_grad():
            src = I_all if kind == "plate" else Fd
            znew = {s: torch.cat([ex(src[s][i:i + 500].to(DEV)).cpu() for i in range(0, len(src[s]), 500)]) for s in ("te", "va")}
        # 新しい出口の下駄を検証用データで合わせる（振り分けの 5 択の交差エントロピー、ほかは固定）
        bshift = torch.zeros(1, requires_grad=True)
        opt2 = torch.optim.Adam([bshift], lr=0.05)
        for _ in range(300):
            lg = torch.cat([Z["va"], (znew["va"] + bshift)[:, None]], 1)
            l2 = F.cross_entropy(lg, e_va)
            opt2.zero_grad()
            l2.backward()
            opt2.step()
        tag = f"{kind}-replay{int(replay * 100)}"
        fft_exit = 0 if kind == "plate" else s2.n_ffts(2)
        for bias_fix in (False, True):
            add = bshift.detach() if bias_fix else torch.zeros(1)
            Z5 = {s: torch.cat([Z[s], (znew[s] + add)[:, None]], 1) for s in ("te", "va")}
            cal5 = s2.fit_joint(Z5["va"], S_va, y_va_col)
            for m in (1, 2):
                pc, top = run(Z5["te"], S_te, cal5, m)
                allr = torch.ones(len(pc), dtype=torch.bool)
                res[f"after-{tag}{'+bias' if bias_fix else ''}-top{m}"] = dict(
                    all=metr(pc, top, allr, e_te), old=metr(pc, top, old_te, e_te), new=metr(pc, top, new_te, e_te),
                    ffts=fft_front + 1 + fft_exit + m * fft_back, bias_shift=float(add))
        # 取消: 新しい出口と後段を外し、足す前の較正に戻す
        for m in (1, 2):
            pc, _ = run(Z["te"][old_te], [s[old_te] for s in S_te[:NEW]], cal_old, m)
            for key in list(res):
                if key.startswith(f"after-{tag}") and key.endswith(f"top{m}"):
                    res[key]["undo_identical"] = bool(torch.equal(pc, before[m]))
        say(f"新しい出口 {tag}: {time.time() - t0:.0f} 秒、下駄の補正 {float(bshift):+.2f}")

# ---- 3. 比べる相手: 元の専門家（板 16 枚、実験06 で保存）を全員動かす ----------------------------------------
mdir = os.path.join(s2.ROOT, "results", "models2d", f"{a.task}_r{a.reject:g}", f"seed{a.seed}")
orig = []
for k in range(K):
    d = torch.load(os.path.join(mdir, f"expert{k}.pt"), map_location="cpu", weights_only=False)
    net = p2.Net2D(**d["config"]).to(DEV)
    net.load_state_dict(d["state"])
    orig.append(net.eval())
So_te = [s2.scores(n_, D["Xte"], DEV, owners[k]) for k, n_ in enumerate(orig)]
So_va = [s2.scores(n_, D["Xva"], DEV, owners[k]) for k, n_ in enumerate(orig)]
_, tj, bj = s2.fit_joint(None, So_va, y_va_col)
pc = s2.combine(None, So_te, torch.tensor(1.0), tj, bj, True).argmax(1)
res["orig-all-jcal"] = dict(all=dict(acc=(pc == y_te_col).float().mean().item()), old=dict(acc=(pc[old_te] == y_te_col[old_te]).float().mean().item()),
                            new=dict(acc=(pc[new_te] == y_te_col[new_te]).float().mean().item()), ffts=FFT_FULL)
for v in res.values():
    v["ffts_pct"] = 100.0 * v["ffts"] / FFT_FULL

append_result("08", {"script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed, "steps": steps,
                     "K": K, "C": C, "new_expert": NEW, "new_solo": new_solo, "fft_full": FFT_FULL,
                     "params_new": {"back": backs[NEW].n_params(), "exit_plate": L * L + 2, "exit_thin": ThinExit(0).net.n_params()},
                     "device_name": device_name(DEV), "log": log, "results": res, "sec": round(time.time() - T_START, 1)}, a.out)
for k, v in res.items():
    if "all" in v:
        print(f"{k:34s} all {v['all']['acc']:.3f} old {v['old']['acc']:.3f} new {v['new']['acc']:.3f} FFT% {v['ffts_pct']:.1f} undo {v.get('undo_identical')}")

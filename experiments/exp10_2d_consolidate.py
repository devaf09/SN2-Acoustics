"""実験10: 2 次元のグレインで「記憶の固定化」。後から付け足した専門家（短期）を、受付係と後段の学び直しでまとめる（長期）。

着想: 実験08 で、共通の前段を凍結したまま、新しい後段 + 薄い板の窓口（thin）を付け足せた（取消は完全）。
      引き換えに、古いクラスが 1.5〜2.8 点下がり、新しい人の受付は 85〜93% どまり、窓口の分だけ計算が増える（つぎはぎの跡）。
      付け足した系をお手本（先生）にして、前段（受付係）と後段を少しだけ学び直し、つぎはぎの跡を消せるか（到達点 B、計画は
      docs/consolidation_plan.md）。Sin-CNN3 の知見（板を混ぜると壊れる・古い物を混ぜないと忘れる）から、
      どれも今の板から出発して短く学び直し、古いデータ 1 割かお手本の答えを必ず混ぜる。
方法: 1. 実験08 と同じ手順で「足す前」の系（前段 8 枚 = 古い K-1 人の受付、後段 6 枚 × K-1）と、
         「付け足した」系（新しい後段 + thin 窓口、古いデータ 1 割 + 下駄の合わせ直し）を作る。これがお手本。
      2. 固定化: 学生 = 前段のコピー（窓を K 人分に作り直す）+ 後段のコピー。新しいデータ全部 + 古いデータ 1 割（お手本と同じ）+
         担当なしの絵で学び直す。お手本の答え（受付の点数 K 個、各後段の担当の生の点数）は最初に 1 回だけ計算する。
         K1 前段だけ   : 前段だけを学習（後段は固定）。損失 = 受付の交差エントロピー + 受付のお手本合わせ（KL）+ 後段の生の点数のお手本合わせ（二乗誤差）
         K2 前段 + 後段: 同じ損失で前段と後段を一緒に学習（本命）
         K3 お手本なし : K2 と同じ形・同じデータで、正解のラベルだけで学習（受付の交差エントロピー + 後段は自分の担当の交差エントロピーと担当なしの絵）
      3. 評価: 全クラス・古いクラス・新しいクラスの正解率（上位 1 人、上位 2 人 + 較正）、受付の正解率、計算量（FFT 回数）、
         取消（固定化した系から新しい人を外したとき、古いテストの答えが足す前と一致する割合）。
比べるもの: 足す前 / 付け足しただけ（お手本）/ 最初から K 人そろって学習（実験07 の前段 8 + 後段 6、別の実験の数字）/ 1 本で全データ（実験06）。
計算量: 元の専門家を全員動かす（K × 33 回）= 100%（実験06〜08 と同じ物差し）。

使い方: python experiments/exp10_2d_consolidate.py --task fashion_sim --seed 0 --device cuda
出力: results/exp10.txt に 1 行の JSON を追記
"""
from __future__ import annotations

import argparse
import copy
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
ap.add_argument("--steps", type=int, default=0, help="0 なら synth4 は 1500、fashion は 3000（前段・後段・窓口の学習）")
ap.add_argument("--csteps", type=int, default=0, help="固定化の学習のステップ数。0 なら --steps と同じ")
ap.add_argument("--clr", type=float, default=0.005, help="固定化の学習率（今の板から短く学び直すので、最初の学習 0.01 より小さく）")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--front", type=int, default=8)
ap.add_argument("--back", type=int, default=6)
ap.add_argument("--reject", type=float, default=0.3)
ap.add_argument("--replay", type=float, default=0.1, help="付け足し・固定化で混ぜる古いデータの割合")
ap.add_argument("--kd_w", type=float, default=1.0, help="受付のお手本合わせ（KL）の重み")
ap.add_argument("--back_w", type=float, default=0.1, help="後段の生の点数のお手本合わせ（二乗誤差）の重み")
ap.add_argument("--variants", default="K1,K2,K3",
                help="K1 / K2 / K3 に続けて、f = 前段の光そのものもお手本に合わせる（重み --field_w）、b<数> = 後段のお手本合わせの重みをその数に（例 K2b10、K1f）")
ap.add_argument("--field_w", type=float, default=1.0, help="f つきの版で、前段の光をお手本の前段の光に合わせる（相対の二乗誤差）重み")
ap.add_argument("--n_train", type=int, default=4000)
ap.add_argument("--n_generic", type=int, default=4000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
a = ap.parse_args()
T_START = time.time()
DEV = a.device
steps = a.steps or (1500 if a.task == "synth4" else 3000)
csteps = a.csteps or steps
torch.manual_seed(a.seed)
D = s2.load_task(a.task, a.seed, L=a.L, n_train=a.n_train)
K, C, owners, L = len(D["owners"]), D["C"], D["owners"], a.L
NEW, OLD = K - 1, list(range(K - 1))
G = t2.make_generic(a.n_generic, L=L, seed=a.seed + 999) if a.reject > 0 else None
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
FFT_E = s2.n_ffts(16)
FFT_FULL = K * FFT_E
fft_front, fft_back, fft_thin = 2 * a.front, s2.n_ffts(a.back), s2.n_ffts(2)
timing = {}

# ---- 1. 足す前の系（実験08 と同じ手順・同じ seed）-------------------------------------------------
t0 = time.time()
etr = D["etr"]
old_tr = etr < NEW
router = p2.Net2D(L, NEW, plates=["xy"] * a.front, patches=a.patches, seed=a.seed * 100 + 70 + a.front).to(DEV)
s2.train_cls(router, D["Xtr"][old_tr], etr[old_tr], steps, a.seed, DEV)
router.eval()
for p in router.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def ff(net, X, bs=500):
    return torch.cat([s2.front_field(net, s2.field(X[i:i + bs].to(DEV))).cpu() for i in range(0, len(X), bs)])


@torch.no_grad()
def exits(net, Fx, bs=500):
    return torch.cat([s2.lens_scores(net, Fx[i:i + bs].to(DEV)).cpu() for i in range(0, len(Fx), bs)])


Ftr, Fte, Fva = ff(router, D["Xtr"]), ff(router, D["Xte"]), ff(router, D["Xva"])
FG = ff(router, G) if G is not None else None
Z = {"tr": exits(router, Ftr), "te": exits(router, Fte), "va": exits(router, Fva)}
timing["front"] = time.time() - t0
say(f"前段（古い {NEW} 人の受付）{timing['front']:.0f} 秒")
backs = []
t0 = time.time()
for k in range(K):
    net = p2.Net2D(L, C, plates=["xy"] * a.back, patches=a.patches, seed=a.seed * 100 + 80 + k).to(DEV)
    m = etr == k
    p2.train(net, Ftr[m], D["ytr"][m], steps=steps, seed=a.seed * 100 + k, reject=FG, reject_weight=a.reject, schedule="cosine")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    backs.append(net)
timing["backs"] = time.time() - t0
say(f"後段 {K} 人 {timing['backs']:.0f} 秒")
S_te = [s2.scores(n_, Fte, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]
S_va = [s2.scores(n_, Fva, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)]


def run(Zr, S, cal, m):
    Kx = len(S)
    top = Zr.argsort(1, descending=True)[:, :m]
    sel = torch.zeros(len(Zr), Kx, dtype=torch.bool)
    sel[torch.arange(len(Zr))[:, None], top] = True
    sc = s2.combine(Zr, S, *cal, True) if m == 2 else s2.combine(Zr, S, torch.tensor(1.0), torch.ones(Kx), torch.zeros(Kx), False)
    return s2.select(sc, cols[:Kx], sel), top


def metr(pred_col, top, rows, e, ycol):
    return dict(acc=(pred_col[rows] == ycol[rows]).float().mean().item(),
                route_top1=(top[rows, 0] == e[rows]).float().mean().item(),
                route_topm=(top[rows] == e[rows][:, None]).any(1).float().mean().item())


def evaluate_full(Zte, Zva, Ste, Sva, fft_extra):
    """K 人そろった系の評価（全・古い・新しい）。"""
    out = {}
    cal = s2.fit_joint(Zva, Sva, y_va_col)
    allr = torch.ones(len(Zte), dtype=torch.bool)
    for m in (1, 2):
        pc, top = run(Zte, Ste, cal, m)
        out[f"top{m}"] = dict(all=metr(pc, top, allr, e_te, y_te_col), old=metr(pc, top, old_te, e_te, y_te_col),
                              new=metr(pc, top, new_te, e_te, y_te_col), ffts=fft_front + 1 + fft_extra + m * fft_back)
    return out, pc


res = {}
cal_old = s2.fit_joint(Z["va"][old_va], [s[old_va] for s in S_va[:NEW]], y_va_col[old_va])
before = {}
for m in (1, 2):
    pc, top = run(Z["te"][old_te], [s[old_te] for s in S_te[:NEW]], cal_old, m)
    before[m] = pc
    res[f"before-top{m}"] = dict(old=metr(pc, top, torch.ones(len(pc), dtype=torch.bool), e_te[old_te], y_te_col[old_te]),
                                 ffts=fft_front + 1 + m * fft_back)


# ---- 2. 付け足した系（お手本）: thin 窓口、古いデータ replay + 下駄 ---------------------------------------
class ThinExit(nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.net = p2.Net2D(L, 1, plates=["xy"] * 2, patches=a.patches, seed=seed)

    def forward(self, Fx):
        return self.net(Fx)[:, 0]


t0 = time.time()
keep = etr == NEW
g = torch.Generator().manual_seed(a.seed)
for k in OLD:
    idx = (etr == k).nonzero()[:, 0]
    keep[idx[torch.randperm(len(idx), generator=g)[:max(1, int(round(len(idx) * a.replay)))]]] = True
sub = keep.nonzero()[:, 0]                                  # 付け足し・固定化に使う学習データ（新しい全部 + 古い 1 割）
thin = ThinExit(a.seed * 100 + 60).to(DEV)
opt = torch.optim.Adam(thin.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
gen = torch.Generator().manual_seed(a.seed)
for _ in range(steps):
    j = sub[torch.randint(len(sub), (256,), generator=gen)]
    logits = torch.cat([Z["tr"][j].to(DEV), thin(Ftr[j].to(DEV))[:, None]], 1)
    loss = F.cross_entropy(logits, etr[j].to(DEV))
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
thin.eval()
for p in thin.parameters():
    p.requires_grad_(False)


@torch.no_grad()
def thin_scores(Fx, bs=500):
    return torch.cat([thin(Fx[i:i + bs].to(DEV)).cpu() for i in range(0, len(Fx), bs)])


znew = {"te": thin_scores(Fte), "va": thin_scores(Fva)}
bshift = torch.zeros(1, requires_grad=True)
opt2 = torch.optim.Adam([bshift], lr=0.05)
for _ in range(300):
    l2 = F.cross_entropy(torch.cat([Z["va"], (znew["va"] + bshift)[:, None]], 1), e_va)
    opt2.zero_grad()
    l2.backward()
    opt2.step()
bshift = bshift.detach()
Zt = {s: torch.cat([Z[s], (znew[s] + bshift)[:, None]], 1) for s in ("te", "va")}
r_add, _ = evaluate_full(Zt["te"], Zt["va"], S_te, S_va, fft_thin)
for m in (1, 2):
    pc, _ = run(Z["te"][old_te], [s[old_te] for s in S_te[:NEW]], cal_old, m)
    r_add[f"top{m}"]["undo_agree"] = (pc == before[m]).float().mean().item()   # 付け足しは外せば完全に元どおり（実験08）
    res[f"added-top{m}"] = r_add[f"top{m}"]
timing["add"] = time.time() - t0
say(f"付け足し（thin、古いデータ {a.replay:.0%} + 下駄 {float(bshift):+.2f}）{timing['add']:.0f} 秒")

# ---- 3. お手本の答えを 1 回だけ計算（固定化の学習データ = sub + 担当なしの絵）-------------------------------------
X_c = D["Xtr"][sub]
e_c, y_c = etr[sub], D["ytr"][sub]
lab_c = torch.ones(len(sub), dtype=torch.bool)
if G is not None:
    X_c = torch.cat([X_c, G])
    e_c = torch.cat([e_c, torch.full((len(G),), -1)])
    y_c = torch.cat([y_c, torch.full((len(G),), -1)])
    lab_c = torch.cat([lab_c, torch.zeros(len(G), dtype=torch.bool)])
F_c = torch.cat([Ftr[sub], FG]) if G is not None else Ftr[sub]
with torch.no_grad():
    T_route = torch.cat([exits(router, F_c), (thin_scores(F_c) + bshift)[:, None]], 1)          # (N, K)
    T_back = torch.cat([s2.scores(n_, F_c, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(backs)], 1)  # (N, Σ担当)
del Ftr
say(f"固定化の学習データ: 新しい {int((e_c == NEW).sum())} + 古い {int(((e_c >= 0) & (e_c < NEW)).sum())} + 担当なし {int((e_c < 0).sum())}")


def make_student():
    fr = copy.deepcopy(router)
    fr.det = p2.window_grid(L, L, K, 4, device=DEV)                     # 窓を K 人分に作り直す（板はそのまま）
    with torch.no_grad():
        fr.bias = nn.Parameter(torch.cat([router.bias, router.bias.mean(0, keepdim=True)]))
    bk = [copy.deepcopy(n_) for n_ in backs]
    return fr, bk


def route_logits(fr, u):
    """前段 → 前段の光 u と、レンズ → 窓の受付の点数。"""
    v = s2.front_field(fr, u)
    return v, s2.lens_scores(fr, v)


def consolidate(name):
    import re
    mt = re.fullmatch(r"(K[123])(f?)(?:b([0-9.]+))?", name)
    if mt is None:
        raise ValueError(f"版の名前が読めない: {name}")
    variant, use_field = mt.group(1), mt.group(2) == "f"
    back_w = float(mt.group(3)) if mt.group(3) else a.back_w
    fr, bk = make_student()
    train_back = variant in ("K2", "K3")
    params = [p for n_, p in fr.named_parameters() if n_ in ("A", "phi", "gain", "bias")]
    for p in params:
        p.requires_grad_(True)
    if train_back:
        for n_ in bk:
            for nm, p in n_.named_parameters():
                if nm in ("A", "phi", "gain", "bias"):
                    p.requires_grad_(True)
                    params.append(p)
    opt = torch.optim.Adam(params, lr=a.clr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, csteps)
    gen = torch.Generator().manual_seed(a.seed + 10)
    fr.train()
    for n_ in bk:
        n_.train()
    t0 = time.time()
    for _ in range(csteps):
        j = torch.randint(len(X_c), (256,), generator=gen)
        u = s2.field(X_c[j].to(DEV))
        lab = lab_c[j].to(DEV)
        v, zr = route_logits(fr, u)
        ej = e_c[j].to(DEV)
        loss = F.cross_entropy(zr[lab], ej[lab])
        if variant in ("K1", "K2"):
            tr = T_route[j].to(DEV)
            loss = loss + a.kd_w * F.kl_div(F.log_softmax(zr, 1), F.log_softmax(tr, 1), log_target=True, reduction="batchmean")
            sb = torch.cat([n_(v)[:, owners[k]] for k, n_ in enumerate(bk)], 1)
            loss = loss + back_w * F.mse_loss(sb, T_back[j].to(DEV))
        if use_field:                                                   # 前段の光そのものをお手本に合わせる（後段が見る光を変えない）
            fv = F_c[j].to(DEV)
            loss = loss + a.field_w * (v - fv).abs().pow(2).mean() / fv.abs().pow(2).mean()
        if variant == "K3":                                             # K3: 正解のラベルだけ
            yj = y_c[j].to(DEV)
            for k, n_ in enumerate(bk):
                mk = ej == k
                if mk.any():
                    loss = loss + F.cross_entropy(n_(v[mk]), yj[mk]) / K
                if a.reject > 0 and (~lab).any():
                    loss = loss - a.reject * F.log_softmax(n_(v[~lab]), 1).mean() / K
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    sec = time.time() - t0
    n_tr = fr.n_params() + (sum(n_.n_params() for n_ in bk) if train_back else 0)   # 固める前に数える（n_params は学習する板だけを数える）
    fr.eval()
    for n_ in bk:
        n_.eval()
        for p in n_.parameters():
            p.requires_grad_(False)
    for p in fr.parameters():
        p.requires_grad_(False)
    return fr, bk, sec, n_tr


for variant in a.variants.split(","):
    fr, bk, sec, n_tr = consolidate(variant)
    Fte2, Fva2 = ff(fr, D["Xte"]), ff(fr, D["Xva"])
    Zs = {"te": exits(fr, Fte2), "va": exits(fr, Fva2)}
    S2_te = [s2.scores(n_, Fte2, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(bk)]
    S2_va = [s2.scores(n_, Fva2, DEV, owners[k], prep=lambda v: v) for k, n_ in enumerate(bk)]
    r, _ = evaluate_full(Zs["te"], Zs["va"], S2_te, S2_va, 0)
    # 取消: 固定化した系から新しい人（窓・後段）を外し、古い検証用データで較正し直して、足す前の答えと比べる
    cal_u = s2.fit_joint(Zs["va"][old_va][:, :NEW], [s[old_va] for s in S2_va[:NEW]], y_va_col[old_va])
    for m in (1, 2):
        pc, top = run(Zs["te"][old_te][:, :NEW], [s[old_te] for s in S2_te[:NEW]], cal_u, m)
        r[f"top{m}"]["undo_agree"] = (pc == before[m]).float().mean().item()
        r[f"top{m}"]["undo_old_acc"] = (pc == y_te_col[old_te]).float().mean().item()
        r[f"top{m}"]["train_sec"] = round(sec, 1)
        r[f"top{m}"]["n_trainable"] = n_tr
        res[f"{variant}-top{m}"] = r[f"top{m}"]
    timing[variant] = sec
    say(f"固定化 {variant}: {sec:.0f} 秒、全 {r['top1']['all']['acc']:.3f} 古い {r['top1']['old']['acc']:.3f} 新しい {r['top1']['new']['acc']:.3f}"
        f" 受付（新しい）{r['top1']['new']['route_top1']:.3f} 取消の一致 {r['top1']['undo_agree']:.3f}")
    del Fte2, Fva2

for v in res.values():
    v["ffts_pct"] = 100.0 * v["ffts"] / FFT_FULL

append_result("10", {"script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed, "steps": steps, "csteps": csteps,
                     "K": K, "C": C, "new_expert": NEW, "fft_full": FFT_FULL, "bias_shift": float(bshift),
                     "n_consolidate": dict(new=int((e_c == NEW).sum()), old=int(((e_c >= 0) & (e_c < NEW)).sum()), generic=int((e_c < 0).sum())),
                     "n_all_train": int(len(etr)), "timing_sec": {k: round(v, 1) for k, v in timing.items()},
                     "device_name": device_name(DEV), "log": log, "results": res, "sec": round(time.time() - T_START, 1)}, a.out)
for k, v in res.items():
    if "all" in v:
        print(f"{k:14s} all {v['all']['acc']:.3f} old {v['old']['acc']:.3f} new {v['new']['acc']:.3f} "
              f"route_new {v['new']['route_top1']:.3f} FFT% {v['ffts_pct']:.1f} undo {v.get('undo_agree')}")

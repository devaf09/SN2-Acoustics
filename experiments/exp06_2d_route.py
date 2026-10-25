"""実験06: 2 次元のグレインで、専門家を全員動かす方式と振り分け役を比べる（1 次元の実験01・02 の 2D 版）。

着想: SN2-Parts-2D で 2D の部品（sn2_parts_2d 0.11.0）が育ち、担当なしの絵を見せて学習すれば足すだけの融合が 4 技能でも 100%
      （SN2-Parts-2D 実験10・12）。では 2D でも「全員を動かさずに、担当だけを動かす」ことで計算を減らせるか（再開の計画、2026-10-26）。
方法: 専門家 = Net2D（板 ① 16 枚、区画 8、64 × 64）を、自分のクラスのデータ + 担当なしの一般の絵（重み 0.3。--reject 0 でなし）で別々に学習。
      全クラスの窓を持ち、自分の担当の窓だけをラベルに学ぶ（SN2-Parts-2D 実験12 と同じ）。学習済みは results/models2d/ に保存。
比べるもの:
  全員を動かす: 足す（p2.merge）・自信で選ぶ（担当の窓の softmax の最大値）・較正（生の点数、検証用データで τ_k・b_k を合わせる）
  正解の専門家へ（上限）、ランダム（ゼロ点）、1 本で全クラス（同じ大きさの Net2D を全データで学習）
  振り分け役: R1（入力の光のレンズ後の明るさ × 専門家ごとの透過率）、R2（薄い Net2D、板 4・8 枚、区画 8）。上位 1 人と、上位 2 人 + 較正
計算量: 2 次元 FFT の回数（板 1 枚 = 2 回、レンズ 1 回）。専門家 1 人 33 回、全員を動かす = 100%。実測時間は GPU で（符号化を除く）。

使い方: python experiments/exp06_2d_route.py --task synth4 --seed 0 --device cuda
出力: results/exp06.txt に 1 行の JSON を追記
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn

import sn2_2d as s2
from sn2_2d import p2, t2
from sn2_io import append_result, device_name

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="synth4", choices=["synth4", "fashion_mix", "fashion_sim"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=0, help="0 なら synth4 は 1500、fashion は 3000")
ap.add_argument("--L", type=int, default=64)
ap.add_argument("--plates", type=int, default=16)
ap.add_argument("--patches", type=int, default=8)
ap.add_argument("--reject", type=float, default=0.3, help="担当なしの絵の重み（0 なら使わない）")
ap.add_argument("--routers", default="R1,R2-d4,R2-d8")
ap.add_argument("--flat", type=int, default=1, help="1 本で全クラスの比較相手を学習するか")
ap.add_argument("--n_train", type=int, default=4000, help="synth4 の 1 課題あたりの学習の数")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default=None)
a = ap.parse_args()
T_START = time.time()
DEV = a.device
steps = a.steps or (1500 if a.task == "synth4" else 3000)
torch.manual_seed(a.seed)
D = s2.load_task(a.task, a.seed, L=a.L, n_train=a.n_train)
K, C, owners = len(D["owners"]), D["C"], D["owners"]
L = a.L
G = t2.make_generic(4000, L=L, seed=a.seed + 999) if a.reject > 0 else None
FFT_E = s2.n_ffts(a.plates)
FFT_ALL = K * FFT_E
log = []


def say(m):
    log.append(m)
    print(m, flush=True)


# ---- 専門家 ------------------------------------------------------------------------------
tag = f"{a.task}_r{a.reject:g}"
mdir = os.path.join(s2.ROOT, "results", "models2d", tag, f"seed{a.seed}")
os.makedirs(mdir, exist_ok=True)
experts, solo = [], []
for k in range(K):
    t0 = time.time()
    net = p2.Net2D(L, C, plates=["xy"] * a.plates, patches=a.patches, seed=a.seed * 100 + k).to(DEV)
    m = D["etr"] == k
    s2.train_cls(net, D["Xtr"][m], D["ytr"][m], steps, a.seed * 100 + k, DEV, reject=G, reject_weight=a.reject)
    net.eval()
    torch.save({"state": net.state_dict(), "config": dict(L=L, n_classes=C, plates=["xy"] * a.plates, patches=a.patches,
                                                          seed=a.seed * 100 + k), "own": owners[k]}, os.path.join(mdir, f"expert{k}.pt"))
    experts.append(net)
    mt = D["ete"] == k
    S = s2.scores(net, D["Xte"][mt], DEV, owners[k])
    solo.append((torch.tensor(owners[k])[S.argmax(1)] == D["yte"][mt]).float().mean().item())
    say(f"専門家 {k}: 単独の正解率 {solo[-1]:.3f}、{time.time() - t0:.0f} 秒")
params_expert = experts[0].n_params()

# ---- 専門家の点数（テスト・検証）-------------------------------------------------------------------
S_te = [s2.scores(e, D["Xte"], DEV, owners[k]) for k, e in enumerate(experts)]
S_va = [s2.scores(e, D["Xva"], DEV, owners[k]) for k, e in enumerate(experts)]
y_te, e_te, y_va = D["yte"], D["ete"], D["yva"]
N = len(y_te)
cols, off = [], 0
for o in owners:
    cols.append(list(range(off, off + len(o))))
    off += len(o)
col2class = torch.tensor([c for o in owners for c in o])
acc = lambda pred_col: (col2class[pred_col] == y_te).float().mean().item()
route_of = lambda pred_col: (torch.tensor([k for k, o in enumerate(owners) for _ in o])[pred_col] == e_te).float().mean().item()
ones, zeros = torch.ones(K), torch.zeros(K)
allsel = torch.ones(N, K, dtype=torch.bool)
Ute_cpu = D["Xte"]


def run_selected(sel, extra=None):
    """選んだ専門家だけを実際に動かす時間（1 入力あたり ms、符号化を除く）"""
    U = s2.field(Ute_cpu.to(DEV))
    seld = sel.to(DEV)

    def fn():
        if extra is not None:
            extra(U)
        for k, e in enumerate(experts):
            idx = seld[:, k].nonzero()[:, 0]
            for i in range(0, len(idx), 500):
                e(U[idx[i:i + 500]])
    return s2.timed(fn, DEV) / N * 1000


res = {}
t_all = run_selected(allsel)
common_all = dict(ffts=FFT_ALL, ffts_pct=100.0, ms=t_all, params=K * params_expert, extra_params=0)
with torch.no_grad():
    merged = p2.merge(experts)
    Sm = s2.scores(merged, Ute_cpu, DEV)
    pred = Sm[:, col2class].argmax(1)                      # 担当のある窓だけから最大
res["all-avg"] = dict(acc=acc(pred), route_top1=route_of(pred), **common_all)
pred = s2.combine(None, S_te, torch.tensor(1.0), ones, zeros, False).argmax(1)
res["all-conf"] = dict(acc=acc(pred), route_top1=route_of(pred), **common_all)
class2col = torch.empty(C, dtype=torch.long)
class2col[col2class] = torch.arange(len(col2class))
y_va_col = class2col[y_va]
_, tj, bj = s2.fit_joint(None, S_va, y_va_col)
pred = s2.combine(None, S_te, torch.tensor(1.0), tj, bj, True).argmax(1)
res["all-jcal"] = dict(acc=acc(pred), route_top1=route_of(pred), **common_all, tau=tj.tolist(), b=bj.tolist())
for name, choice in (("oracle", e_te), ("random", torch.randint(K, (N,), generator=torch.Generator().manual_seed(a.seed)))):
    sel = torch.nn.functional.one_hot(choice, K).bool()
    pred = s2.select(s2.combine(None, S_te, torch.tensor(1.0), ones, zeros, False), cols, sel)
    res[f"{name}-top1"] = dict(acc=acc(pred), route_top1=(choice == e_te).float().mean().item(), ffts=FFT_E,
                               ffts_pct=100.0 / K, ms=run_selected(sel), params=K * params_expert, extra_params=0)

# ---- 1 本で全クラス ------------------------------------------------------------------------------
if a.flat:
    t0 = time.time()
    flat = p2.Net2D(L, C, plates=["xy"] * a.plates, patches=a.patches, seed=a.seed * 100 + 90).to(DEV)
    s2.train_cls(flat, D["Xtr"], D["ytr"], steps, a.seed * 100 + 90, DEV)
    flat.eval()
    Sf = s2.scores(flat, Ute_cpu, DEV)
    pf = Sf[:, col2class].argmax(1)
    res["flat-1net"] = dict(acc=acc(pf), route_top1=route_of(pf), ffts=FFT_E, ffts_pct=100.0 / K,
                            ms=run_selected(torch.zeros(N, K, dtype=torch.bool), extra=lambda U: [flat(U[i:i + 500]) for i in range(0, N, 500)]),
                            params=flat.n_params(), extra_params=None)
    say(f"1 本で全クラス: {res['flat-1net']['acc']:.3f}、{time.time() - t0:.0f} 秒")

# ---- 振り分け役 ----------------------------------------------------------------------------------
for rname in a.routers.split(","):
    t0 = time.time()
    if rname == "R1":
        r = s2.R1(K, L, a.seed).to(DEV)
        p2.train(r, D["Xtr"], D["etr"], steps=steps, lr=0.05, seed=a.seed, to_input=s2.field, schedule="cosine")
        ffts_r, params_r = 1, sum(p.numel() for p in r.parameters())
    else:
        d = int(rname.split("-d")[1])
        r = p2.Net2D(L, K, plates=["xy"] * d, patches=a.patches, seed=a.seed * 100 + 50 + d).to(DEV)
        s2.train_cls(r, D["Xtr"], D["etr"], steps, a.seed, DEV)
        ffts_r, params_r = s2.n_ffts(d), r.n_params()
    r.eval()
    Zte, Zva = s2.scores(r, Ute_cpu, DEV), s2.scores(r, D["Xva"], DEV)
    al, tau, b = s2.fit_joint(Zva, S_va, y_va_col)
    top_all = Zte.argsort(1, descending=True)
    for m in (1, 2):
        if m > K:
            continue
        top = top_all[:, :m]
        sel = torch.zeros(N, K, dtype=torch.bool)
        sel[torch.arange(N)[:, None], top] = True
        if m == 1:
            sc = s2.combine(Zte, S_te, torch.tensor(1.0), ones, zeros, False)
        else:
            sc = s2.combine(Zte, S_te, al, tau, b, True)
        pred = s2.select(sc, cols, sel)
        ffts = ffts_r + m * FFT_E
        res[f"{rname}-top{m}" + ("-jcal" if m == 2 else "")] = dict(
            acc=acc(pred), route_top1=(top[:, 0] == e_te).float().mean().item(),
            route_topm=(top == e_te[:, None]).any(1).float().mean().item(), ffts=ffts, ffts_pct=100.0 * ffts / FFT_ALL,
            ms=run_selected(sel, extra=lambda U, r=r: [r(U[i:i + 500]) for i in range(0, N, 500)]),
            params=K * params_expert + params_r, extra_params=params_r)
    say(f"{rname}: 振り分けの正解率 {res[f'{rname}-top1']['route_top1']:.3f}、{time.time() - t0:.0f} 秒")

append_result("06", {"script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed, "steps": steps,
                     "K": K, "C": C, "n_test": N, "solo": solo, "fft_per_expert": FFT_E, "params_expert": params_expert,
                     "device_name": device_name(DEV), "log": log, "results": res, "sec": round(time.time() - T_START, 1)}, a.out)
for k, v in res.items():
    print(f"{k:16s} acc {v['acc']:.3f}  route1 {v['route_top1']:.3f}  FFT {v['ffts']} ({v['ffts_pct']:.1f}%)  ms {v['ms']:.4f}")

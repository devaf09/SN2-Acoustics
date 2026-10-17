"""実験01・02 の 1 段目: 専門家のグレイン（SinWaveNet 既定設定）を 4 人、各自のデータだけで学習して保存する。

着想: 振り分け役を比べるには、まず「得意分野の違う専門家」が要る。Sin-CNN2 の最小版 v1.1.0 の既定設定をそのまま使う。
方法: 専門家 i はクラス 4i..4i+3 を担当（窓は全部で 16）。自分のデータ（学習 500 件）だけで 1500 ステップ学習する。
      2 専門家の比較には、このうち 0 番と 1 番をそのまま使う（task_skills(2, seed) と同じデータになる）。
比べるもの: 専門家単独の正解率（自分の 4 クラスの中で）。

使い方: python experiments/exp01_experts.py --task freq --seed 0 --device cuda   （task=order は実験02 用）
保存先: results/models/{task}/seed{seed}/expert{i}.pt と train_log.txt
結果: 専門家 1 人 = 1 行の JSON を results/exp01.txt（task=freq）または results/exp02.txt（task=order）に追記（part="expert"）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import torch

from vendor import sinwave_minimal_v1_1_0 as sw
from sn2_io import append_result, device_name, models_dir
from sn2_tasks import get_skills

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--device", default="cpu")
ap.add_argument("--task", default="freq", help="freq（実験01、task_skills）/ order（実験02、順序だけで決まる課題）")
ap.add_argument("--out", default=None, help="結果の 1 行 JSON の追記先（既定は results/exp01.txt または exp02.txt）")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

out = models_dir(a.task, a.seed)
os.makedirs(out, exist_ok=True)
skills = [(tr, te) for tr, te, _ in get_skills(a.task, a.K, a.seed)]
C = a.K * 4
log = open(os.path.join(out, "train_log.txt"), "a", encoding="utf-8")
for i, ((xtr, ytr), (xte, yte)) in enumerate(skills):
    path = os.path.join(out, f"expert{i}.pt")
    if os.path.exists(path):
        continue
    t0 = time.time()
    net = sw.SinWaveNet(n_slots=C, own=list(range(4 * i, 4 * i + 4)), seed=a.seed * 100 + i).to(a.device)
    sw.train(net, xtr, ytr, steps=a.steps, seed=a.seed * 100 + i)
    acc = sum(p == y for p, y in zip(sw.predict(net, xte), yte)) / len(yte)
    sw.save(net.cpu(), path)
    sec = time.time() - t0
    msg = (f"seed {a.seed} expert {i}: own {net.own}, train {len(xtr)}, test {len(xte)}, "
           f"steps {a.steps}, 単独の正解率 {acc:.4f}, {sec:.0f} s, threads {torch.get_num_threads()}, device {a.device}")
    log.write(msg + "\n")
    log.flush()
    append_result("01" if a.task == "freq" else "02",
                  {"part": "expert", "script": os.path.basename(__file__), "args": vars(a), "task": a.task, "seed": a.seed,
                   "expert": i, "own": net.own, "n_train": len(xtr), "n_test": len(xte), "solo_acc": acc,
                   "params": sum(p.numel() for p in net.parameters() if p.requires_grad), "device_name": device_name(a.device),
                   "sec": round(sec, 1)}, a.out)

"""実験 A1 の 1 段目: 専門家（SinWaveNet 既定設定）を 4 人、各自のデータだけで学習して保存する。

task_skills(4, seed) の専門家 i はクラス 4i..4i+3 を担当（窓は全部で 16）。
2 専門家の比較には、このうち 0 番と 1 番をそのまま使う（task_skills(2, seed) と同じデータになる）。

使い方: python experiments/a1_experts.py --seed 0 [--steps 1500] [--threads 4]
保存先: results/a1/seed{seed}/expert{i}.pt と train_log.txt
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
from vendor.sinwave_tasks import task_skills

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

out = os.path.join(ROOT, "results", "a1", f"seed{a.seed}")
os.makedirs(out, exist_ok=True)
skills = task_skills(a.K, seed=a.seed)
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
    msg = (f"seed {a.seed} expert {i}: own {net.own}, train {len(xtr)}, test {len(xte)}, "
           f"steps {a.steps}, 単独の正解率 {acc:.4f}, {time.time() - t0:.0f} s, threads {torch.get_num_threads()}, device {a.device}")
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

"""計測: 1 専門家（SinWaveNet 既定: wave+rainbow, L=1024, 16 層, 64 区画, T=8）の
学習 1 ステップ（バッチ 256）と、1 入力の推論にかかる時間。

使い方: python experiments/time_expert.py [--device cpu] [--batch 256] [--threads N]
GPU が他のプロセスに使われているときは CUDA_VISIBLE_DEVICES="" を付けて CPU で測る。
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import torch.nn.functional as F

from vendor import sinwave_minimal_v1_1_0 as sw
from vendor.sinwave_tasks import task_skills

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)

print(f"torch {torch.__version__}, device={a.device}, threads={torch.get_num_threads()}, "
      f"cpu={platform.processor()}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
skills = task_skills(4, seed=0)
(xtr, ytr), _ = skills[0]
net = sw.SinWaveNet(n_slots=16, own=[0, 1, 2, 3]).to(a.device)
print("学習する数:", sum(p.numel() for p in net.parameters() if p.requires_grad))
X = net.encode(xtr, device="cpu")
Y = torch.tensor(ytr)
opt = torch.optim.Adam(net.parameters(), lr=0.01)


def step(i):
    loss = F.cross_entropy(net(X[i].to(a.device)), Y[i].to(a.device))
    opt.zero_grad()
    loss.backward()
    opt.step()


g = torch.Generator().manual_seed(0)
step(torch.randint(len(Y), (a.batch,), generator=g))  # 準備運転
ts = []
for _ in range(a.steps):
    i = torch.randint(len(Y), (a.batch,), generator=g)
    t0 = time.perf_counter()
    step(i)
    if a.device != "cpu":
        torch.cuda.synchronize()
    ts.append(time.perf_counter() - t0)
ts.sort()
print(f"学習 1 ステップ（バッチ {a.batch}）: 中央値 {ts[len(ts)//2]*1000:.0f} ms（最小 {ts[0]*1000:.0f}, 最大 {ts[-1]*1000:.0f}）")

with torch.no_grad():
    u1 = net.encode(xtr[:1])
    net(u1)
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        net(u1)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"推論 1 入力（符号化を除く）: 中央値 {ts[25]*1000:.2f} ms")
    ts = []
    for k in range(20):
        t0 = time.perf_counter()
        net(net.encode([xtr[k]]))
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"推論 1 入力（符号化を含む）: 中央値 {ts[10]*1000:.2f} ms")
    ub = net.encode(xtr[:256])
    t0 = time.perf_counter()
    net(ub)
    print(f"推論 256 入力まとめて: {(time.perf_counter()-t0)*1000:.0f} ms（1 入力あたり {(time.perf_counter()-t0)/256*1000:.2f} ms）")

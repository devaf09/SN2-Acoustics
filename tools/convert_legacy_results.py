"""移行用（1 回だけ使う）: results/legacy/ と results/models/*/train_log.txt にある旧形式の結果を、
results/expNN.txt（1 回の実行 = 1 行の JSON）に書き出す。元のファイルは消さない。

    python tools/convert_legacy_results.py          # results/exp01.txt〜exp05.txt を作る（既にあれば止まる）
    python tools/convert_legacy_results.py --force  # 上書き

旧形式の結果には、実行時の引数が全部は入っていない。スクリプトの既定値と、実行したコマンド（experiments/sweeps/）から補い、
"args_reconstructed": true と、元のファイルを "migrated_from" に書く。実行時間は、専門家の学習（train_log）以外は残っていない（null）。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LEG = os.path.join(ROOT, "results", "legacy")
DATE = "2026-10-17"
lines = {k: [] for k in ("01", "02", "03", "04", "05")}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def add(exp, rec, src):
    lines[exp].append({"exp": exp, **rec, "args_reconstructed": True, "migrated_from": rel(src), "date": DATE})


def load(p):
    return json.load(open(p, encoding="utf-8"))


# ---- 実験01: 時間の計測 -----------------------------------------------------------------------
for dev, fn in (("cpu", "time_expert_cpu.txt"), ("cuda", "time_expert_gpu.txt")):
    p = os.path.join(LEG, fn)
    t = open(p, encoding="utf-8").read()
    num = lambda pat: float(re.search(pat, t).group(1))
    add("01", {"part": "time", "script": "exp01_time.py",
               "args": {"device": dev, "batch": 256, "steps": 5 if dev == "cpu" else 10, "threads": 0},
               "device_name": "cpu (Intel64 Family 6 Model 151, 12 threads)" if dev == "cpu" else "NVIDIA GeForce RTX 3060",
               "params": int(num(r"学習する数: (\d+)")), "train_step_ms": num(r"中央値 (\d+) ms"),
               "infer1_ms": num(r"符号化を除く）: 中央値 ([\d.]+) ms"), "infer1_with_encode_ms": num(r"符号化を含む）: 中央値 ([\d.]+) ms"),
               "infer_batch256_ms_per_input": num(r"1 入力あたり ([\d.]+) ms")}, p)

# ---- 専門家の学習（実験01 = freq、実験02 = order）---------------------------------------------------
pat = re.compile(r"seed (\d+) expert (\d+): own \[([\d, ]+)\], train (\d+), test (\d+), steps (\d+), 単独の正解率 ([\d.]+), (\d+) s, threads (\d+), device (\w+)")
for task, exp in (("freq", "01"), ("order", "02")):
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "models", task, "seed*", "train_log.txt"))):
        for ln in open(p, encoding="utf-8"):
            m = pat.search(ln)
            if not m:
                continue
            seed, i, own, ntr, nte, steps, acc, sec, thr, dev = m.groups()
            add(exp, {"part": "expert", "script": "exp01_experts.py",
                      "args": {"seed": int(seed), "steps": int(steps), "threads": 0, "K": 4, "device": dev, "task": task},
                      "task": task, "seed": int(seed), "expert": int(i), "own": [int(x) for x in own.split(",")],
                      "n_train": int(ntr), "n_test": int(nte), "solo_acc": float(acc), "params": 2065,
                      "device_name": "NVIDIA GeForce RTX 3060", "sec": float(sec)}, p)

# ---- 実験01: 振り分け（頻度課題）----------------------------------------------------------------
for p in sorted(glob.glob(os.path.join(LEG, "a1", "seed*", "route.json"))):
    d = load(p)
    add("01", {"part": "route", "script": "exp01_route.py",
               "args": {"seed": d["seed"], "steps": d["steps"], "threads": 0, "Ks": "4,2", "device": d["device"]},
               "task": "freq", "seed": d["seed"], "device_name": d["device_name"], "steps": d["steps"], "threads": d["threads"],
               "runs": d["runs"], "sec": None}, p)

# ---- 実験02: 順序課題と較正（直した版と、最初の誤った較正の版）----------------------------------------------
for sub, variant in (("a1b", "fixed"), ("a1b_v1_buggy_jcal", "jcal_bug_v1")):
    for p in sorted(glob.glob(os.path.join(LEG, sub, "*", "seed*", "route.json"))):
        d = load(p)
        add("02", {"script": "exp02_order.py", "variant": variant,
                   "args": {"task": d["task"], "seed": d["seed"], "steps": d["steps"], "Ks": "4,2", "device": d["device"]},
                   **{k: v for k, v in d.items() if k != "device"}, "sec": None}, p)

# ---- 実験03・04・05 ---------------------------------------------------------------------------
for p in sorted(glob.glob(os.path.join(LEG, "a1c", "*", "seed*", "route.json"))):
    d = load(p)
    add("03", {"script": "exp03_colors.py",
               "args": {"task": d["task"], "seed": d["seed"], "steps": d["steps"], "device": "cuda",
                        "configs": "8:8,8:4,8:2,8:1,16:4,16:2,16:1"}, **d, "sec": None}, p)
for sub in ("a1d", "a1d_f8b6"):
    for p in sorted(glob.glob(os.path.join(LEG, sub, "*", "seed*", "route.json"))):
        d = load(p)
        add("04", {"script": "exp04_shared_front.py",
                   "args": {"task": d["task"], "seed": d["seed"], "steps": d["steps"], "front": d["front"], "back": d["back"],
                            "device": "cuda"}, "device_name": "NVIDIA GeForce RTX 3060", **d, "sec": None}, p)
for p in sorted(glob.glob(os.path.join(LEG, "a1e", "*", "seed*", "route.json"))):
    d = load(p)
    add("05", {"script": "exp05_add_expert.py",
               "args": {"task": d["task"], "seed": d["seed"], "steps": d["steps"], "back": d["back"], "device": "cuda"},
               "device_name": "NVIDIA GeForce RTX 3060", **d, "sec": None}, p)

for exp, recs in lines.items():
    path = os.path.join(ROOT, "results", f"exp{exp}.txt")
    if os.path.exists(path) and "--force" not in sys.argv:
        sys.exit(f"{rel(path)} が既にある（上書きするなら --force）")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{rel(path)}: {len(recs)} 行")

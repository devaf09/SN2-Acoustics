"""results/expNN.txt（1 行 JSON）から、実験ごとの表（平均 ± 標準誤差）を作り直す。

    python tools/summarize.py            # results/summary.md を書き出して表示
    python tools/summarize.py --check    # 旧形式のまとめ（results/legacy/*summary.txt）と平均値が一致するか確かめる

正解率などは % で表す。seed が 1 本のときは「(seed 1 本)」と付ける。
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(exp):
    p = os.path.join(ROOT, "results", f"exp{exp}.txt")
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def mse(xs, scale=100.0, f="{:.1f}"):
    xs = [x * scale for x in xs if x is not None]
    if not xs:
        return "-"
    if len(xs) == 1:
        return f.format(xs[0]) + " (seed 1 本)"
    return (f + "±" + f).format(st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


# ---- 平均値の表（検算用にも使う）: {(exp, task, K, 方式): [各 seed の正解率]} --------------------------------
def collect():
    acc = defaultdict(list)
    extra = defaultdict(dict)
    route = defaultdict(list)
    for r in load("01"):
        if r.get("part") == "route":
            for run in r["runs"]:
                for m, v in run["results"].items():
                    acc[("01", "freq", run["K"], m)].append(v["acc"])
                    route[("01", "freq", run["K"], m)].append(v.get("route_top1"))
                    extra[("01", "freq", run["K"], m)] = v
    for r in load("02"):
        if "runs" in r and r.get("variant", "fixed") == "fixed":
            for run in r["runs"]:
                for m, v in run["results"].items():
                    acc[("02", r["task"], run["K"], m)].append(v["acc"])
                    route[("02", r["task"], run["K"], m)].append(v.get("route_top1"))
                    extra[("02", r["task"], run["K"], m)] = v
    for r in load("03"):
        for m, v in r["results"].items():
            acc[("03", r["task"], 4, m)].append(v["acc"])
            route[("03", r["task"], 4, m)].append(v.get("route_top1"))
            extra[("03", r["task"], 4, m)] = v
    for r in load("04"):
        tag = f"f{r['front']}b{r['back']}"
        for m, v in r["results"].items():
            acc[("04", r["task"], 4, f"{tag}:{m}")].append(v["acc"])
            route[("04", r["task"], 4, f"{tag}:{m}")].append(v.get("route_top1"))
            extra[("04", r["task"], 4, f"{tag}:{m}")] = v
    for r in load("05"):
        for m, v in r["results"].items():
            if "all" in v:
                for part in ("all", "old", "new"):
                    acc[("05", r["task"], 5, f"{m}:{part}")].append(v[part]["acc"])
                    route[("05", r["task"], 5, f"{m}:{part}")].append(v[part]["route_top1"])
            elif "old" in v:
                acc[("05", r["task"], 5, f"{m}:all")].append(v["acc"])
                acc[("05", r["task"], 5, f"{m}:old")].append(v["old"])
                acc[("05", r["task"], 5, f"{m}:new")].append(v["new"])
            else:
                acc[("05", r["task"], 5, f"{m}:all")].append(v["acc"])
    return acc, extra, route


def markdown():
    acc, extra, route = collect()
    out = ["# 実験の集計（tools/summarize.py が results/expNN.txt から作る）", "",
           "正解率は %、平均 ± 標準誤差（seed 3 本）。計算量は FFT の回数で、専門家を全員動かす場合を 100% とする。", ""]
    times = [r for r in load("01") if r.get("part") == "time"]
    if times:
        out += ["## 実験01: 1 専門家の時間", "", "| 装置 | 学習 1 ステップ（バッチ 256）ms | 推論 1 入力 ms | 256 入力まとめて ms/入力 |", "|---|---|---|---|"]
        for r in times:
            out.append(f"| {r['device_name']} | {r['train_step_ms']:.0f} | {r['infer1_ms']:.2f} | {r['infer_batch256_ms_per_input']:.2f} |")
        out.append("")
    for exp in ("01", "02", "03", "04", "05"):
        keys = sorted({k[1:3] for k in acc if k[0] == exp})
        experts = [r for r in load(exp) if r.get("part") == "expert"]
        for task, K in keys:
            out += [f"## 実験{exp}: 課題 {task}、専門家 {K} 人", ""]
            ex = [r["solo_acc"] for r in experts if r["task"] == task]
            if ex and K == 4:
                out += [f"専門家単独の正解率（{len(ex)} 人分）: {mse(ex)}%", ""]
            out += ["| 方式 | 全体の正解率 % | 振り分け top1 % | 計算量 % |", "|---|---|---|---|"]
            for k in [k for k in acc if k[0] == exp and k[1] == task and k[2] == K]:
                v = extra.get(k, {})
                pct = v.get("ffts_pct")
                out.append(f"| {k[3]} | {mse(acc[k])} | {mse(route.get(k, []))} | {'-' if pct is None else f'{pct:.1f}'} |")
            out.append("")
    return "\n".join(out) + "\n"


# ---- 検算: 旧形式のまとめの平均値と比べる -----------------------------------------------------------
def legacy_means():
    """旧まとめの表の行「| 名前 | 88.2±0.4 | ...」から、(exp, task, K, 方式) → 平均（%）を読む。"""
    got = {}
    files = {"01": ("a1_summary.txt", "freq"), "02o": ("a1b_order_summary.txt", "order"), "02f": ("a1b_freq_summary.txt", "freq"),
             "03": ("a1cd_summary.txt", None), "05": ("a1e_summary.txt", None)}
    for key, (fn, task) in files.items():
        txt = open(os.path.join(ROOT, "results", "legacy", fn), encoding="utf-8").read()
        exp, K, sec = key[:2], None, None
        for ln in txt.splitlines():
            m = re.match(r"## K=(\d)", ln)
            if m:
                K = int(m.group(1))
            m = re.match(r"## (a1c|a1d|a1d_f8b6) 課題 (\w+)", ln)
            if m:
                sec, task = m.group(1), m.group(2)
                K = 4
            m = re.match(r"## 課題 (\w+)", ln)
            if m:
                task, K = m.group(1), 5
            m = re.match(r"\| ([^|]+?) \| ([\d.]+)(?:±[\d.]+)? \|", ln)
            if not m or task is None:
                continue
            name, val = m.group(1), float(m.group(2))
            if exp == "03":
                e = {"a1c": "03", "a1d": "04", "a1d_f8b6": "04"}[sec]
                nm = name if sec == "a1c" else ("f8b8:" if sec == "a1d" else "f8b6:") + name
                got[(e, task, K, nm)] = val
            elif exp == "05":
                continue   # 実験05 は列が多いので、下で別に読む
            else:
                got[(exp, task, K, name)] = val
        if exp == "05":
            for ln in txt.splitlines():
                m = re.match(r"## 課題 (\w+)", ln)
                if m:
                    task = m.group(1)
                m = re.match(r"\| 足した後 (replay\d+) top(\d)[^|]*\| ([\d.]+)±[\d.]+ \| ([\d.]+)±[\d.]+ \| ([\d.]+)±[\d.]+ \|", ln)
                if m:
                    tag, mm, a_all, a_old, a_new = m.groups()
                    for part, v in (("all", a_all), ("old", a_old), ("new", a_new)):
                        got[("05", task, 5, f"after5-{tag}-top{mm}{'-jcal' if False else ''}:{part}")] = float(v)
                m = re.match(r"\| 比べる相手: 元の専門家 5 人全員 (\w+) \| ([\d.]+)±[\d.]+ \| ([\d.]+)±[\d.]+ \| ([\d.]+)±[\d.]+ \|", ln)
                if m:
                    nm, a_all, a_old, a_new = m.groups()
                    for part, v in (("all", a_all), ("old", a_old), ("new", a_new)):
                        got[("05", task, 5, f"orig5-all-{nm}:{part}")] = float(v)
    return got


def check():
    acc, _, _ = collect()
    old = legacy_means()
    n_ok, bad, missing = 0, [], []
    for k, v in old.items():
        if k not in acc:
            missing.append(k)
            continue
        new = round(100 * st.mean(acc[k]), 1)
        if abs(new - v) < 0.051:
            n_ok += 1
        else:
            bad.append((k, v, new))
    print(f"旧まとめの値 {len(old)} 個のうち、一致 {n_ok}、不一致 {len(bad)}、新しい集計に無い {len(missing)}")
    for b in bad:
        print("  不一致:", b)
    for m in missing:
        print("  無い:", m)
    return not bad and not missing


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if check() else 1)
    md = markdown()
    open(os.path.join(ROOT, "results", "summary.md"), "w", encoding="utf-8", newline="\n").write(md)
    print(md)

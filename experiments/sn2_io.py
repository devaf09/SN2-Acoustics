"""実験の入出力の共通部品。

- 学習済みの専門家（グレイン）の置き場所: results/models/{task}/seed{seed}/expert{i}.pt
- 結果: results/expNN.txt に「1 回の実行 = 1 行の JSON」で追記する（引数・結果・パラメータ数・時間を入れる）。
"""
from __future__ import annotations

import datetime
import json
import os
import platform

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def models_dir(task: str, seed: int) -> str:
    return os.path.join(ROOT, "results", "models", task, f"seed{seed}")


def device_name(dev: str) -> str:
    return torch.cuda.get_device_name() if dev != "cpu" else f"cpu ({platform.processor()}, {torch.get_num_threads()} threads)"


def append_result(exp: str, record: dict, out: str | None = None) -> str:
    """record を 1 行の JSON にして results/exp{exp}.txt（または out）に追記し、その行を返す。"""
    path = out or os.path.join(ROOT, "results", f"exp{exp}.txt")
    rec = {"exp": exp, **record, "date": datetime.datetime.now().isoformat(timespec="seconds")}
    line = json.dumps(rec, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return line

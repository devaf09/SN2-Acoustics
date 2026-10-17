"""SN2-Acoustics の合成課題。

get_skills(task, K, seed) → skills[k] = ((xtr, ytr), (xte, yte), (xva, yva))。専門家 k はクラス 4k..4k+3 を担当。
  task="freq" : Sin-CNN2 の task_skills（文字の頻度で解ける。実験 A1）。検証用は n_test=500 で作った後半 250 件
                （先頭 250 件は既定のテストと同じになることを確かめている）
  task="order": 順序だけで決まる課題（実験 A1b、task_skills_order）
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vendor.sinwave_tasks import HIRA, KATA, task_skills


def task_skills_order(n_skills: int, per: int = 4, n_train: int = 500, n_test: int = 250, n_val: int = 250,
                      seed: int = 0, n_key: int = 5):
    """順序だけで決まる課題。
    キー文字 n_key 個（既定 5 個）を全クラス共通で使い、どの文字列にもキー文字を 1 回ずつ必ず入れる。
    クラスは「どのキー文字 2 個が、どの順に隣り合っているか」（例: 「あい」と「いあ」は別のクラス）。
    残りのキー文字は、詰め物の文字（キー以外）で隔てて置くので、キー文字どうしが隣り合うのはその 1 組だけ。
    → キー文字の種類と回数はどのクラスでも同じ。詰め物の文字は一様にランダム。文字の頻度では解けない。
    長さ 8〜10 文字。クラス（順序つきの組 n_key·(n_key-1) = 20 通りから n_skills·per 個）は seed で専門家に割り当てる。"""
    rng = random.Random(seed)
    pool = HIRA + KATA
    rng.shuffle(pool)
    keys, filler = pool[:n_key], pool[n_key:]
    pairs = [(a, b) for a in keys for b in keys if a != b]
    rng.shuffle(pairs)
    C = n_skills * per
    assert C <= len(pairs)
    cls = pairs[:C]

    def sample(r, y):
        a, b = cls[y]
        units = [a + b] + [k for k in keys if k not in (a, b)]
        r.shuffle(units)
        n_extra = r.randint(0, 2)                     # 区切りの詰め物（必ず 1 個）に加える数
        gaps = [1] * (len(units) - 1)                 # 単位どうしの間
        ends = [0, 0]
        for _ in range(n_extra):
            j = r.randrange(len(gaps) + 2)
            if j < len(gaps):
                gaps[j] += 1
            else:
                ends[j - len(gaps)] += 1
        f = lambda n: "".join(r.choice(filler) for _ in range(n))
        s = f(ends[0]) + "".join(u + (f(gaps[i]) if i < len(gaps) else "") for i, u in enumerate(units)) + f(ends[1])
        return s

    skills = []
    for i in range(n_skills):
        rs = [random.Random(seed * 1000 + i * 3 + j + 1) for j in range(3)]
        out = []
        for r, n in zip(rs, (n_train, n_test, n_val)):
            ys = [r.randrange(i * per, (i + 1) * per) for _ in range(n * per // 4)]
            out.append(([sample(r, y) for y in ys], ys))
        skills.append(tuple(out))
    return skills, dict(keys=keys, classes=["".join(p) for p in cls])


def get_skills(task: str, K: int, seed: int):
    if task == "order":
        return task_skills_order(K, seed=seed)[0]
    if task == "freq":
        base = task_skills(K, seed=seed)
        big = task_skills(K, seed=seed, n_test=500)
        out = []
        for (tr, te), (_, te2) in zip(base, big):
            assert te2[0][:len(te[0])] == te[0] and te2[1][:len(te[1])] == te[1], "検証用の作り方が崩れている"
            n = len(te[0])
            out.append((tr, te, (te2[0][n:], te2[1][n:])))
        return out
    raise ValueError(task)


if __name__ == "__main__":
    sk, info = task_skills_order(4, seed=0)
    print(info)
    for k in range(4):
        print(k, list(zip(sk[k][0][0][:3], sk[k][0][1][:3])))

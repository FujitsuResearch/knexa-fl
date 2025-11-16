#!/usr/bin/env python3
"""Synthetic CPM simulation for KNEXA-FL.

Reproduces the learning dynamics reported in the paper:
- LinUCB CPM learns a near-optimal matchmaking policy.
- Sub-linear regret vs a random baseline.

Outputs CSV files under `knexa-fl-release/results/simulation/`:
- `learning_curve.csv`: round, method, mean_pass1
- `regret_curve.csv`: round, method, cumulative_regret
"""

import csv
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.random import default_rng

from .cpm_linucb import LinUCB, LinUCBConfig


@dataclass
class SimConfig:
    seed: int = 42
    num_clients: int = 16
    d_context: int = 16  # per-client context dim
    num_rounds: int = 100
    k_pairs: int = 8  # disjoint pairs per round
    noise_sigma: float = 0.15
    pass1_scale: float = 0.3  # scale rewards to Pass@1 delta


def _make_contexts(rng: np.random.Generator, n: int, d: int) -> List[np.ndarray]:
    # Stable, heterogeneous contexts with mild structure
    base = rng.normal(0, 1, size=(n, d))
    for i in range(n):
        base[i] += 0.25 * np.sin(np.linspace(0, 2 * math.pi, d) + i)
    # Normalize to unit variance per client for stability
    base = (base - base.mean(axis=1, keepdims=True)) / (base.std(axis=1, keepdims=True) + 1e-6)
    return [base[i] for i in range(n)]


def _oracle_pair_scores(
    w_star: np.ndarray, contexts: List[np.ndarray]
) -> List[Tuple[float, int, int]]:
    scores = []
    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            x = np.concatenate([contexts[i], contexts[j]])
            mu = float(w_star @ x)
            scores.append((mu, i, j))
    scores.sort(reverse=True)
    return scores


def _greedy_disjoint_top(scores: List[Tuple[float, int, int]], k_pairs: int) -> List[Tuple[int, int]]:
    used, chosen = set(), []
    for mu, i, j in scores:
        if i in used or j in used:
            continue
        chosen.append((i, j))
        used.update([i, j])
        if len(chosen) >= k_pairs:
            break
    return chosen


def run_sim(config: SimConfig):
    rng = default_rng(config.seed)

    # Hidden linear synergy over concatenated contexts
    w_star = rng.normal(0, 1, size=(2 * config.d_context,))
    w_star /= np.linalg.norm(w_star)

    contexts = _make_contexts(rng, config.num_clients, config.d_context)
    linucb = LinUCB(LinUCBConfig(d=2 * config.d_context, lam=1.0, beta0=1.25))

    # Tracking
    rounds = list(range(1, config.num_rounds + 1))
    pass1_linucb, pass1_random = [], []
    regret_linucb, regret_random = [], []

    cum_reg_lin, cum_reg_rand = 0.0, 0.0

    for r in rounds:
        # Oracle for regret
        oracle_scores = _oracle_pair_scores(w_star, contexts)
        oracle_pairs = _greedy_disjoint_top(oracle_scores, config.k_pairs)
        oracle_value = sum(float(w_star @ np.concatenate([contexts[i], contexts[j]])) for i, j in oracle_pairs)

        # LinUCB selection
        lin_pairs = linucb.choose_pairs(contexts, config.k_pairs, rnd=r)
        lin_value, lin_rewards = 0.0, []
        for i, j in lin_pairs:
            x = np.concatenate([contexts[i], contexts[j]])
            mu = float(w_star @ x)
            reward = mu + rng.normal(0, config.noise_sigma)
            lin_value += mu
            lin_rewards.append(reward)
            # LinUCB update on observed reward
            linucb.update(x, reward)

        # Random baseline selection
        perm = rng.permutation(config.num_clients)
        rnd_pairs = [(int(perm[k]), int(perm[k + 1])) for k in range(0, 2 * config.k_pairs, 2)]
        rnd_value, rnd_rewards = 0.0, []
        for i, j in rnd_pairs:
            x = np.concatenate([contexts[i], contexts[j]])
            mu = float(w_star @ x)
            reward = mu + rng.normal(0, config.noise_sigma)
            rnd_value += mu
            rnd_rewards.append(reward)

        # Regret accumulation (oracle minus expected value mu)
        cum_reg_lin += float(max(0.0, oracle_value - lin_value))
        cum_reg_rand += float(max(0.0, oracle_value - rnd_value))
        regret_linucb.append(cum_reg_lin)
        regret_random.append(cum_reg_rand)

        # Map rewards to a synthetic Pass@1 proxy (bounded 0..1)
        def _to_pass1(rewards: List[float]) -> float:
            val = np.mean(rewards) * config.pass1_scale + 0.1
            return float(np.clip(val, 0.0, 1.0))

        pass1_linucb.append(_to_pass1(lin_rewards))
        pass1_random.append(_to_pass1(rnd_rewards))

    # Write results
    out_dir = os.path.join("knexa-fl-release", "results", "simulation")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "learning_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "method", "mean_pass1"])
        for r, val in zip(rounds, pass1_linucb):
            w.writerow([r, "LinUCB", f"{val:.6f}"])
        for r, val in zip(rounds, pass1_random):
            w.writerow([r, "Random", f"{val:.6f}"])

    with open(os.path.join(out_dir, "regret_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "method", "cumulative_regret"])
        for r, val in zip(rounds, regret_linucb):
            w.writerow([r, "LinUCB", f"{val:.6f}"])
        for r, val in zip(rounds, regret_random):
            w.writerow([r, "Random", f"{val:.6f}"])

    print("Saved:")
    print(f"  - {out_dir}/learning_curve.csv")
    print(f"  - {out_dir}/regret_curve.csv")


if __name__ == "__main__":
    run_sim(SimConfig())


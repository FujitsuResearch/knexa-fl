import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class LinUCBConfig:
    d: int = 16
    lam: float = 1.0
    beta0: float = 1.0


class LinUCB:
    """Minimal LinUCB for CPM matchmaking.

    - Maintains A, b for linear UCB.
    - Scores pair contexts and greedily picks disjoint pairs.
    - Updates on observed scalar reward.
    """

    def __init__(self, config: LinUCBConfig):
        self.d = config.d
        self.lam = config.lam
        self.beta0 = config.beta0
        self.A = self.lam * np.eye(self.d, dtype=np.float32)
        self.b = np.zeros((self.d, 1), dtype=np.float32)

    def _theta(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)

    def choose_pairs(
        self,
        profiles: List[np.ndarray],
        k_pairs: int,
        rnd: int,
    ) -> List[Tuple[int, int]]:
        """Pick up to k disjoint pairs based on UCB over concatenated contexts.

        Returns a list of index pairs (i, j).
        """
        beta = self.beta0 / np.sqrt(max(1, rnd))
        candidates: List[Tuple[float, int, int]] = []
        A_inv = np.linalg.inv(self.A)
        theta = self._theta()
        for i, j in itertools.combinations(range(len(profiles)), 2):
            ctx = np.concatenate([profiles[i], profiles[j]]).astype(np.float32).reshape(-1, 1)
            mu = float(theta.T @ ctx)
            conf = float(np.sqrt(ctx.T @ A_inv @ ctx))
            ucb = mu + beta * conf
            candidates.append((ucb, i, j))

        candidates.sort(reverse=True)
        pairs, used = [], set()
        for ucb, i, j in candidates:
            if i not in used and j not in used and len(pairs) < k_pairs:
                pairs.append((i, j))
                used.update([i, j])
        return pairs

    def update(self, ctx_vec: np.ndarray, reward: float):
        """Update A, b given context and observed reward."""
        x = np.asarray(ctx_vec, dtype=np.float32).reshape(-1, 1)
        r = np.asarray([[reward]], dtype=np.float32)
        self.A += x @ x.T
        self.b += r * x

    def ucb_score(self, ctx_vec: np.ndarray, rnd: int = 1) -> float:
        beta = self.beta0 / np.sqrt(max(1, rnd))
        x = np.asarray(ctx_vec, dtype=np.float32).reshape(-1, 1)
        theta = self._theta()
        A_inv = np.linalg.inv(self.A)
        mu = float(theta.T @ x)
        conf = float(np.sqrt(x.T @ A_inv @ x))
        return float(mu + beta * conf)

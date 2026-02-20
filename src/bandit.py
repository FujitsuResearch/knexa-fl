import itertools, torch, numpy as np, logging
from src.globals import *
 
logger = logging.getLogger(__name__)
 
class LinUCB:
    def __init__(self, d=16):
        self.lam = LINUCB_LAMBDA
        self.A = self.lam * torch.eye(d)
        self.b = torch.zeros(d, 1)
        self.beta0 = LINUCB_BETA0
 
    def _theta(self):
        return torch.linalg.solve(self.A, self.b)
 
    def choose_pairs(self, profiles, k_pairs, rnd):
        beta = self.beta0 / np.sqrt(max(1, rnd))
        cand = []
        # Precompute A^{-1} once for efficiency; A is fixed during scoring
        with torch.no_grad():
            A_inv = torch.linalg.inv(self.A)
            theta = self._theta()
            for i, j in itertools.combinations(range(len(profiles)), 2):
                ctx = torch.tensor(
                    np.concatenate([profiles[i], profiles[j]]), dtype=torch.float32
                ).reshape(-1, 1)
                mu = (theta.T @ ctx).item()
                conf = torch.sqrt(ctx.T @ A_inv @ ctx).item()
                ucb = mu + beta * conf
                cand.append((ucb, i, j))
        cand.sort(reverse=True)
        pairs, used = [], set()
        for ucb, i, j in cand:
            if i not in used and j not in used and len(pairs) < k_pairs:
                alpha = KD_ALPHA_GRID[rnd % len(KD_ALPHA_GRID)]
                T = TEMP_DEFAULT + (rnd % 3) * 0.5
                pairs.append((i, j, alpha, T))
                used.update([i, j])
        return pairs
 
    def update(self, ctx_vec, reward, rnd):
        # Convert numpy array to tensor if needed
        if isinstance(ctx_vec, np.ndarray):
            ctx_vec = torch.tensor(ctx_vec, dtype=torch.float32)
        ctx_vec = ctx_vec.unsqueeze(1)
        self.A += ctx_vec @ ctx_vec.T
        self.b += reward * ctx_vec
        if rnd > 2:
            from src.globals_runtime import GLOBAL_KB_LOG
            avg_kb = np.mean(GLOBAL_KB_LOG)
            global DELTA_KB
            DELTA_KB *= avg_kb / KB_TARGET
            logger.info(f"Auto-tuned DELTA_KB to {DELTA_KB} (avg_kb={avg_kb})")
    
    def get_ucb_score(self, ctx_vec, rnd=1):
        """Get UCB score for a given context vector"""
        # Convert numpy array to tensor if needed
        if isinstance(ctx_vec, np.ndarray):
            ctx_vec = torch.tensor(ctx_vec, dtype=torch.float32)
        # Ensure column vector shape to avoid 1-D transpose warnings
        ctx_vec = ctx_vec.reshape(-1, 1)

        beta = self.beta0 / np.sqrt(max(1, rnd))
        with torch.no_grad():
            theta = self._theta()
            A_inv = torch.linalg.inv(self.A)
            mu = (theta.T @ ctx_vec).item()
            conf = torch.sqrt(ctx_vec.T @ A_inv @ ctx_vec).item()
            ucb = mu + beta * conf
        return float(ucb)

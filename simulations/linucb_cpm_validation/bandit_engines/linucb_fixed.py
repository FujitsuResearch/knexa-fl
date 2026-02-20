"""
Fixed LinUCB implementation for simulations

Avoids issues with the auto-tuning code that requires actual P2P communication.

Author: Inderjeet Singh
"""

import torch
import numpy as np
import itertools
from typing import List, Tuple


class LinUCBFixed:
    """LinUCB implementation without auto-tuning dependencies"""
    
    def __init__(self, d: int = 16, lambda_reg: float = 1.0, beta0: float = 1.0):
        self.d = d
        self.lam = lambda_reg
        self.A = self.lam * torch.eye(d)
        self.b = torch.zeros(d, 1)
        self.beta0 = beta0
        self.t = 0  # Track updates
        
    def _theta(self):
        """Compute theta with regularization to ensure stability"""
        # Always add small regularization for numerical stability
        # Use device and dtype of self.A to avoid type/device mismatches
        A_reg = self.A + 1e-4 * torch.eye(self.d, dtype=self.A.dtype, device=self.A.device)
        try:
            # Primary path: direct solve (fast and accurate)
            return torch.linalg.solve(A_reg, self.b)
        except RuntimeError as e:
            # Fallback: use Moore-Penrose pseudoinverse if A_reg is (near-)singular
            # This prevents crashes due to unexpected singularities
            return torch.linalg.pinv(A_reg) @ self.b
    
    def choose_pairs(self, profiles: List[np.ndarray], k_pairs: int, rnd: int) -> List[Tuple]:
        """Select k pairs using LinUCB"""
        beta = self.beta0 / np.sqrt(max(1, rnd + 1))
        cand = []
        
        for i, j in itertools.combinations(range(len(profiles)), 2):
            ctx = torch.tensor(np.concatenate([profiles[i], profiles[j]]), dtype=torch.float32)
            
            # Compute UCB with numerical stability
            theta = self._theta().squeeze()
            # Compute inverse with regularization; fall back to pseudoinverse if singular
            A_inv_reg = self.A + 1e-4 * torch.eye(self.d, dtype=self.A.dtype, device=self.A.device)
            try:
                A_inv = torch.linalg.inv(A_inv_reg)
            except RuntimeError:
                A_inv = torch.linalg.pinv(A_inv_reg)
            
            mean_reward = torch.dot(theta, ctx).item()
            confidence = beta * torch.sqrt(torch.dot(ctx, A_inv @ ctx)).item()
            ucb = mean_reward + confidence
            
            cand.append((ucb, i, j))
        
        cand.sort(reverse=True)
        pairs, used = [], set()
        
        for ucb, i, j in cand:
            if i not in used and j not in used and len(pairs) < k_pairs:
                # Fixed parameters for simulation
                alpha = 0.5 + (rnd % 3) * 0.1  # 0.5, 0.6, or 0.7
                T = 1.0 + (rnd % 3) * 0.5  # 1.0, 1.5, or 2.0
                pairs.append((i, j, alpha, T))
                used.update([i, j])
        
        return pairs
    
    def update(self, ctx_vec: np.ndarray, reward: float, rnd: int):
        """Update LinUCB parameters"""
        if isinstance(ctx_vec, np.ndarray):
            ctx_vec = torch.tensor(ctx_vec, dtype=torch.float32)
        if ctx_vec.dim() == 1:
            ctx_vec = ctx_vec.unsqueeze(1)
        
        self.A += ctx_vec @ ctx_vec.T
        self.b += reward * ctx_vec
        self.t += 1
    
    def get_ucb_score(self, ctx_vec: np.ndarray, rnd: int = 1) -> float:
        """Get UCB score for a context vector"""
        if isinstance(ctx_vec, np.ndarray):
            ctx_vec = torch.tensor(ctx_vec, dtype=torch.float32)
        
        beta = self.beta0 / np.sqrt(max(1, rnd))
        theta = self._theta().squeeze()
        # Robust inverse with regularization and fallback
        A_inv_reg = self.A + 1e-4 * torch.eye(self.d, dtype=self.A.dtype, device=self.A.device)
        try:
            A_inv = torch.linalg.inv(A_inv_reg)
        except RuntimeError:
            A_inv = torch.linalg.pinv(A_inv_reg)
        
        mean_reward = torch.dot(theta, ctx_vec).item()
        confidence = beta * torch.sqrt(torch.dot(ctx_vec, A_inv @ ctx_vec)).item()
        
        return mean_reward + confidence

"""
LinUCB Bandit Implementation for KNEXA-FL
Implements contextual bandit for intelligent peer matching
"""

import numpy as np
import torch
from typing import List, Tuple, Optional

class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) algorithm for contextual bandits.
    Used by the Central Profiler/Matchmaker (CPM) to learn optimal peer pairings.
    """
    
    def __init__(self, d: int = 32, lambda_param: float = 1.0, beta0: float = 1.0):
        """
        Initialize LinUCB bandit.
        
        Args:
            d: Dimension of context vectors (default: 32 for pairwise profiles)
            lambda_param: Regularization parameter
            beta0: Initial exploration parameter
        """
        self.d = d
        self.lambda_param = lambda_param
        self.beta0 = beta0
        
        # Initialize parameters
        self.A = self.lambda_param * torch.eye(d)  # d x d matrix
        self.b = torch.zeros(d, 1)  # d x 1 vector
        
        # Tracking
        self.num_rounds = 0
        self.total_reward = 0.0
    
    def _compute_theta(self) -> torch.Tensor:
        """Compute current parameter estimate theta = A^{-1}b"""
        return torch.linalg.solve(self.A, self.b)
    
    def get_ucb_score(self, context: np.ndarray, round_num: int = None) -> float:
        """
        Calculate UCB score for a given context vector.
        
        Args:
            context: Context vector (concatenated pairwise profiles)
            round_num: Current round number for exploration decay
            
        Returns:
            UCB score for the context
        """
        if round_num is None:
            round_num = max(1, self.num_rounds)
        
        # Convert to tensor if needed
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32)
        
        # Compute exploration parameter with decay
        beta = self.beta0 / np.sqrt(round_num)
        
        # Compute UCB: theta^T * x + beta * sqrt(x^T * A^{-1} * x)
        theta = self._compute_theta()
        A_inv = torch.linalg.inv(self.A)
        
        mean_reward = (theta.T @ context).item()
        confidence_width = beta * torch.sqrt(context.T @ A_inv @ context).item()
        
        ucb_score = mean_reward + confidence_width
        
        return ucb_score
    
    def select_pairs(self, 
                     profiles: List[np.ndarray], 
                     k_pairs: int,
                     round_num: int) -> List[Tuple[int, int, float, float]]:
        """
        Select k best disjoint pairs using LinUCB.
        
        Args:
            profiles: List of agent profile vectors
            k_pairs: Number of pairs to select
            round_num: Current round number
            
        Returns:
            List of (agent_i, agent_j, alpha, temperature) tuples
        """
        n_agents = len(profiles)
        candidates = []
        
        # Generate all possible pairs with UCB scores
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                # Create pairwise context
                context = np.concatenate([profiles[i], profiles[j]])
                ucb_score = self.get_ucb_score(context, round_num)
                
                candidates.append((ucb_score, i, j))
        
        # Sort by UCB score (descending)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select top k disjoint pairs
        selected_pairs = []
        used_agents = set()
        
        for ucb_score, i, j in candidates:
            if i not in used_agents and j not in used_agents:
                # Generate adaptive hyperparameters based on round
                alpha = 0.3 + 0.2 * (round_num % 5) / 5.0  # Cycle between 0.3-0.5
                temperature = 1.5 + 1.0 * (round_num % 3) / 3.0  # Cycle between 1.5-2.5
                
                selected_pairs.append((i, j, alpha, temperature))
                used_agents.update([i, j])
                
                if len(selected_pairs) >= k_pairs:
                    break
        
        return selected_pairs
    
    def update(self, context: np.ndarray, reward: float):
        """
        Update LinUCB parameters based on observed reward.
        
        Args:
            context: Context vector for the selected action
            reward: Observed reward from the action
        """
        # Convert to tensor if needed
        if isinstance(context, np.ndarray):
            context = torch.tensor(context, dtype=torch.float32)
        
        # Ensure context is column vector
        if len(context.shape) == 1:
            context = context.unsqueeze(1)
        
        # Update A and b
        self.A += context @ context.T
        self.b += reward * context
        
        # Update tracking
        self.num_rounds += 1
        self.total_reward += reward
    
    def get_statistics(self) -> dict:
        """Get bandit statistics"""
        avg_reward = self.total_reward / max(1, self.num_rounds) if self.num_rounds > 0 else 0.0
        
        return {
            'num_rounds': self.num_rounds,
            'total_reward': self.total_reward,
            'average_reward': avg_reward,
            'context_dimension': self.d,
            'lambda': self.lambda_param,
            'beta0': self.beta0
        }
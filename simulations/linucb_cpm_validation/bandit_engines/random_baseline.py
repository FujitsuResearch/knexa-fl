"""
Random Baseline Engine

Selects client pairs uniformly at random. Serves as the baseline
for comparison with intelligent pairing strategies.

Author: Inderjeet Singh
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import logging

from .base_engine import BasePairingEngine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_environment import SyntheticClient

logger = logging.getLogger(__name__)


class RandomBaselineEngine(BasePairingEngine):
    """Random pairing baseline"""
    
    def __init__(self, seed: int = None):
        """
        Initialize random baseline engine
        
        Args:
            seed: Random seed for reproducibility
        """
        super().__init__("Random")
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select k random disjoint pairs
        """
        n = len(clients)
        if n < 2:
            return []
        
        # Generate all possible pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Shuffle randomly
        np.random.shuffle(all_pairs)
        
        # Select disjoint pairs
        selected_pairs = []
        used_clients = set()
        
        for i, j in all_pairs:
            if i not in used_clients and j not in used_clients:
                # Randomly choose exchange type
                if clients[i].model_family == clients[j].model_family and np.random.random() > 0.5:
                    method = 'peft'
                    alpha = None
                    temperature = None
                else:
                    method = 'kd'
                    alpha = np.random.choice([0.5, 0.6, 0.7])
                    temperature = np.random.choice([1.0, 1.5, 2.0])
                
                metadata = {
                    'method': method,
                    'alpha': alpha,
                    'temperature': temperature,
                    'round': round_id
                }
                
                selected_pairs.append((i, j, metadata))
                used_clients.update([i, j])
                self.pairing_history.append((i, j, round_id))
                
                if len(selected_pairs) >= k_pairs:
                    break
        
        self.round_count = round_id
        logger.info(f"Random baseline selected {len(selected_pairs)} pairs for round {round_id}")
        
        return selected_pairs
    
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Update reward history (no learning for random baseline)
        """
        self.reward_history.append(reward)
        
    def reset(self):
        """Reset engine and reseed if needed"""
        super().reset()
        if self.seed is not None:
            np.random.seed(self.seed)
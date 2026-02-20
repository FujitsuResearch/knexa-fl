"""
Oracle Engine

Uses ground-truth rewards to select optimal pairings. Provides an
upper bound on achievable performance.

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
from reward_models import GroundTruthReward

logger = logging.getLogger(__name__)


class OracleEngine(BasePairingEngine):
    """Oracle pairing using ground-truth rewards"""
    
    def __init__(self):
        """Initialize oracle engine"""
        super().__init__("Oracle")
        self.reward_model = GroundTruthReward()
        self.true_rewards = []
        
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select optimal pairs based on true rewards
        """
        n = len(clients)
        candidates = []
        
        # Calculate true rewards for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                # Determine best exchange type
                if clients[i].model_family == clients[j].model_family:
                    # Try both methods
                    reward_peft, _ = self.reward_model.compute_reward(
                        clients[i], clients[j], 'peft', round_id
                    )
                    reward_kd, _ = self.reward_model.compute_reward(
                        clients[i], clients[j], 'kd', round_id
                    )
                    
                    if reward_peft > reward_kd:
                        method = 'peft'
                        reward = reward_peft
                        alpha = None
                        temperature = None
                    else:
                        method = 'kd'
                        reward = reward_kd
                        alpha = 0.6  # Optimal from experiments
                        temperature = 1.5
                else:
                    # Only KD possible
                    method = 'kd'
                    reward, _ = self.reward_model.compute_reward(
                        clients[i], clients[j], 'kd', round_id
                    )
                    alpha = 0.6
                    temperature = 1.5
                
                candidates.append((reward, i, j, method, alpha, temperature))
        
        # Sort by reward (descending)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select top k disjoint pairs
        selected_candidates = self._ensure_disjoint_pairs(candidates, k_pairs)
        
        # Convert to output format
        selected_pairs = []
        for reward, i, j, method, alpha, temperature in selected_candidates:
            self.true_rewards.append(reward)
            
            metadata = {
                'method': method,
                'alpha': alpha,
                'temperature': temperature,
                'true_reward': float(reward),
                'round': round_id
            }
            
            selected_pairs.append((i, j, metadata))
            self.pairing_history.append((i, j, round_id))
        
        self.round_count = round_id
        
        # Log selection
        avg_reward = np.mean([p[2]['true_reward'] for p in selected_pairs])
        logger.info(f"Oracle selected {len(selected_pairs)} pairs for round {round_id}, "
                   f"avg true reward: {avg_reward:.3f}")
        
        return selected_pairs
    
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Track observed rewards (for comparison with true rewards)
        """
        self.reward_history.append(reward)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including oracle performance"""
        stats = super().get_statistics()
        
        if self.true_rewards:
            stats['avg_true_reward'] = float(np.mean(self.true_rewards))
            stats['cumulative_true_reward'] = float(sum(self.true_rewards))
            
            # Compare observed vs true rewards
            if len(self.reward_history) == len(self.true_rewards):
                reward_diff = [obs - true for obs, true in 
                              zip(self.reward_history, self.true_rewards)]
                stats['avg_reward_error'] = float(np.mean(np.abs(reward_diff)))
                stats['reward_correlation'] = float(np.corrcoef(
                    self.reward_history, self.true_rewards
                )[0, 1]) if len(self.reward_history) > 1 else 0.0
        
        return stats
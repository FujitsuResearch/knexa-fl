"""
Heterogeneity-Greedy Engine

Greedily selects pairs with highest data heterogeneity (JS divergence).
Tests the core hypothesis that heterogeneous pairings are beneficial.

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
from profile_builders import EnhancedProfileBuilder

logger = logging.getLogger(__name__)


class HeterogeneityGreedyEngine(BasePairingEngine):
    """Greedy pairing based on data heterogeneity"""
    
    def __init__(self):
        """Initialize heterogeneity-greedy engine"""
        super().__init__("Hetero-Greedy")
        self.profile_builder = EnhancedProfileBuilder()
        self.selected_heterogeneity_scores = []
        
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select pairs greedily based on highest heterogeneity
        """
        n = len(clients)
        candidates = []
        
        # Calculate heterogeneity for all pairs
        for i in range(n):
            for j in range(i + 1, n):
                heterogeneity = self.profile_builder._calculate_data_heterogeneity(
                    clients[i], clients[j]
                )
                candidates.append((heterogeneity, i, j))
        
        # Sort by heterogeneity (descending)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select top k disjoint pairs
        selected_candidates = self._ensure_disjoint_pairs(candidates, k_pairs)
        
        # Convert to output format
        selected_pairs = []
        for heterogeneity, i, j in selected_candidates:
            self.selected_heterogeneity_scores.append(heterogeneity)
            
            # Choose method based on architecture
            if clients[i].model_family == clients[j].model_family:
                method = 'peft'
                alpha = None
                temperature = None
            else:
                method = 'kd'
                # Use fixed good parameters
                alpha = 0.6
                temperature = 1.5
            
            metadata = {
                'method': method,
                'alpha': alpha,
                'temperature': temperature,
                'heterogeneity': float(heterogeneity),
                'round': round_id
            }
            
            selected_pairs.append((i, j, metadata))
            self.pairing_history.append((i, j, round_id))
        
        self.round_count = round_id
        
        # Log selection
        avg_hetero = np.mean([p[2]['heterogeneity'] for p in selected_pairs])
        logger.info(f"Hetero-Greedy selected {len(selected_pairs)} pairs for round {round_id}, "
                   f"avg heterogeneity: {avg_hetero:.3f}")
        
        return selected_pairs
    
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Update reward history (no learning for greedy)
        """
        self.reward_history.append(reward)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including heterogeneity analysis"""
        stats = super().get_statistics()
        
        if self.selected_heterogeneity_scores:
            stats['avg_selected_heterogeneity'] = float(np.mean(self.selected_heterogeneity_scores))
            stats['min_selected_heterogeneity'] = float(np.min(self.selected_heterogeneity_scores))
            stats['max_selected_heterogeneity'] = float(np.max(self.selected_heterogeneity_scores))
            
            # Check if heterogeneity is consistently high
            recent_scores = self.selected_heterogeneity_scores[-10:]
            if len(recent_scores) >= 5:
                stats['recent_avg_heterogeneity'] = float(np.mean(recent_scores))
        
        return stats
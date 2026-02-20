"""
Base Engine for Client Pairing Strategies

Defines the interface for all pairing engines used in simulations.

Author: Inderjeet Singh
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_environment import SyntheticClient

logger = logging.getLogger(__name__)


class BasePairingEngine(ABC):
    """Abstract base class for client pairing strategies"""
    
    def __init__(self, name: str):
        """
        Initialize base engine
        
        Args:
            name: Name of the pairing strategy
        """
        self.name = name
        self.round_count = 0
        self.pairing_history = []
        self.reward_history = []
        
    @abstractmethod
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select k client pairs for collaboration
        
        Args:
            clients: List of all clients
            k_pairs: Number of pairs to select
            round_id: Current round number
            
        Returns:
            List of tuples (client_i_id, client_j_id, metadata)
            where metadata contains pairing-specific information
        """
        pass
    
    @abstractmethod
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Update the engine based on observed reward
        
        Args:
            client_i: First client ID
            client_j: Second client ID
            reward: Observed reward from collaboration
            round_id: Current round number
            context: Additional context (e.g., feature vector)
        """
        pass
    
    def reset(self):
        """Reset engine state for new experiment"""
        self.round_count = 0
        self.pairing_history = []
        self.reward_history = []
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics for analysis"""
        stats = {
            'name': self.name,
            'rounds': self.round_count,
            'total_pairings': len(self.pairing_history),
            'cumulative_reward': sum(self.reward_history) if self.reward_history else 0,
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'reward_variance': np.var(self.reward_history) if self.reward_history else 0
        }
        return stats
    
    def _ensure_disjoint_pairs(self, candidate_pairs: List[Tuple], 
                              k_pairs: int) -> List[Tuple]:
        """
        Ensure selected pairs are disjoint (no client in multiple pairs)
        
        Args:
            candidate_pairs: List of (score, i, j, ...) tuples sorted by score
            k_pairs: Maximum number of pairs to select
            
        Returns:
            List of disjoint pairs
        """
        selected_pairs = []
        used_clients = set()
        
        for pair_info in candidate_pairs:
            # Extract client IDs (assuming they're at positions 1 and 2)
            i, j = pair_info[1], pair_info[2]
            
            if i not in used_clients and j not in used_clients:
                selected_pairs.append(pair_info)
                used_clients.update([i, j])
                
                if len(selected_pairs) >= k_pairs:
                    break
        
        return selected_pairs
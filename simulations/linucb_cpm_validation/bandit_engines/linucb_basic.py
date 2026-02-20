"""
LinUCB Engine with Basic Profiles

Uses the actual LinUCB implementation from src.bandit with 16D profiles.

Author: Inderjeet Singh
"""

import sys
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import logging

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Use fixed version for simulations
from .linucb_fixed import LinUCBFixed as LinUCB
# Default parameters
LINUCB_LAMBDA = 1.0
LINUCB_BETA0 = 1.0

from .base_engine import BasePairingEngine
from synthetic_environment import SyntheticClient
from profile_builders import BasicProfileBuilder

logger = logging.getLogger(__name__)


class LinUCBBasicEngine(BasePairingEngine):
    """LinUCB engine using 16D basic profiles"""
    
    def __init__(self, 
                 d: int = 32,  # 16D + 16D concatenated profiles
                 lambda_reg: float = None,
                 beta0: float = None):
        """
        Initialize LinUCB engine with basic profiles
        
        Args:
            d: Context dimension (32 for concatenated 16D profiles)
            lambda_reg: Regularization parameter (uses LINUCB_LAMBDA if None)
            beta0: Exploration parameter (uses LINUCB_BETA0 if None)
        """
        super().__init__("LinUCB-Basic")
        
        # Use actual parameters from globals if not specified
        self.lambda_reg = lambda_reg if lambda_reg is not None else LINUCB_LAMBDA
        self.beta0 = beta0 if beta0 is not None else LINUCB_BETA0
        
        # Initialize LinUCB
        self.bandit = LinUCB(d=d, lambda_reg=self.lambda_reg, beta0=self.beta0)
            
        self.profile_builder = BasicProfileBuilder()
        self.context_cache = {}
        
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select pairs using LinUCB with basic profiles
        """
        n = len(clients)
        
        # Build profiles for all clients
        profiles = []
        for client in clients:
            profile = self.profile_builder.build_profile(client)
            profiles.append(profile)
        
        # Use the actual LinUCB choose_pairs method
        # Note: The actual method returns (i, j, alpha, T) tuples
        pairs_with_params = self.bandit.choose_pairs(profiles, k_pairs, round_id)
        
        # Convert to our format and cache contexts
        selected_pairs = []
        for i, j, alpha, T in pairs_with_params:
            # Build and cache pairwise context
            context = self.profile_builder.build_pairwise_context(clients[i], clients[j])
            context_key = (i, j, round_id)
            self.context_cache[context_key] = context
            
            metadata = {
                'alpha': alpha,      # KD alpha parameter
                'temperature': T,    # KD temperature
                'method': 'kd',      # Knowledge distillation
                'round': round_id
            }
            
            selected_pairs.append((i, j, metadata))
            self.pairing_history.append((i, j, round_id))
        
        self.round_count = round_id
        logger.info(f"LinUCB-Basic selected {len(selected_pairs)} pairs for round {round_id}")
        
        return selected_pairs
    
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Update LinUCB with observed reward
        """
        # Retrieve cached context or use provided one
        context_key = (client_i, client_j, round_id)
        if context is None:
            context = self.context_cache.get(context_key)
            if context is None:
                logger.warning(f"No context found for pair ({client_i}, {client_j}) in round {round_id}")
                return
        
        # Update using actual LinUCB update method
        self.bandit.update(context, reward, round_id)
        
        # Track reward
        self.reward_history.append(reward)
        
        # Clean up old cache entries
        if len(self.context_cache) > 1000:
            # Keep only recent contexts
            keys_to_remove = [k for k in self.context_cache.keys() if k[2] < round_id - 10]
            for k in keys_to_remove:
                del self.context_cache[k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extended statistics including LinUCB state"""
        stats = super().get_statistics()
        
        # Add LinUCB-specific statistics
        theta = self.bandit._theta().numpy()
        stats['theta_norm'] = float(np.linalg.norm(theta))
        stats['theta_mean'] = float(np.mean(theta))
        stats['theta_std'] = float(np.std(theta))
        
        # Exploration vs exploitation analysis
        if hasattr(self, 'ucb_scores'):
            stats['avg_ucb_score'] = float(np.mean(self.ucb_scores))
            stats['ucb_score_std'] = float(np.std(self.ucb_scores))
        
        return stats
    
    def get_ucb_score(self, client_i: SyntheticClient, client_j: SyntheticClient, 
                     round_id: int) -> float:
        """
        Get UCB score for a specific pair (useful for analysis)
        """
        context = self.profile_builder.build_pairwise_context(client_i, client_j)
        return self.bandit.get_ucb_score(context, round_id)
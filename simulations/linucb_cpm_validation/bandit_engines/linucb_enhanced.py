"""
LinUCB Engine with Enhanced Profiles

Uses LinUCB with the sophisticated enhanced context vectors from
enhanced_context_vector.py, including heterogeneity features.

Author: Inderjeet Singh
"""

import sys
import os
import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Use fixed version for simulations
from .linucb_fixed import LinUCBFixed as LinUCB
# Default parameters
LINUCB_LAMBDA = 1.0
LINUCB_BETA0 = 1.0

from .base_engine import BasePairingEngine
from synthetic_environment import SyntheticClient
from profile_builders import EnhancedProfileBuilder

logger = logging.getLogger(__name__)


class LinUCBEnhancedEngine(BasePairingEngine):
    """LinUCB engine using enhanced profiles with heterogeneity features"""
    
    def __init__(self, 
                 context_dim: int = 32,
                 lambda_reg: float = None,
                 beta0: float = None):
        """
        Initialize LinUCB engine with enhanced profiles
        
        Args:
            context_dim: Individual profile dimension (32 default)
            lambda_reg: Regularization parameter
            beta0: Exploration parameter
        """
        super().__init__("LinUCB-Enhanced")
        
        self.context_dim = context_dim
        # Enhanced pairwise context: 2 * context_dim + 8 pairwise features
        self.pairwise_dim = 2 * context_dim + 8
        
        # Use parameters from globals if not specified
        self.lambda_reg = lambda_reg if lambda_reg is not None else LINUCB_LAMBDA
        self.beta0 = beta0 if beta0 is not None else LINUCB_BETA0
        
        # Initialize LinUCB with enhanced dimension
        self.bandit = LinUCB(d=self.pairwise_dim, lambda_reg=self.lambda_reg, beta0=self.beta0)
            
        self.profile_builder = EnhancedProfileBuilder(context_dim=context_dim)
        self.context_cache = {}
        self.heterogeneity_scores = []  # Track for analysis
        
    def select_pairs(self, clients: List[SyntheticClient], 
                    k_pairs: int,
                    round_id: int) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Select pairs using enhanced profiles with heterogeneity awareness
        """
        n = len(clients)
        candidates = []
        
        # Generate all possible pairs with UCB scores
        for i in range(n):
            for j in range(i + 1, n):
                # Build enhanced pairwise context
                context = self.profile_builder.build_pairwise_context(
                    clients[i], clients[j], round_id
                )
                
                # Get UCB score
                ucb_score = self.bandit.get_ucb_score(context, round_id)
                
                # Calculate heterogeneity for this pair
                heterogeneity = self.profile_builder._calculate_data_heterogeneity(
                    clients[i], clients[j]
                )
                
                candidates.append((ucb_score, i, j, context, heterogeneity))
        
        # Sort by UCB score
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Select top k disjoint pairs
        selected_candidates = self._ensure_disjoint_pairs(candidates, k_pairs)
        
        # Convert to output format
        selected_pairs = []
        for ucb_score, i, j, context, heterogeneity in selected_candidates:
            # Cache context
            context_key = (i, j, round_id)
            self.context_cache[context_key] = context
            
            # Track heterogeneity
            self.heterogeneity_scores.append(heterogeneity)
            
            # Determine exchange type based on architecture compatibility
            if clients[i].model_family == clients[j].model_family:
                method = 'peft'  # Can do PEFT exchange
                alpha = None
                temperature = None
            else:
                method = 'kd'  # Knowledge distillation
                # Use round-dependent parameters like in original
                alpha = 0.5 + (round_id % 3) * 0.1  # 0.5, 0.6, or 0.7
                temperature = 1.0 + (round_id % 3) * 0.5  # 1.0, 1.5, or 2.0
            
            metadata = {
                'method': method,
                'alpha': alpha,
                'temperature': temperature,
                'ucb_score': float(ucb_score),
                'heterogeneity': float(heterogeneity),
                'round': round_id
            }
            
            selected_pairs.append((i, j, metadata))
            self.pairing_history.append((i, j, round_id))
        
        self.round_count = round_id
        
        # Log selection statistics
        avg_hetero = np.mean([p[2]['heterogeneity'] for p in selected_pairs])
        logger.info(f"LinUCB-Enhanced selected {len(selected_pairs)} pairs for round {round_id}, "
                   f"avg heterogeneity: {avg_hetero:.3f}")
        
        return selected_pairs
    
    def update(self, client_i: int, client_j: int, 
              reward: float, round_id: int, 
              context: Any = None):
        """
        Update LinUCB with observed reward
        """
        # Retrieve cached context
        context_key = (client_i, client_j, round_id)
        if context is None:
            context = self.context_cache.get(context_key)
            if context is None:
                logger.warning(f"No context found for pair ({client_i}, {client_j}) in round {round_id}")
                return
        
        # Update bandit
        self.bandit.update(context, reward, round_id)
        
        # Track reward
        self.reward_history.append(reward)
        
        # Clean cache periodically
        if len(self.context_cache) > 1000:
            keys_to_remove = [k for k in self.context_cache.keys() if k[2] < round_id - 10]
            for k in keys_to_remove:
                del self.context_cache[k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extended statistics including heterogeneity analysis"""
        stats = super().get_statistics()
        
        # LinUCB state
        theta = self.bandit._theta().numpy()
        stats['theta_norm'] = float(np.linalg.norm(theta))
        
        # Heterogeneity statistics
        if self.heterogeneity_scores:
            stats['avg_heterogeneity'] = float(np.mean(self.heterogeneity_scores))
            stats['min_heterogeneity'] = float(np.min(self.heterogeneity_scores))
            stats['max_heterogeneity'] = float(np.max(self.heterogeneity_scores))
            stats['heterogeneity_std'] = float(np.std(self.heterogeneity_scores))
            
            # Heterogeneity utilization (how well we're exploiting diversity)
            recent_hetero = self.heterogeneity_scores[-10:]  # Last 10 pairings
            if len(recent_hetero) >= 5:
                stats['heterogeneity_trend'] = float(np.polyfit(range(len(recent_hetero)), 
                                                               recent_hetero, 1)[0])
        
        # Feature importance (simplified)
        if theta.shape[0] == self.pairwise_dim:
            # Analyze which features have highest weights
            feature_groups = {
                'performance': theta[0:8].mean(),
                'data_characteristics': theta[8:17].mean(),
                'model_characteristics': theta[17:21].mean(),
                'trends': theta[21:25].mean(),
                'pairwise_features': theta[-8:].mean()
            }
            stats['feature_importance'] = {k: float(abs(v)) for k, v in feature_groups.items()}
        
        return stats
    
    def get_heterogeneity_matrix(self, clients: List[SyntheticClient]) -> np.ndarray:
        """Get heterogeneity matrix for analysis"""
        return self.profile_builder.get_heterogeneity_matrix(clients)
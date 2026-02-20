"""
Profile Builders for KNEXA-FL Simulations

Implements both basic (16D) and enhanced (32D+) profile generation,
matching the actual KNEXA-FL implementation.

Author: Inderjeet Singh
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy.spatial.distance import jensenshannon

from synthetic_environment import SyntheticClient, PROBLEM_TYPES, DIFFICULTY_LEVELS

logger = logging.getLogger(__name__)


class BasicProfileBuilder:
    """
    Builds 16-dimensional profiles matching client.py:make_profile()
    """
    
    def __init__(self):
        self.profile_dim = 16
        
    def build_profile(self, client: SyntheticClient) -> np.ndarray:
        """
        Build 16D profile vector for a client
        
        Profile structure:
        [0-2]: Performance metrics (last_perf, last_codebleu, sier_avg)
        [3-6]: One-hot encoded model family (4 bits)
        [7-12]: Model & communication characteristics
        [13-15]: Padding zeros
        """
        # One-hot encode model family
        fam_bits = [0, 0, 0, 0]
        family_map = {'pythia': 0, 'opt': 1, 'gpt2': 2, 't5': 3}
        fam_idx = family_map.get(client.model_family, client.client_id % 4)
        fam_bits[fam_idx] = 1
        
        profile = np.array([
            client.local_pass_at_1,      # last_perf
            client.local_codebleu,       # last_codebleu
            client.sier_avg,             # sier_avg
            *fam_bits,                   # one-hot family (4 dims)
            client.model_size_mb / 1000, # params_m / 1000
            client.trust_score,          # trust
            client.historical_delta,     # historical_delta
            client.comm_kb,              # comm_kb
            client.pre_post_diff,        # pre_post_diff
            client.effective_bandwidth,  # effective_bandwidth
            0, 0, 0                      # padding zeros
        ], dtype=np.float32)
        
        return profile
    
    def build_pairwise_context(self, client_i: SyntheticClient, 
                              client_j: SyntheticClient) -> np.ndarray:
        """Build 32D pairwise context by concatenating two profiles"""
        profile_i = self.build_profile(client_i)
        profile_j = self.build_profile(client_j)
        return np.concatenate([profile_i, profile_j])


class EnhancedProfileBuilder:
    """
    Builds enhanced profiles matching enhanced_context_vector.py
    Includes sophisticated heterogeneity calculations and pairwise features
    """
    
    def __init__(self, context_dim: int = 32):
        self.context_dim = context_dim
        self.profile_cache = {}
        
    def build_profile(self, client: SyntheticClient) -> np.ndarray:
        """
        Build enhanced profile vector with comprehensive features
        
        Matches ClientProfile.to_context_vector() from enhanced_context_vector.py
        """
        vector = np.zeros(self.context_dim, dtype=np.float32)
        
        # Performance metrics (8 dimensions)
        vector[0] = client.local_pass_at_1
        vector[1] = client.local_pass_at_1  # transfer_performance (simulated as local)
        vector[2] = client.global_pass_at_1
        vector[3] = client.global_pass_at_1 - client.local_pass_at_1  # Generalization gap
        vector[4] = client.trust_score
        vector[5] = client.learning_rate
        vector[6] = client.collaboration_quality
        vector[7] = client.communication_efficiency
        
        # Data characteristics (9 dimensions)
        for i, dtype in enumerate(PROBLEM_TYPES):
            vector[8 + i] = client.data_distribution.get(dtype, 0.0)
        
        for i, difficulty in enumerate(DIFFICULTY_LEVELS):
            vector[13 + i] = client.difficulty_distribution.get(difficulty, 0.0)
        
        # Specialization score (how concentrated the distribution is)
        data_entropy = -sum(p * np.log(p + 1e-8) for p in client.data_distribution.values() if p > 0)
        max_entropy = -np.log(1.0 / len(PROBLEM_TYPES))
        specialization_score = 1.0 - (data_entropy / max_entropy)
        vector[16] = specialization_score
        
        # Model characteristics (4 dimensions)
        vector[17] = client.model_size_mb / 1000.0  # Normalized
        vector[18] = 1.0 if 'gpt' in client.model_family.lower() else 0.0
        vector[19] = 1.0 if 'code' in client.model_family.lower() else 0.0
        vector[20] = 1.0 if client.model_family in ['t5', 'mt5'] else 0.0  # encoder-decoder
        
        # Trend indicators (4 dimensions)
        if len(client.performance_history) >= 3:
            recent_trend = np.polyfit(range(3), client.performance_history[-3:], 1)[0]
            vector[21] = 1.0 if recent_trend > 0.02 else 0.0  # improving
            vector[22] = 1.0 if -0.02 <= recent_trend <= 0.02 else 0.0  # stable
            vector[23] = 1.0 if recent_trend < -0.02 else 0.0  # declining
        else:
            vector[22] = 1.0  # Default to stable
            
        vector[24] = len(client.collaboration_history) / 10.0  # Normalized collaboration count
        
        # Derived metrics (remaining dimensions)
        for i in range(25, self.context_dim):
            if i == 25:
                vector[i] = client.local_pass_at_1 * client.trust_score  # Reliable local performance
            elif i == 26:
                vector[i] = client.local_pass_at_1 * client.collaboration_quality  # Transfer effectiveness
            elif i == 27:
                vector[i] = client.global_pass_at_1 * client.communication_efficiency  # Efficient global performance
            else:
                vector[i] = np.random.normal(0, 0.01)  # Small noise for regularization
        
        return vector
    
    def build_pairwise_context(self, client_i: SyntheticClient, 
                              client_j: SyntheticClient,
                              round_id: int = 0) -> np.ndarray:
        """
        Build comprehensive pairwise context with heterogeneity features
        
        Matches EnhancedContextBuilder.build_pairwise_context()
        """
        # Get individual context vectors
        context_i = self.build_profile(client_i)
        context_j = self.build_profile(client_j)
        
        # Create pairwise features (8 dimensions)
        pairwise_features = np.zeros(8, dtype=np.float32)
        
        # Data heterogeneity (core hypothesis)
        pairwise_features[0] = self._calculate_data_heterogeneity(client_i, client_j)
        pairwise_features[1] = self._calculate_performance_complementarity(client_i, client_j)
        pairwise_features[2] = self._calculate_trust_compatibility(client_i, client_j)
        pairwise_features[3] = self._calculate_collaboration_potential(client_i, client_j)
        
        # Model heterogeneity
        pairwise_features[4] = self._calculate_model_diversity(client_i, client_j)
        pairwise_features[5] = self._calculate_learning_rate_compatibility(client_i, client_j)
        
        # Temporal features
        pairwise_features[6] = round_id / 25.0  # Normalized round progress
        pairwise_features[7] = self._calculate_previous_collaboration_score(client_i, client_j)
        
        # Concatenate all features
        pairwise_context = np.concatenate([context_i, context_j, pairwise_features])
        
        return pairwise_context
    
    def _calculate_data_heterogeneity(self, client_i: SyntheticClient, 
                                     client_j: SyntheticClient) -> float:
        """
        Calculate data heterogeneity using Jensen-Shannon divergence
        Higher score = more heterogeneous = better for collaboration
        """
        # Problem type diversity
        types_i = np.array([client_i.data_distribution.get(t, 0) for t in PROBLEM_TYPES])
        types_j = np.array([client_j.data_distribution.get(t, 0) for t in PROBLEM_TYPES])
        
        # Normalize
        types_i = types_i / (types_i.sum() + 1e-8)
        types_j = types_j / (types_j.sum() + 1e-8)
        
        # JS divergence
        js_divergence = jensenshannon(types_i, types_j)
        
        # Difficulty diversity
        diff_i = np.array([client_i.difficulty_distribution.get(d, 0) for d in DIFFICULTY_LEVELS])
        diff_j = np.array([client_j.difficulty_distribution.get(d, 0) for d in DIFFICULTY_LEVELS])
        
        diff_i = diff_i / (diff_i.sum() + 1e-8)
        diff_j = diff_j / (diff_j.sum() + 1e-8)
        
        difficulty_divergence = jensenshannon(diff_i, diff_j)
        
        # Combined heterogeneity
        heterogeneity_score = 0.7 * js_divergence + 0.3 * difficulty_divergence
        
        return float(heterogeneity_score)
    
    def _calculate_performance_complementarity(self, client_i: SyntheticClient,
                                             client_j: SyntheticClient) -> float:
        """Calculate how well clients complement each other's performance"""
        # Local performance gap
        local_gap = abs(client_i.local_pass_at_1 - client_j.local_pass_at_1)
        
        # Global performance gap
        global_gap = abs(client_i.global_pass_at_1 - client_j.global_pass_at_1)
        
        # Specialization complementarity
        spec_i = self._calculate_specialization_score(client_i)
        spec_j = self._calculate_specialization_score(client_j)
        specialization_complement = abs(spec_i - spec_j)
        
        # Moderate gaps are good, extreme gaps are bad
        complementarity = np.exp(-2 * local_gap) + np.exp(-2 * global_gap) + specialization_complement
        
        return float(complementarity)
    
    def _calculate_trust_compatibility(self, client_i: SyntheticClient,
                                     client_j: SyntheticClient) -> float:
        """Calculate trust compatibility for reliable collaboration"""
        min_trust = min(client_i.trust_score, client_j.trust_score)
        avg_trust = (client_i.trust_score + client_j.trust_score) / 2
        
        # Prefer pairs where both have high trust
        trust_compatibility = 0.6 * min_trust + 0.4 * avg_trust
        
        return float(trust_compatibility)
    
    def _calculate_collaboration_potential(self, client_i: SyntheticClient,
                                         client_j: SyntheticClient) -> float:
        """Calculate potential for effective collaboration"""
        # Collaboration quality
        collab_quality = (client_i.collaboration_quality + client_j.collaboration_quality) / 2
        
        # Performance trend bonus
        trend_bonus = 0.0
        if len(client_i.performance_history) >= 3 and len(client_j.performance_history) >= 3:
            trend_i = np.polyfit(range(3), client_i.performance_history[-3:], 1)[0]
            trend_j = np.polyfit(range(3), client_j.performance_history[-3:], 1)[0]
            
            if trend_i >= -0.02 and trend_j >= -0.02:  # Both stable or improving
                trend_bonus = 0.3
        
        # Communication efficiency
        comm_efficiency = (client_i.communication_efficiency + client_j.communication_efficiency) / 2
        
        collaboration_potential = collab_quality + trend_bonus + 0.2 * comm_efficiency
        
        return float(collaboration_potential)
    
    def _calculate_model_diversity(self, client_i: SyntheticClient,
                                  client_j: SyntheticClient) -> float:
        """Calculate model architecture diversity bonus"""
        # Different families = good
        family_diversity = 1.0 if client_i.model_family != client_j.model_family else 0.3
        
        # Different sizes complement each other
        size_ratio = min(client_i.model_size_mb, client_j.model_size_mb) / \
                    max(client_i.model_size_mb, client_j.model_size_mb)
        size_diversity = 1.0 - size_ratio
        
        model_diversity = 0.7 * family_diversity + 0.3 * size_diversity
        
        return float(model_diversity)
    
    def _calculate_learning_rate_compatibility(self, client_i: SyntheticClient,
                                             client_j: SyntheticClient) -> float:
        """Calculate learning rate compatibility"""
        avg_learning_rate = (client_i.learning_rate + client_j.learning_rate) / 2
        
        # Bonus if both are learning
        both_learning = 1.0 if client_i.learning_rate > 0 and client_j.learning_rate > 0 else 0.5
        
        learning_compatibility = both_learning * max(0, avg_learning_rate)
        
        return float(learning_compatibility)
    
    def _calculate_previous_collaboration_score(self, client_i: SyntheticClient,
                                              client_j: SyntheticClient) -> float:
        """Calculate score based on previous collaborations"""
        # Penalty for recent collaborations (encourage exploration)
        if client_j.client_id in client_i.collaboration_history[-3:]:
            return 0.2  # Recent collaboration
        elif client_j.client_id in client_i.collaboration_history:
            return 0.4  # Previous collaboration
        else:
            return 0.8  # New collaboration (bonus)
    
    def _calculate_specialization_score(self, client: SyntheticClient) -> float:
        """Calculate how specialized vs generalized a client is"""
        # Using entropy of data distribution
        data_probs = list(client.data_distribution.values())
        entropy = -sum(p * np.log(p + 1e-8) for p in data_probs if p > 0)
        max_entropy = -np.log(1.0 / len(PROBLEM_TYPES))
        
        # Normalized: 0 = uniform (generalized), 1 = concentrated (specialized)
        return 1.0 - (entropy / max_entropy)
    
    def get_heterogeneity_matrix(self, clients: List[SyntheticClient]) -> np.ndarray:
        """
        Generate heterogeneity matrix for a set of clients
        Useful for analysis and visualization
        """
        n = len(clients)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i, j] = self._calculate_data_heterogeneity(clients[i], clients[j])
        
        return matrix


if __name__ == "__main__":
    # Test profile builders
    from synthetic_environment import SyntheticEnvironment
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Profile Builders")
    print("=" * 50)
    
    # Generate test clients
    env = SyntheticEnvironment(num_clients=4, heterogeneity_level='high')
    clients = env.generate_clients()
    
    # Test basic profile builder
    print("\nBasic Profile Builder:")
    basic_builder = BasicProfileBuilder()
    for i in range(2):
        profile = basic_builder.build_profile(clients[i])
        print(f"Client {i} profile shape: {profile.shape}")
        print(f"  Performance: {profile[0]:.3f}, Trust: {profile[7]:.3f}")
    
    pairwise_basic = basic_builder.build_pairwise_context(clients[0], clients[1])
    print(f"Pairwise context shape: {pairwise_basic.shape}")
    
    # Test enhanced profile builder
    print("\nEnhanced Profile Builder:")
    enhanced_builder = EnhancedProfileBuilder()
    for i in range(2):
        profile = enhanced_builder.build_profile(clients[i])
        print(f"Client {i} enhanced profile shape: {profile.shape}")
    
    pairwise_enhanced = enhanced_builder.build_pairwise_context(clients[0], clients[1], round_id=5)
    print(f"Enhanced pairwise context shape: {pairwise_enhanced.shape}")
    
    # Test heterogeneity calculation
    hetero_score = enhanced_builder._calculate_data_heterogeneity(clients[0], clients[1])
    print(f"\nHeterogeneity score between client 0 and 1: {hetero_score:.3f}")
    
    # Generate heterogeneity matrix
    hetero_matrix = enhanced_builder.get_heterogeneity_matrix(clients)
    print(f"\nHeterogeneity matrix:\n{hetero_matrix}")
"""
Reward Models for KNEXA-FL Simulations

Implements ground-truth reward functions based on the paper's formulation
and the actual KNEXA-FL implementation.

Author: Inderjeet Singh
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from scipy.spatial.distance import jensenshannon

from synthetic_environment import SyntheticClient, PROBLEM_TYPES, DIFFICULTY_LEVELS
from profile_builders import EnhancedProfileBuilder

logger = logging.getLogger(__name__)


class GroundTruthReward:
    """
    Ground-truth reward model for evaluating client pairings
    Based on KNEXA-FL paper Equation 7 and enhanced_context_vector.py
    """
    
    def __init__(self, 
                 gamma: float = 10.0,      # Performance improvement weight (increased)
                 delta: float = 0.00001,   # Communication cost penalty (reduced)
                 noise_std: float = 0.02):  # Reward noise (reduced)
        """
        Initialize reward model
        
        Args:
            gamma: Weight for performance improvement component
            delta: Penalty coefficient for communication cost (KB)
            noise_std: Standard deviation of reward noise
        """
        self.gamma = gamma
        self.delta = delta
        self.noise_std = noise_std
        self.enhanced_builder = EnhancedProfileBuilder()
        
        # Component weights for interpretability
        self.component_weights = {
            'data_heterogeneity': 0.5,      # Core hypothesis
            'performance_complementarity': 0.2,
            'trust_compatibility': 0.2,
            'communication_efficiency': 0.1
        }
        
    def compute_reward(self, 
                      client_i: SyntheticClient, 
                      client_j: SyntheticClient,
                      exchange_type: str = 'kd',
                      round_id: int = 0) -> Tuple[float, Dict[str, float]]:
        """
        Compute ground-truth reward for a client pairing
        
        Args:
            client_i: First client
            client_j: Second client
            exchange_type: 'kd' for knowledge distillation, 'peft' for PEFT exchange
            round_id: Current round number
            
        Returns:
            (reward, components): Total reward and component breakdown
        """
        components = {}
        
        # 1. Data Heterogeneity Component (Core Hypothesis)
        data_hetero = self._compute_data_heterogeneity(client_i, client_j)
        components['data_heterogeneity'] = data_hetero
        
        # 2. Performance Complementarity
        perf_comp = self._compute_performance_complementarity(client_i, client_j)
        components['performance_complementarity'] = perf_comp
        
        # 3. Trust Compatibility
        trust_compat = self._compute_trust_compatibility(client_i, client_j)
        components['trust_compatibility'] = trust_compat
        
        # 4. Communication Efficiency
        comm_eff = self._compute_communication_efficiency(client_i, client_j)
        components['communication_efficiency'] = comm_eff
        
        # Weighted combination
        base_reward = sum(self.component_weights[k] * v for k, v in components.items())
        
        # 5. Expected Performance Improvement
        expected_improvement = self._estimate_performance_improvement(
            client_i, client_j, base_reward, exchange_type
        )
        components['expected_improvement'] = expected_improvement
        
        # 6. Communication Cost
        comm_kb = self._estimate_communication_cost(client_i, client_j, exchange_type)
        components['communication_kb'] = comm_kb
        
        # Final reward (matching paper Eq. 7)
        reward = self.gamma * expected_improvement - self.delta * comm_kb
        
        # Add realistic noise
        noise = np.random.normal(0, self.noise_std)
        reward_with_noise = np.clip(reward + noise, 0, 1)
        
        components['base_reward'] = base_reward
        components['final_reward'] = reward_with_noise
        components['noise'] = noise
        
        return reward_with_noise, components
    
    def _compute_data_heterogeneity(self, client_i: SyntheticClient, 
                                   client_j: SyntheticClient) -> float:
        """
        Compute data heterogeneity using Jensen-Shannon divergence
        Matches EnhancedContextBuilder._calculate_data_heterogeneity()
        """
        # Problem type diversity
        types_i = np.array([client_i.data_distribution.get(t, 0) for t in PROBLEM_TYPES])
        types_j = np.array([client_j.data_distribution.get(t, 0) for t in PROBLEM_TYPES])
        
        types_i = types_i / (types_i.sum() + 1e-8)
        types_j = types_j / (types_j.sum() + 1e-8)
        
        js_divergence = jensenshannon(types_i, types_j)
        
        # Difficulty diversity
        diff_i = np.array([client_i.difficulty_distribution.get(d, 0) for d in DIFFICULTY_LEVELS])
        diff_j = np.array([client_j.difficulty_distribution.get(d, 0) for d in DIFFICULTY_LEVELS])
        
        diff_i = diff_i / (diff_i.sum() + 1e-8)
        diff_j = diff_j / (diff_j.sum() + 1e-8)
        
        difficulty_divergence = jensenshannon(diff_i, diff_j)
        
        # Combined heterogeneity (higher is better)
        heterogeneity_score = 0.7 * js_divergence + 0.3 * difficulty_divergence
        
        return float(heterogeneity_score)
    
    def _compute_performance_complementarity(self, client_i: SyntheticClient,
                                           client_j: SyntheticClient) -> float:
        """
        Compute how well clients complement each other
        """
        # Performance gap analysis
        local_gap = abs(client_i.local_pass_at_1 - client_j.local_pass_at_1)
        global_gap = abs(client_i.global_pass_at_1 - client_j.global_pass_at_1)
        
        # One strong + one weak = good potential for knowledge transfer
        # Both strong or both weak = limited benefit
        optimal_gap = 0.3  # Optimal performance difference
        
        local_complement = np.exp(-((local_gap - optimal_gap) ** 2) / 0.1)
        global_complement = np.exp(-((global_gap - optimal_gap) ** 2) / 0.1)
        
        # Model diversity bonus
        model_diversity = 1.0 if client_i.model_family != client_j.model_family else 0.5
        
        complementarity = 0.4 * local_complement + 0.4 * global_complement + 0.2 * model_diversity
        
        return float(complementarity)
    
    def _compute_trust_compatibility(self, client_i: SyntheticClient,
                                    client_j: SyntheticClient) -> float:
        """
        Compute trust-based compatibility
        """
        # Both clients should be trustworthy
        min_trust = min(client_i.trust_score, client_j.trust_score)
        avg_trust = (client_i.trust_score + client_j.trust_score) / 2
        
        # Historical collaboration quality
        hist_collab_quality = (client_i.collaboration_quality + client_j.collaboration_quality) / 2
        
        # Penalize if they collaborated too recently (encourage exploration)
        recency_penalty = 0
        if client_j.client_id in client_i.collaboration_history[-3:]:
            recency_penalty = 0.3
        
        trust_compatibility = 0.5 * min_trust + 0.3 * avg_trust + 0.2 * hist_collab_quality - recency_penalty
        
        return float(max(0, trust_compatibility))
    
    def _compute_communication_efficiency(self, client_i: SyntheticClient,
                                         client_j: SyntheticClient) -> float:
        """
        Compute communication efficiency score
        """
        # Average bandwidth capability
        avg_bandwidth = (client_i.effective_bandwidth + client_j.effective_bandwidth) / 2
        
        # Communication efficiency from past collaborations
        avg_comm_efficiency = (client_i.communication_efficiency + client_j.communication_efficiency) / 2
        
        # Combined score
        comm_score = 0.6 * avg_bandwidth + 0.4 * avg_comm_efficiency
        
        return float(comm_score)
    
    def _estimate_performance_improvement(self, client_i: SyntheticClient,
                                        client_j: SyntheticClient,
                                        base_reward: float,
                                        exchange_type: str) -> float:
        """
        Estimate expected performance improvement from collaboration
        """
        # Base improvement proportional to reward and learning capacity
        learning_capacity = 1.0 - client_i.local_pass_at_1  # Room for improvement
        
        # Knowledge gap (what j can teach i)
        knowledge_gap = max(0, client_j.local_pass_at_1 - client_i.local_pass_at_1)
        
        # Transfer efficiency based on exchange type
        if exchange_type == 'peft':
            # PEFT exchange is more direct but requires compatible architectures
            architecture_compat = 1.0 if client_i.model_family == client_j.model_family else 0.3
            transfer_efficiency = 0.8 * architecture_compat
        else:  # knowledge distillation
            # KD works across architectures but with some loss
            transfer_efficiency = 0.6
        
        # Learning rate of the receiver
        receiver_learning_rate = client_i.learning_rate
        
        # Expected improvement - ensure there's always some benefit from collaboration
        # Even similar clients can learn from each other due to data heterogeneity
        base_improvement = base_reward * 0.1  # Base 10% of reward translates to improvement
        
        # Additional improvement from knowledge gap
        gap_improvement = (
            learning_capacity * 
            knowledge_gap * 
            transfer_efficiency * 
            (1 + receiver_learning_rate)
        )
        
        # Symmetric improvement (both clients benefit)
        improvement_j = max(0, client_i.local_pass_at_1 - client_j.local_pass_at_1) * \
                       (1.0 - client_j.local_pass_at_1) * transfer_efficiency * \
                       (1 + client_j.learning_rate)
        
        # Total improvement includes base + gap-based improvements
        total_improvement = base_improvement + 0.3 * (gap_improvement + improvement_j)
        
        return float(np.clip(total_improvement, 0, 0.5))  # Cap at 0.5
    
    def _estimate_communication_cost(self, client_i: SyntheticClient,
                                    client_j: SyntheticClient,
                                    exchange_type: str) -> float:
        """
        Estimate communication cost in KB
        """
        if exchange_type == 'peft':
            # PEFT module size depends on model size
            base_cost_mb = 5.0  # Base 5MB for PEFT exchange
            size_factor = (client_j.model_size_mb / 1000.0)  # Larger models = larger adapters
            comm_kb = base_cost_mb * 1024 * (1 + size_factor * 0.5)  # Reduced size factor impact
        else:  # knowledge distillation
            # KD requires transferring generated text
            base_cost_mb = 2.0  # Base 2MB for text transfer
            comm_kb = base_cost_mb * 1024
        
        # Adjust for bandwidth limitations
        bandwidth_factor = 2.0 - min(client_i.effective_bandwidth, client_j.effective_bandwidth)
        comm_kb *= bandwidth_factor
        
        return float(comm_kb)
    
    def compute_oracle_pairings(self, clients: list, k_pairs: int) -> list:
        """
        Compute oracle (optimal) pairings based on true rewards
        Used as upper bound baseline
        """
        n = len(clients)
        all_rewards = []
        
        # Compute rewards for all possible pairs
        for i in range(n):
            for j in range(i + 1, n):
                reward, _ = self.compute_reward(clients[i], clients[j])
                all_rewards.append((reward, i, j))
        
        # Sort by reward
        all_rewards.sort(reverse=True)
        
        # Select top k disjoint pairs
        selected_pairs = []
        used_clients = set()
        
        for reward, i, j in all_rewards:
            if i not in used_clients and j not in used_clients and len(selected_pairs) < k_pairs:
                selected_pairs.append((i, j, reward))
                used_clients.update([i, j])
        
        return selected_pairs


class PerformanceUpdateModel:
    """
    Models how client performance updates after collaboration
    """
    
    def __init__(self, learning_rate_base: float = 0.05):
        self.learning_rate_base = learning_rate_base
        
    def update_performance(self, client: SyntheticClient, 
                          reward: float,
                          partner: SyntheticClient,
                          exchange_type: str = 'kd') -> Dict[str, float]:
        """
        Update client performance based on collaboration reward
        
        Returns:
            Dictionary of performance deltas
        """
        # Base improvement proportional to reward
        base_improvement = reward * self.learning_rate_base
        
        # Adjust based on current performance (diminishing returns)
        improvement_factor = 1.0 - client.local_pass_at_1
        
        # Knowledge transfer effectiveness
        if exchange_type == 'peft' and client.model_family == partner.model_family:
            transfer_mult = 1.2  # Better transfer with same architecture
        else:
            transfer_mult = 1.0
        
        # Calculate improvements
        pass1_delta = base_improvement * improvement_factor * transfer_mult
        pass5_delta = pass1_delta * 0.8  # Pass@5 improves slightly less
        pass10_delta = pass1_delta * 0.6  # Pass@10 improves even less
        codebleu_delta = pass1_delta * 0.7  # CodeBLEU correlated with pass@1
        
        # Global performance improves more slowly
        global_pass1_delta = pass1_delta * 0.5
        global_pass10_delta = pass10_delta * 0.5
        global_codebleu_delta = codebleu_delta * 0.5
        
        deltas = {
            'local_pass_at_1': pass1_delta,
            'local_pass_at_5': pass5_delta,
            'local_pass_at_10': pass10_delta,
            'local_codebleu': codebleu_delta,
            'global_pass_at_1': global_pass1_delta,
            'global_pass_at_10': global_pass10_delta,
            'global_codebleu': global_codebleu_delta
        }
        
        return deltas


if __name__ == "__main__":
    # Test reward models
    from synthetic_environment import SyntheticEnvironment
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Reward Models")
    print("=" * 50)
    
    # Generate test clients
    env = SyntheticEnvironment(num_clients=4, heterogeneity_level='high')
    clients = env.generate_clients()
    
    # Test ground truth reward
    reward_model = GroundTruthReward()
    
    print("\nPairwise Rewards:")
    for i in range(len(clients)):
        for j in range(i + 1, len(clients)):
            reward, components = reward_model.compute_reward(clients[i], clients[j])
            print(f"\nClient {i} <-> Client {j}:")
            print(f"  Total Reward: {reward:.3f}")
            print(f"  Data Heterogeneity: {components['data_heterogeneity']:.3f}")
            print(f"  Performance Complementarity: {components['performance_complementarity']:.3f}")
            print(f"  Trust Compatibility: {components['trust_compatibility']:.3f}")
    
    # Test oracle pairings
    oracle_pairs = reward_model.compute_oracle_pairings(clients, k_pairs=2)
    print("\nOracle Pairings:")
    for i, j, reward in oracle_pairs:
        print(f"  Pair ({i}, {j}) with reward {reward:.3f}")
    
    # Test performance update
    update_model = PerformanceUpdateModel()
    deltas = update_model.update_performance(clients[0], 0.8, clients[1])
    print(f"\nPerformance Updates for Client 0:")
    for metric, delta in deltas.items():
        print(f"  {metric}: +{delta:.4f}")
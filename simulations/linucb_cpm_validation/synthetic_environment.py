"""
Synthetic Environment for KNEXA-FL Simulations

Generates realistic synthetic clients with heterogeneous data distributions,
model architectures, and performance characteristics based on the actual
KNEXA-FL implementation.

Author: Inderjeet Singh
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Import actual model registry from codebase
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.globals import LLM_REGISTRY

# Problem types and difficulty levels from enhanced_context_vector.py
PROBLEM_TYPES = ['algorithms', 'data_structures', 'string_processing', 'mathematics', 'recursion']
DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

# Model families extracted from LLM_REGISTRY
MODEL_FAMILIES = ['pythia', 'opt', 'gpt2', 't5', 'codegen', 'cerebras', 'mt5', 'gpt-neo']


@dataclass
class SyntheticClient:
    """Represents a synthetic client with all necessary attributes"""
    client_id: int
    model_family: str
    model_size_mb: float
    
    # Data characteristics
    data_distribution: Dict[str, float] = field(default_factory=dict)
    difficulty_distribution: Dict[str, float] = field(default_factory=dict)
    num_samples: int = 1000
    
    # Performance metrics
    local_pass_at_1: float = 0.0
    local_pass_at_5: float = 0.0
    local_pass_at_10: float = 0.0
    local_codebleu: float = 0.0
    global_pass_at_1: float = 0.0
    global_pass_at_10: float = 0.0
    global_codebleu: float = 0.0
    
    # Dynamic characteristics
    trust_score: float = 0.8
    collaboration_quality: float = 0.5
    communication_efficiency: float = 0.8
    learning_rate: float = 0.01
    
    # Security metrics
    sier_avg: float = 0.0  # Sensitive Information Exposure Rate
    
    # Communication metrics
    effective_bandwidth: float = 0.5
    historical_delta: float = 0.0
    comm_kb: float = 0.0
    pre_post_diff: float = 0.0
    
    # Tracking
    performance_history: List[float] = field(default_factory=list)
    collaboration_history: List[int] = field(default_factory=list)
    round_participated: List[int] = field(default_factory=list)


class SyntheticEnvironment:
    """Generates and manages synthetic clients for simulation"""
    
    def __init__(self, 
                 num_clients: int,
                 heterogeneity_level: str = 'high',
                 seed: int = 42):
        """
        Initialize synthetic environment
        
        Args:
            num_clients: Number of clients to generate
            heterogeneity_level: 'low', 'medium', or 'high'
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.heterogeneity_level = heterogeneity_level
        self.seed = seed
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)
        
        # Heterogeneity parameters (Dirichlet alpha)
        self.alpha_map = {
            'low': 2.0,     # More uniform distributions
            'medium': 0.7,  # Moderate heterogeneity
            'high': 0.3     # High heterogeneity
        }
        self.alpha = self.alpha_map[heterogeneity_level]
        
        # Performance parameters based on model size
        self.perf_params = {
            '70M-160M': {'mean': 0.25, 'std': 0.05},
            '250M-410M': {'mean': 0.35, 'std': 0.05},
            '1B-1.5B': {'mean': 0.45, 'std': 0.05},
            '3B+': {'mean': 0.55, 'std': 0.05}
        }
        
        self.clients = []
        
    def generate_clients(self) -> List[SyntheticClient]:
        """Generate synthetic clients with realistic characteristics"""
        logger.info(f"Generating {self.num_clients} synthetic clients with {self.heterogeneity_level} heterogeneity")
        
        for i in range(self.num_clients):
            client = self._generate_single_client(i)
            self.clients.append(client)
            
        # Log statistics
        self._log_client_statistics()
        
        return self.clients
    
    def _generate_single_client(self, client_id: int) -> SyntheticClient:
        """Generate a single synthetic client"""
        
        # Select model architecture
        model_family = np.random.choice(MODEL_FAMILIES[:4])  # Focus on main families
        model_size_mb = self._sample_model_size(model_family)
        
        # Generate data distribution using Dirichlet
        data_dist_raw = np.random.dirichlet([self.alpha] * len(PROBLEM_TYPES))
        data_distribution = dict(zip(PROBLEM_TYPES, data_dist_raw))
        
        # Generate difficulty distribution
        diff_dist_raw = np.random.dirichlet([self.alpha * 1.5] * len(DIFFICULTY_LEVELS))
        difficulty_distribution = dict(zip(DIFFICULTY_LEVELS, diff_dist_raw))
        
        # Calculate initial performance based on model size
        size_category = self._get_size_category(model_size_mb)
        perf_params = self.perf_params[size_category]
        
        # Local performance with noise
        local_pass_at_1 = np.clip(
            np.random.normal(perf_params['mean'], perf_params['std']), 
            0.0, 1.0
        )
        
        # Pass@k follows empirical relationship
        local_pass_at_5 = np.clip(local_pass_at_1 + np.random.uniform(0.1, 0.15), 0.0, 1.0)
        local_pass_at_10 = np.clip(local_pass_at_5 + np.random.uniform(0.05, 0.1), 0.0, 1.0)
        
        # CodeBLEU typically lower than pass@1
        local_codebleu = np.clip(
            local_pass_at_1 - np.random.uniform(0.05, 0.15),
            0.0, 1.0
        )
        
        # Global performance starts slightly lower (generalization gap)
        global_pass_at_1 = np.clip(
            local_pass_at_1 - np.random.uniform(0.02, 0.08),
            0.0, 1.0
        )
        global_pass_at_10 = np.clip(
            local_pass_at_10 - np.random.uniform(0.02, 0.08),
            0.0, 1.0
        )
        global_codebleu = np.clip(
            local_codebleu - np.random.uniform(0.02, 0.08),
            0.0, 1.0
        )
        
        # Trust score from Beta distribution (most clients trustworthy)
        trust_score = np.random.beta(4, 2)
        
        # Collaboration quality uniform in [0.3, 0.7]
        collaboration_quality = np.random.uniform(0.3, 0.7)
        
        # Communication efficiency uniform in [0.6, 0.9]
        communication_efficiency = np.random.uniform(0.6, 0.9)
        
        # SIER (Sensitive Information Exposure Rate) - should be low
        sier_avg = np.random.beta(1, 20)  # Most values near 0
        
        # Effective bandwidth normalized [0.3, 1.0]
        effective_bandwidth = np.random.uniform(0.3, 1.0)
        
        # Create client
        client = SyntheticClient(
            client_id=client_id,
            model_family=model_family,
            model_size_mb=model_size_mb,
            data_distribution=data_distribution,
            difficulty_distribution=difficulty_distribution,
            num_samples=np.random.randint(500, 2000),
            local_pass_at_1=local_pass_at_1,
            local_pass_at_5=local_pass_at_5,
            local_pass_at_10=local_pass_at_10,
            local_codebleu=local_codebleu,
            global_pass_at_1=global_pass_at_1,
            global_pass_at_10=global_pass_at_10,
            global_codebleu=global_codebleu,
            trust_score=trust_score,
            collaboration_quality=collaboration_quality,
            communication_efficiency=communication_efficiency,
            sier_avg=sier_avg,
            effective_bandwidth=effective_bandwidth
        )
        
        return client
    
    def _sample_model_size(self, model_family: str) -> float:
        """Sample realistic model size based on family"""
        size_ranges = {
            'pythia': [70, 160, 410, 1000, 1400],
            'opt': [125, 350, 1300, 2700],
            'gpt2': [82, 124, 355, 774],
            't5': [77, 220, 770, 3000],
            'codegen': [350, 2000, 6000],
            'cerebras': [111, 256, 590, 1300],
            'mt5': [300, 580, 1200],
            'gpt-neo': [125, 1300, 2700]
        }
        
        available_sizes = size_ranges.get(model_family, [100, 500, 1000])
        return float(np.random.choice(available_sizes))
    
    def _get_size_category(self, size_mb: float) -> str:
        """Categorize model size for performance parameters"""
        if size_mb <= 160:
            return '70M-160M'
        elif size_mb <= 410:
            return '250M-410M'
        elif size_mb <= 1500:
            return '1B-1.5B'
        else:
            return '3B+'
    
    def update_client_performance(self, client_id: int, performance_delta: Dict[str, float]):
        """Update client performance after collaboration"""
        client = self.clients[client_id]
        
        # Update metrics with bounds
        for metric, delta in performance_delta.items():
            if hasattr(client, metric):
                old_value = getattr(client, metric)
                new_value = np.clip(old_value + delta, 0.0, 1.0)
                setattr(client, metric, new_value)
                
                # Track pre-post difference
                if metric == 'local_pass_at_1':
                    client.pre_post_diff = new_value - old_value
                    client.historical_delta = 0.7 * client.historical_delta + 0.3 * client.pre_post_diff
        
        # Update performance history
        client.performance_history.append(client.local_pass_at_1)
        
        # Update learning rate estimate
        if len(client.performance_history) > 1:
            recent_improvement = client.performance_history[-1] - client.performance_history[-2]
            client.learning_rate = 0.8 * client.learning_rate + 0.2 * max(0, recent_improvement)
    
    def update_collaboration_metrics(self, client_i: int, client_j: int, 
                                   comm_kb: float, collaboration_success: bool):
        """Update collaboration-related metrics"""
        self.clients[client_i].collaboration_history.append(client_j)
        self.clients[client_j].collaboration_history.append(client_i)
        
        self.clients[client_i].comm_kb = comm_kb
        self.clients[client_j].comm_kb = comm_kb
        
        if collaboration_success:
            # Slightly improve collaboration quality
            self.clients[client_i].collaboration_quality = min(1.0, 
                self.clients[client_i].collaboration_quality * 1.05)
            self.clients[client_j].collaboration_quality = min(1.0,
                self.clients[client_j].collaboration_quality * 1.05)
    
    def _log_client_statistics(self):
        """Log statistics about generated clients"""
        model_counts = {}
        for client in self.clients:
            model_counts[client.model_family] = model_counts.get(client.model_family, 0) + 1
        
        avg_local_perf = np.mean([c.local_pass_at_1 for c in self.clients])
        avg_global_perf = np.mean([c.global_pass_at_1 for c in self.clients])
        avg_trust = np.mean([c.trust_score for c in self.clients])
        
        logger.info(f"Generated {self.num_clients} clients:")
        logger.info(f"  Model distribution: {model_counts}")
        logger.info(f"  Avg local pass@1: {avg_local_perf:.3f}")
        logger.info(f"  Avg global pass@1: {avg_global_perf:.3f}")
        logger.info(f"  Avg trust score: {avg_trust:.3f}")
        
        # Calculate heterogeneity statistics
        from scipy.spatial.distance import jensenshannon
        heterogeneity_scores = []
        for i in range(min(10, len(self.clients))):
            for j in range(i+1, min(10, len(self.clients))):
                dist_i = np.array([self.clients[i].data_distribution.get(t, 0) for t in PROBLEM_TYPES])
                dist_j = np.array([self.clients[j].data_distribution.get(t, 0) for t in PROBLEM_TYPES])
                dist_i = dist_i / (dist_i.sum() + 1e-8)
                dist_j = dist_j / (dist_j.sum() + 1e-8)
                heterogeneity_scores.append(jensenshannon(dist_i, dist_j))
        
        if heterogeneity_scores:
            logger.info(f"  Avg data heterogeneity (JS divergence): {np.mean(heterogeneity_scores):.3f}")
    
    def get_client_profiles_batch(self, client_ids: List[int]) -> List[np.ndarray]:
        """Get profile vectors for a batch of clients"""
        profiles = []
        for client_id in client_ids:
            if 0 <= client_id < len(self.clients):
                # Generate 16D profile matching client.py:make_profile()
                client = self.clients[client_id]
                
                # One-hot encode model family
                fam_bits = [0, 0, 0, 0]
                fam_idx = MODEL_FAMILIES.index(client.model_family) % 4
                fam_bits[fam_idx] = 1
                
                profile = np.array([
                    client.local_pass_at_1,  # last_perf
                    client.local_codebleu,   # last_codebleu
                    client.sier_avg,         # sier_avg
                    *fam_bits,               # one-hot family (4 dims)
                    client.model_size_mb / 1000,  # params_m / 1000
                    client.trust_score,      # trust
                    client.historical_delta, # historical_delta
                    client.comm_kb,          # comm_kb
                    client.pre_post_diff,    # pre_post_diff
                    client.effective_bandwidth,  # effective_bandwidth
                    0, 0, 0                  # padding zeros
                ])
                profiles.append(profile)
            else:
                profiles.append(np.zeros(16))
                
        return profiles


if __name__ == "__main__":
    # Test synthetic environment
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Synthetic Environment Generation")
    print("=" * 50)
    
    for heterogeneity in ['low', 'medium', 'high']:
        print(f"\nGenerating environment with {heterogeneity} heterogeneity:")
        env = SyntheticEnvironment(num_clients=8, heterogeneity_level=heterogeneity)
        clients = env.generate_clients()
        
        # Test profile generation
        profiles = env.get_client_profiles_batch([0, 1, 2])
        print(f"Generated {len(profiles)} profiles, shape: {profiles[0].shape}")
        
        # Test performance update
        env.update_client_performance(0, {
            'local_pass_at_1': 0.05,
            'local_codebleu': 0.03
        })
        print(f"Updated client 0 performance: {clients[0].local_pass_at_1:.3f}")
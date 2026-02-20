#!/usr/bin/env python3
"""
Enhanced Context Vector for KNEXA-FL with Three-Tier Evaluation Integration
Enables intelligent client pairing based on comprehensive performance metrics
"""
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class ClientProfile:
    """
    Comprehensive client profile incorporating three-tier evaluation metrics
    """
    client_id: int
    
    # Model characteristics
    model_family: str
    model_size_mb: float
    architecture_type: str
    
    # Data characteristics
    data_distribution: Dict[str, float]  # Problem type distribution
    difficulty_distribution: Dict[str, float]  # Easy/medium/hard distribution
    specialization_score: float  # How specialized vs generalized
    
    # Performance metrics (three-tier)
    local_performance: float
    transfer_performance: Optional[float]
    global_performance: float
    
    # Dynamic metrics
    trust_score: float
    learning_rate: float  # How quickly client improves
    collaboration_quality: float
    communication_efficiency: float
    
    # Historical context
    performance_trend: str  # 'improving', 'stable', 'declining'
    collaboration_history: List[int]  # Recent collaboration partners
    
    def to_context_vector(self, context_dim: int = 32) -> np.ndarray:
        """Convert profile to numerical context vector for LinUCB"""
        vector = np.zeros(context_dim)
        
        # Performance metrics (8 dimensions)
        vector[0] = self.local_performance
        vector[1] = self.transfer_performance if self.transfer_performance is not None else 0.0
        vector[2] = self.global_performance
        vector[3] = self.global_performance - self.local_performance  # Generalization gap
        vector[4] = self.trust_score
        vector[5] = self.learning_rate
        vector[6] = self.collaboration_quality
        vector[7] = self.communication_efficiency
        
        # Data characteristics (8 dimensions)
        data_types = ['algorithms', 'data_structures', 'string_processing', 'mathematics', 'recursion']
        for i, dtype in enumerate(data_types[:5]):
            vector[8 + i] = self.data_distribution.get(dtype, 0.0)
        
        difficulty_levels = ['easy', 'medium', 'hard']
        for i, difficulty in enumerate(difficulty_levels):
            vector[13 + i] = self.difficulty_distribution.get(difficulty, 0.0)
        
        vector[16] = self.specialization_score
        
        # Model characteristics (4 dimensions)
        vector[17] = self.model_size_mb / 1000.0  # Normalized
        vector[18] = 1.0 if 'gpt' in self.model_family.lower() else 0.0
        vector[19] = 1.0 if 'code' in self.model_family.lower() else 0.0
        vector[20] = 1.0 if 'dialog' in self.model_family.lower() else 0.0
        
        # Trend indicators (4 dimensions)
        vector[21] = 1.0 if self.performance_trend == 'improving' else 0.0
        vector[22] = 1.0 if self.performance_trend == 'stable' else 0.0
        vector[23] = 1.0 if self.performance_trend == 'declining' else 0.0
        vector[24] = len(self.collaboration_history) / 10.0  # Normalized collaboration count
        
        # Fill remaining dimensions with derived metrics
        for i in range(25, context_dim):
            if i == 25:
                vector[i] = self.local_performance * self.trust_score  # Reliable local performance
            elif i == 26:
                vector[i] = (self.transfer_performance or 0) * self.collaboration_quality  # Transfer effectiveness
            elif i == 27:
                vector[i] = self.global_performance * self.communication_efficiency  # Efficient global performance
            else:
                vector[i] = np.random.normal(0, 0.01)  # Small noise for regularization
        
        return vector


class EnhancedContextBuilder:
    """
    Builds comprehensive context vectors incorporating three-tier evaluation metrics
    """
    
    def __init__(self, context_dim: int = 32, history_window: int = 10):
        self.context_dim = context_dim
        self.history_window = history_window
        self.client_profiles = {}
        self.evaluation_history = {}
        self.collaboration_history = {}
        
    def update_client_profile(self, client_id: int, three_tier_results: Dict[str, Any],
                            model_info: Dict[str, Any], data_info: Dict[str, Any]):
        """
        Update client profile with latest three-tier evaluation results
        """
        # Initialize if new client
        if client_id not in self.client_profiles:
            self.client_profiles[client_id] = ClientProfile(
                client_id=client_id,
                model_family=model_info.get('family', 'unknown'),
                model_size_mb=model_info.get('size_mb', 1000),
                architecture_type=model_info.get('architecture', 'transformer'),
                data_distribution=data_info.get('type_distribution', {}),
                difficulty_distribution=data_info.get('difficulty_distribution', {}),
                specialization_score=data_info.get('specialization_score', 0.5),
                local_performance=0.0,
                transfer_performance=None,
                global_performance=0.0,
                trust_score=0.8,  # Initial trust
                learning_rate=0.0,
                collaboration_quality=0.5,
                communication_efficiency=0.8,
                performance_trend='stable',
                collaboration_history=[]
            )
        
        profile = self.client_profiles[client_id]
        
        # Update performance metrics
        new_local = three_tier_results.get('local_pass@1', profile.local_performance)
        new_transfer = three_tier_results.get('transfer_pass@1')
        new_global = three_tier_results.get('global_pass@1', profile.global_performance)
        
        # Calculate learning rate
        local_improvement = new_local - profile.local_performance
        global_improvement = new_global - profile.global_performance
        profile.learning_rate = 0.7 * profile.learning_rate + 0.3 * (local_improvement + global_improvement)
        
        # Update performance
        profile.local_performance = new_local
        profile.transfer_performance = new_transfer
        profile.global_performance = new_global
        
        # Update performance trend
        if client_id not in self.evaluation_history:
            self.evaluation_history[client_id] = deque(maxlen=self.history_window)
        
        self.evaluation_history[client_id].append({
            'local': new_local,
            'global': new_global,
            'transfer': new_transfer,
            'timestamp': datetime.now()
        })
        
        profile.performance_trend = self._calculate_performance_trend(client_id)
        
        # Update collaboration quality based on recent transfer performance
        if new_transfer is not None:
            transfer_improvement = new_transfer - profile.local_performance
            profile.collaboration_quality = 0.6 * profile.collaboration_quality + 0.4 * max(0, transfer_improvement)
        
        # Update trust score based on performance consistency
        performance_consistency = self._calculate_performance_consistency(client_id)
        profile.trust_score = 0.8 * profile.trust_score + 0.2 * performance_consistency
        profile.trust_score = np.clip(profile.trust_score, 0.1, 1.0)
        
        logger.info(f"Updated profile for Client {client_id}: "
                   f"Local={new_local:.3f}, Global={new_global:.3f}, "
                   f"Trust={profile.trust_score:.3f}, Trend={profile.performance_trend}")
        
        return profile
    
    def build_pairwise_context(self, client_i: int, client_j: int, round_id: int) -> np.ndarray:
        """
        Build pairwise context vector for LinUCB bandit decision
        """
        profile_i = self.client_profiles.get(client_i)
        profile_j = self.client_profiles.get(client_j)
        
        if profile_i is None or profile_j is None:
            logger.warning(f"Missing profiles for pairing ({client_i}, {client_j})")
            return np.zeros(self.context_dim * 2 + 8)  # Expanded for pairwise features
        
        # Get individual context vectors
        context_i = profile_i.to_context_vector(self.context_dim)
        context_j = profile_j.to_context_vector(self.context_dim)
        
        # Create pairwise features
        pairwise_features = np.zeros(8)
        
        # Data heterogeneity features (key hypothesis)
        pairwise_features[0] = self._calculate_data_heterogeneity(profile_i, profile_j)
        pairwise_features[1] = self._calculate_performance_complementarity(profile_i, profile_j)
        pairwise_features[2] = self._calculate_trust_compatibility(profile_i, profile_j)
        pairwise_features[3] = self._calculate_collaboration_potential(profile_i, profile_j)
        
        # Model heterogeneity features
        pairwise_features[4] = self._calculate_model_diversity(profile_i, profile_j)
        pairwise_features[5] = self._calculate_learning_rate_compatibility(profile_i, profile_j)
        
        # Temporal features
        pairwise_features[6] = round_id / 25.0  # Normalized round progress
        pairwise_features[7] = self._calculate_previous_collaboration_score(client_i, client_j)
        
        # Concatenate all features
        pairwise_context = np.concatenate([context_i, context_j, pairwise_features])
        
        return pairwise_context
    
    def _calculate_data_heterogeneity(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate data heterogeneity bonus - core hypothesis of KNEXA-FL
        Higher score for more diverse problem type distributions
        """
        # Problem type diversity
        types_i = np.array([profile_i.data_distribution.get(t, 0) for t in 
                           ['algorithms', 'data_structures', 'string_processing', 'mathematics', 'recursion']])
        types_j = np.array([profile_j.data_distribution.get(t, 0) for t in 
                           ['algorithms', 'data_structures', 'string_processing', 'mathematics', 'recursion']])
        
        # Normalize distributions
        types_i = types_i / (types_i.sum() + 1e-8)
        types_j = types_j / (types_j.sum() + 1e-8)
        
        # Calculate Jensen-Shannon divergence (higher = more heterogeneous)
        from scipy.spatial.distance import jensenshannon
        js_divergence = jensenshannon(types_i, types_j)
        
        # Difficulty diversity
        diff_i = np.array([profile_i.difficulty_distribution.get(d, 0) for d in ['easy', 'medium', 'hard']])
        diff_j = np.array([profile_j.difficulty_distribution.get(d, 0) for d in ['easy', 'medium', 'hard']])
        
        diff_i = diff_i / (diff_i.sum() + 1e-8)
        diff_j = diff_j / (diff_j.sum() + 1e-8)
        
        difficulty_divergence = jensenshannon(diff_i, diff_j)
        
        # Combined heterogeneity score
        heterogeneity_score = 0.7 * js_divergence + 0.3 * difficulty_divergence
        
        return float(heterogeneity_score)
    
    def _calculate_performance_complementarity(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate how well clients complement each other's strengths and weaknesses
        """
        # Local performance complementarity
        local_gap = abs(profile_i.local_performance - profile_j.local_performance)
        
        # Global performance complementarity
        global_gap = abs(profile_i.global_performance - profile_j.global_performance)
        
        # Specialization complementarity (one specialized + one generalized = good)
        specialization_complement = abs(profile_i.specialization_score - profile_j.specialization_score)
        
        # Combined complementarity (moderate gaps are good, extreme gaps are bad)
        complementarity = np.exp(-2 * local_gap) + np.exp(-2 * global_gap) + specialization_complement
        
        return float(complementarity)
    
    def _calculate_trust_compatibility(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate trust compatibility for reliable collaboration
        """
        # Both clients should have reasonable trust scores
        min_trust = min(profile_i.trust_score, profile_j.trust_score)
        avg_trust = (profile_i.trust_score + profile_j.trust_score) / 2
        
        # Prefer pairs where both have high trust
        trust_compatibility = 0.6 * min_trust + 0.4 * avg_trust
        
        return float(trust_compatibility)
    
    def _calculate_collaboration_potential(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate potential for effective collaboration
        """
        # Both should have reasonable collaboration quality
        collab_quality = (profile_i.collaboration_quality + profile_j.collaboration_quality) / 2
        
        # Both should be improving or stable
        trend_bonus = 0.0
        if profile_i.performance_trend in ['improving', 'stable'] and profile_j.performance_trend in ['improving', 'stable']:
            trend_bonus = 0.3
        
        # Communication efficiency
        comm_efficiency = (profile_i.communication_efficiency + profile_j.communication_efficiency) / 2
        
        collaboration_potential = collab_quality + trend_bonus + 0.2 * comm_efficiency
        
        return float(collaboration_potential)
    
    def _calculate_model_diversity(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate model architecture diversity bonus
        """
        # Different model families should pair well
        family_diversity = 1.0 if profile_i.model_family != profile_j.model_family else 0.3
        
        # Different sizes can complement each other
        size_ratio = min(profile_i.model_size_mb, profile_j.model_size_mb) / max(profile_i.model_size_mb, profile_j.model_size_mb)
        size_diversity = 1.0 - size_ratio  # Higher for more different sizes
        
        model_diversity = 0.7 * family_diversity + 0.3 * size_diversity
        
        return float(model_diversity)
    
    def _calculate_learning_rate_compatibility(self, profile_i: ClientProfile, profile_j: ClientProfile) -> float:
        """
        Calculate learning rate compatibility
        """
        # Prefer clients with positive learning rates
        avg_learning_rate = (profile_i.learning_rate + profile_j.learning_rate) / 2
        
        # Bonus if both are learning
        both_learning = 1.0 if profile_i.learning_rate > 0 and profile_j.learning_rate > 0 else 0.5
        
        learning_compatibility = both_learning * max(0, avg_learning_rate)
        
        return float(learning_compatibility)
    
    def _calculate_previous_collaboration_score(self, client_i: int, client_j: int) -> float:
        """
        Calculate score based on previous collaborations
        """
        if client_i not in self.collaboration_history:
            return 0.5  # Neutral for no history
        
        # Slight penalty for recent collaborations (encourage exploration)
        recent_collaborations = self.collaboration_history.get(client_i, [])
        if client_j in recent_collaborations[-3:]:  # Last 3 collaborations
            return 0.2  # Penalty for recent collaboration
        elif client_j in recent_collaborations:
            return 0.4  # Some penalty for previous collaboration
        else:
            return 0.8  # Bonus for new collaboration
    
    def _calculate_performance_trend(self, client_id: int) -> str:
        """
        Calculate performance trend from evaluation history
        """
        if client_id not in self.evaluation_history or len(self.evaluation_history[client_id]) < 3:
            return 'stable'
        
        history = list(self.evaluation_history[client_id])
        recent_scores = [h['global'] for h in history[-3:]]
        
        if len(recent_scores) < 3:
            return 'stable'
        
        # Simple trend analysis
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        if trend > 0.02:
            return 'improving'
        elif trend < -0.02:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_performance_consistency(self, client_id: int) -> float:
        """
        Calculate performance consistency for trust scoring
        """
        if client_id not in self.evaluation_history or len(self.evaluation_history[client_id]) < 2:
            return 0.8  # Default trust
        
        history = list(self.evaluation_history[client_id])
        global_scores = [h['global'] for h in history]
        
        if len(global_scores) < 2:
            return 0.8
        
        # Higher consistency = lower variance
        consistency = 1.0 / (1.0 + np.var(global_scores))
        return min(1.0, consistency)
    
    def record_collaboration(self, client_i: int, client_j: int, collaboration_result: Dict[str, Any]):
        """
        Record collaboration result for future pairing decisions
        """
        if client_i not in self.collaboration_history:
            self.collaboration_history[client_i] = []
        if client_j not in self.collaboration_history:
            self.collaboration_history[client_j] = []
        
        # Record mutual collaboration
        self.collaboration_history[client_i].append(client_j)
        self.collaboration_history[client_j].append(client_i)
        
        # Keep only recent collaborations
        self.collaboration_history[client_i] = self.collaboration_history[client_i][-10:]
        self.collaboration_history[client_j] = self.collaboration_history[client_j][-10:]
        
        # Update collaboration quality based on results
        if 'knowledge_gain' in collaboration_result:
            knowledge_gain = collaboration_result['knowledge_gain']
            
            # Update collaboration quality for both clients
            if client_i in self.client_profiles:
                current_quality = self.client_profiles[client_i].collaboration_quality
                self.client_profiles[client_i].collaboration_quality = 0.7 * current_quality + 0.3 * max(0, knowledge_gain)
            
            if client_j in self.client_profiles:
                current_quality = self.client_profiles[client_j].collaboration_quality
                self.client_profiles[client_j].collaboration_quality = 0.7 * current_quality + 0.3 * max(0, knowledge_gain)
    
    def get_heterogeneity_analysis(self) -> Dict[str, Any]:
        """
        Generate comprehensive heterogeneity analysis for validation
        """
        if len(self.client_profiles) < 2:
            return {'error': 'Insufficient client profiles for analysis'}
        
        client_ids = list(self.client_profiles.keys())
        heterogeneity_matrix = np.zeros((len(client_ids), len(client_ids)))
        
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i != j:
                    profile_i = self.client_profiles[client_i]
                    profile_j = self.client_profiles[client_j]
                    heterogeneity_score = self._calculate_data_heterogeneity(profile_i, profile_j)
                    heterogeneity_matrix[i][j] = heterogeneity_score
        
        analysis = {
            'client_ids': client_ids,
            'heterogeneity_matrix': heterogeneity_matrix.tolist(),
            'mean_heterogeneity': float(np.mean(heterogeneity_matrix[heterogeneity_matrix > 0])),
            'min_heterogeneity': float(np.min(heterogeneity_matrix[heterogeneity_matrix > 0])),
            'max_heterogeneity': float(np.max(heterogeneity_matrix)),
            'heterogeneity_validation': {
                'sufficient_diversity': np.mean(heterogeneity_matrix[heterogeneity_matrix > 0]) > 0.3,
                'all_pairs_diverse': np.min(heterogeneity_matrix[heterogeneity_matrix > 0]) > 0.1
            }
        }
        
        return analysis


if __name__ == "__main__":
    print("🧠 Enhanced Context Vector for KNEXA-FL")
    print("=" * 50)
    print("✅ Three-tier evaluation integration")
    print("✅ Data heterogeneity scoring")
    print("✅ Performance complementarity analysis") 
    print("✅ Trust and collaboration quality tracking")
    print("✅ Model diversity considerations")
    print()
    print("Ready for intelligent client pairing with LinUCB bandit optimization")
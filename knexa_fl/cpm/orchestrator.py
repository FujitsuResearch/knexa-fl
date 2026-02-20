"""
Central Profiler/Matchmaker (CPM) Orchestrator for KNEXA-FL
Non-aggregating orchestrator for intelligent peer matching
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .linucb import LinUCB
from .privacy_profile import (
    ProfileSanitizer, PerformanceFeedbackSanitizer,
    AbstractProfile, PrivacyParameters
)

@dataclass
class PeerPairing:
    """Information about a peer pairing decision"""
    student_id: str
    teacher_id: str
    alpha: float = 0.5              # Knowledge distillation weight
    temperature: float = 2.0        # Distillation temperature
    expected_reward: float = 0.0    # Expected utility from LinUCB
    round_id: int = 0
    timestamp: float = 0.0

class CPMOrchestrator:
    """
    Central Profiler/Matchmaker - CONTENT-BLIND Orchestrator
    
    PRIVACY GUARANTEES:
    - NEVER has access to knowledge transfer content (logits, parameters)
    - NEVER has access to raw model parameters or training data
    - ONLY sees k-anonymous, DP-protected abstract profiles
    - ONLY receives aggregate performance deltas as feedback
    """
    
    def __init__(self, 
                 max_pairs_per_round: int = 10,
                 context_dim: int = 32,
                 privacy_params: Optional[PrivacyParameters] = None):
        """
        Initialize CPM Orchestrator.
        
        Args:
            max_pairs_per_round: Maximum number of pairs to form per round
            context_dim: Dimension of context vectors for LinUCB
            privacy_params: Privacy parameters for profile sanitization
        """
        # Privacy-preserving components
        self.privacy_params = privacy_params or PrivacyParameters()
        self.profile_sanitizer = ProfileSanitizer(self.privacy_params)
        self.feedback_sanitizer = PerformanceFeedbackSanitizer()
        
        # Abstract profile storage (NO raw agent data)
        self.abstract_profiles: Dict[str, AbstractProfile] = {}
        
        # LinUCB bandit for intelligent matching
        self.bandit = LinUCB(d=context_dim)
        
        # Configuration
        self.max_pairs_per_round = max_pairs_per_round
        
        # Performance tracking (AGGREGATE ONLY)
        self.matching_history: List[Dict[str, Any]] = []
        self.aggregate_feedback: Dict[str, List[float]] = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=16)
        
        # CPM Orchestrator initialized with privacy parameters
    
    def update_agent_profile(self, agent_id: str, raw_profile: Dict[str, Any]) -> bool:
        """
        Update agent profile (PRIVACY-PRESERVING).
        
        Args:
            agent_id: Agent identifier
            raw_profile: Raw profile data from agent
            
        Returns:
            Success status
        """
        try:
            # CRITICAL: Sanitize the profile to ensure privacy
            abstract_profile = self.profile_sanitizer.sanitize_profile(
                raw_profile, agent_id
            )
            
            # Store only the abstract profile
            self.abstract_profiles[agent_id] = abstract_profile
            
            return True
            
        except Exception as e:
            # Error updating profile
            return False
    
    def request_matching(self, 
                        available_agents: List[str], 
                        round_id: int) -> List[PeerPairing]:
        """
        Request intelligent peer matching using LinUCB.
        
        Args:
            available_agents: List of available agent IDs
            round_id: Current round number
            
        Returns:
            List of peer pairings
        """
        # Filter to agents with profiles
        valid_agents = [
            agent_id for agent_id in available_agents 
            if agent_id in self.abstract_profiles
        ]
        
        if len(valid_agents) < 2:
            return []
        
        # Get abstract profile vectors
        profile_vectors = []
        agent_indices = []
        
        for agent_id in valid_agents:
            profile = self.abstract_profiles[agent_id]
            profile_vectors.append(profile.to_vector())
            agent_indices.append(agent_id)
        
        # Use LinUCB to select pairs
        max_pairs = min(self.max_pairs_per_round, len(valid_agents) // 2)
        selected_pairs = self.bandit.select_pairs(
            profile_vectors, max_pairs, round_id
        )
        
        # Convert to PeerPairing objects
        pairings = []
        for i, j, alpha, temperature in selected_pairs:
            student_id = agent_indices[i]
            teacher_id = agent_indices[j]
            
            # Get expected reward from LinUCB
            context = np.concatenate([profile_vectors[i], profile_vectors[j]])
            expected_reward = self.bandit.get_ucb_score(context, round_id)
            
            pairing = PeerPairing(
                student_id=student_id,
                teacher_id=teacher_id,
                alpha=alpha,
                temperature=temperature,
                expected_reward=expected_reward,
                round_id=round_id,
                timestamp=time.time()
            )
            pairings.append(pairing)
        
        # Store matching history (no sensitive information)
        self.matching_history.append({
            'round_id': round_id,
            'num_agents': len(valid_agents),
            'num_pairings': len(pairings),
            'avg_expected_reward': np.mean([p.expected_reward for p in pairings]) if pairings else 0.0,
            'timestamp': time.time()
        })
        
        return pairings
    
    def update_feedback(self, 
                       student_id: str, 
                       teacher_id: str,
                       raw_feedback: Dict[str, Any]) -> bool:
        """
        Update bandit with performance feedback (PRIVACY-PRESERVING).
        
        Args:
            student_id: Student agent ID
            teacher_id: Teacher agent ID
            raw_feedback: Raw performance feedback
            
        Returns:
            Success status
        """
        try:
            # CRITICAL: Sanitize feedback to ensure no sensitive information
            sanitized_feedback = self.feedback_sanitizer.sanitize_performance_feedback(
                raw_feedback
            )
            
            # Store only aggregate feedback
            if student_id not in self.aggregate_feedback:
                self.aggregate_feedback[student_id] = []
            
            perf_delta = sanitized_feedback['performance_delta']
            self.aggregate_feedback[student_id].append(perf_delta)
            
            # Update LinUCB bandit
            if student_id in self.abstract_profiles and teacher_id in self.abstract_profiles:
                student_profile = self.abstract_profiles[student_id]
                teacher_profile = self.abstract_profiles[teacher_id]
                
                # Build pairwise context
                student_vector = student_profile.to_vector()
                teacher_vector = teacher_profile.to_vector()
                pairwise_context = np.concatenate([student_vector, teacher_vector])
                
                # Update bandit with sanitized reward
                self.bandit.update(pairwise_context, perf_delta)
                
                # Update trust scores in abstract profiles
                trust_change = sanitized_feedback['trust_change']
                new_trust = student_profile.trust_score_quantized + int(trust_change * 10)
                student_profile.trust_score_quantized = max(0, min(10, new_trust))
                student_profile.updated_at = time.time()
            
            return True
            
        except Exception as e:
            # Error updating feedback
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get CPM statistics (CONTENT-BLIND)"""
        bandit_stats = self.bandit.get_statistics()
        
        return {
            'num_agents': len(self.abstract_profiles),
            'num_rounds': len(self.matching_history),
            'total_pairings': sum(h['num_pairings'] for h in self.matching_history),
            'bandit_stats': bandit_stats,
            'aggregate_feedback_agents': len(self.aggregate_feedback),
            'privacy_guarantees': {
                'content_blind': True,
                'k_anonymity': self.privacy_params.k_anonymity,
                'dp_epsilon': self.privacy_params.dp_epsilon,
                'knowledge_content_access': False,
                'model_parameters_access': False
            }
        }
    
    def shutdown(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
#!/usr/bin/env python3
"""
Privacy-Preserving Profile System for KNEXA-FL
Implements k-anonymity and differential privacy for client profiles
Ensures CPM only sees abstract, sanitized information
"""

import numpy as np
import hashlib
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ModelSizeCategory(Enum):
    """Categorical model size representation"""
    ULTRA_SMALL = 1    # < 100M parameters
    SMALL = 2          # 100M - 300M parameters  
    MEDIUM = 3         # 300M - 1B parameters
    LARGE = 4          # 1B - 10B parameters
    ULTRA_LARGE = 5    # > 10B parameters

class ArchitectureType(Enum):
    """Model architecture categories"""
    DECODER_ONLY = 1      # GPT-style decoders
    ENCODER_DECODER = 2   # T5-style models
    ENCODER_ONLY = 3      # BERT-style encoders

class PerformanceCategory(Enum):
    """Performance level categories"""
    POOR = 1          # Bottom 20%
    BELOW_AVERAGE = 2 # 20-40%
    AVERAGE = 3       # 40-60%
    ABOVE_AVERAGE = 4 # 60-80%
    EXCELLENT = 5     # Top 20%

@dataclass
class PrivacyParameters:
    """Privacy parameters for profile sanitization"""
    k_anonymity: int = 3               # Minimum group size for k-anonymity
    dp_epsilon: float = 1.0            # Differential privacy epsilon
    dp_delta: float = 1e-5             # Differential privacy delta
    noise_scale: float = 0.1           # Noise scale for continuous values
    quantization_levels: int = 10      # Quantization levels for continuous values
    hash_salt: str = "knexa_privacy"   # Salt for hashing sensitive identifiers

@dataclass
class AbstractProfile:
    """
    Abstract, privacy-preserving client profile
    Contains only sanitized information visible to CPM
    """
    # Anonymous identifier (hashed)
    profile_id: str
    
    # Categorical information (k-anonymous)
    model_size_category: ModelSizeCategory
    architecture_type: ArchitectureType
    performance_category: PerformanceCategory
    
    # Quantized continuous values (DP-protected)
    trust_score_quantized: int          # 0-10 quantized trust score
    specialization_score_quantized: int # 0-10 quantized specialization
    collaboration_score_quantized: int  # 0-10 quantized collaboration quality
    
    # Derived features (aggregated)
    capability_vector: List[float]      # 8-dimensional capability vector
    compatibility_hash: str             # Hash of compatibility features
    
    # Temporal information
    profile_age: int                    # Hours since first registration
    last_update_age: int               # Hours since last update
    
    # Aggregated statistics (no raw data)
    successful_exchanges: int           # Count of successful exchanges
    total_exchanges: int               # Total exchange attempts
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert profile to numerical vector for LinUCB bandit"""
        vector = np.zeros(16)  # 16-dimensional vector
        
        # Categorical features (one-hot encoded)
        vector[0] = float(self.model_size_category.value) / 5.0
        vector[1] = float(self.architecture_type.value) / 3.0
        vector[2] = float(self.performance_category.value) / 5.0
        
        # Quantized continuous features
        vector[3] = self.trust_score_quantized / 10.0
        vector[4] = self.specialization_score_quantized / 10.0
        vector[5] = self.collaboration_score_quantized / 10.0
        
        # Capability vector (8 dimensions)
        vector[6:14] = self.capability_vector
        
        # Derived features
        vector[14] = min(1.0, self.profile_age / 168.0)  # Normalize to weekly
        success_rate = self.successful_exchanges / max(1, self.total_exchanges)
        vector[15] = success_rate
        
        return vector
    
    def get_compatibility_features(self) -> Dict[str, Any]:
        """Get compatibility features for matching"""
        return {
            'model_size_category': self.model_size_category.value,
            'architecture_type': self.architecture_type.value,
            'performance_category': self.performance_category.value,
            'trust_level': self.trust_score_quantized,
            'specialization_level': self.specialization_score_quantized,
            'collaboration_level': self.collaboration_score_quantized,
            'capability_hash': self.compatibility_hash
        }


class ProfileSanitizer:
    """
    Sanitizes client profiles for privacy-preserving CPM access
    Implements k-anonymity and differential privacy
    """
    
    def __init__(self, privacy_params: PrivacyParameters = None):
        self.privacy_params = privacy_params or PrivacyParameters()
        self.profile_groups: Dict[str, List[str]] = {}  # For k-anonymity
        self.global_stats: Dict[str, float] = {}        # For normalization
        
        logger.info(f"Profile sanitizer initialized with k={self.privacy_params.k_anonymity}, ε={self.privacy_params.dp_epsilon}")
    
    def sanitize_profile(self, raw_profile: Dict[str, Any], client_id: str) -> AbstractProfile:
        """
        Sanitize raw client profile to abstract profile
        
        Args:
            raw_profile: Raw profile data from client
            client_id: Client identifier
            
        Returns:
            Sanitized abstract profile
        """
        try:
            # Create anonymous profile ID
            profile_id = self._create_anonymous_id(client_id)
            
            # Categorize model size (ensure it's numeric)
            model_size_mb = raw_profile.get('model_size_mb', 100)
            # Ensure model_size_mb is numeric
            if isinstance(model_size_mb, str):
                try:
                    model_size_mb = float(model_size_mb.replace('M', '').replace('B', ''))
                except:
                    logger.error(f"Invalid model_size_mb format: {model_size_mb}")
                    model_size_mb = 100  # Default fallback
            model_size_category = self._categorize_model_size(float(model_size_mb))
            
            # Categorize architecture
            architecture_type = self._categorize_architecture(
                raw_profile.get('architecture_type', 'decoder')
            )
            
            # Categorize performance
            performance = raw_profile.get('performance', 0.5)
            performance_category = self._categorize_performance(performance)
            
            # Quantize continuous values with DP noise
            trust_score = raw_profile.get('trust_score', 0.8)
            trust_quantized = self._quantize_with_dp(trust_score, 0, 1)
            
            specialization = raw_profile.get('specialization_score', 0.5)
            specialization_quantized = self._quantize_with_dp(specialization, 0, 1)
            
            collaboration = raw_profile.get('collaboration_quality', 0.5)
            collaboration_quantized = self._quantize_with_dp(collaboration, 0, 1)
            
            # Create capability vector
            capability_vector = self._create_capability_vector(raw_profile)
            
            # Create compatibility hash
            compatibility_hash = self._create_compatibility_hash(
                model_size_category, architecture_type, performance_category
            )
            
            # Calculate temporal information
            profile_age = self._calculate_profile_age(client_id)
            last_update_age = 0  # Just updated
            
            # Get exchange statistics
            exchange_stats = raw_profile.get('exchange_stats', {'successful': 0, 'total': 0})
            
            # Create abstract profile
            abstract_profile = AbstractProfile(
                profile_id=profile_id,
                model_size_category=model_size_category,
                architecture_type=architecture_type,
                performance_category=performance_category,
                trust_score_quantized=trust_quantized,
                specialization_score_quantized=specialization_quantized,
                collaboration_score_quantized=collaboration_quantized,
                capability_vector=capability_vector,
                compatibility_hash=compatibility_hash,
                profile_age=profile_age,
                last_update_age=last_update_age,
                successful_exchanges=exchange_stats['successful'],
                total_exchanges=exchange_stats['total']
            )
            
            # Apply k-anonymity check
            if self._check_k_anonymity(abstract_profile):
                logger.info(f"Profile sanitized successfully for client {client_id}")
                return abstract_profile
            else:
                # Apply generalization for k-anonymity
                return self._generalize_for_k_anonymity(abstract_profile)
                
        except Exception as e:
            logger.error(f"Error sanitizing profile for client {client_id}: {e}")
            return self._create_default_profile(client_id)
    
    def _create_anonymous_id(self, client_id: str) -> str:
        """Create anonymous profile ID using cryptographic hash"""
        hash_input = f"{client_id}:{self.privacy_params.hash_salt}:{int(time.time() // 3600)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _categorize_model_size(self, size_mb: float) -> ModelSizeCategory:
        """Categorize model size to protect exact parameter counts"""
        if size_mb < 100:
            return ModelSizeCategory.ULTRA_SMALL
        elif size_mb < 300:
            return ModelSizeCategory.SMALL
        elif size_mb < 1000:
            return ModelSizeCategory.MEDIUM
        elif size_mb < 10000:
            return ModelSizeCategory.LARGE
        else:
            return ModelSizeCategory.ULTRA_LARGE
    
    def _categorize_architecture(self, arch_type: str) -> ArchitectureType:
        """Categorize architecture type"""
        if 'encoder-decoder' in arch_type.lower():
            return ArchitectureType.ENCODER_DECODER
        elif 'encoder' in arch_type.lower():
            return ArchitectureType.ENCODER_ONLY
        else:
            return ArchitectureType.DECODER_ONLY
    
    def _categorize_performance(self, performance: float) -> PerformanceCategory:
        """Categorize performance into quintiles"""
        if performance < 0.2:
            return PerformanceCategory.POOR
        elif performance < 0.4:
            return PerformanceCategory.BELOW_AVERAGE
        elif performance < 0.6:
            return PerformanceCategory.AVERAGE
        elif performance < 0.8:
            return PerformanceCategory.ABOVE_AVERAGE
        else:
            return PerformanceCategory.EXCELLENT
    
    def _quantize_with_dp(self, value: float, min_val: float, max_val: float) -> int:
        """Quantize continuous value with differential privacy noise"""
        # Clip value to range
        clipped_value = np.clip(value, min_val, max_val)
        
        # Add Laplace noise for differential privacy
        noise_scale = self.privacy_params.noise_scale / self.privacy_params.dp_epsilon
        noisy_value = clipped_value + np.random.laplace(0, noise_scale)
        
        # Quantize to discrete levels
        quantized = int(np.clip(
            noisy_value * self.privacy_params.quantization_levels,
            0, self.privacy_params.quantization_levels
        ))
        
        return quantized
    
    def _create_capability_vector(self, raw_profile: Dict[str, Any]) -> List[float]:
        """Create 8-dimensional capability vector"""
        # Extract capabilities with privacy protection
        capabilities = []
        
        # General capabilities (normalized and noisy)
        capabilities.append(self._add_dp_noise(raw_profile.get('code_capability', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('reasoning_capability', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('language_capability', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('math_capability', 0.5)))
        
        # Derived capabilities
        capabilities.append(self._add_dp_noise(raw_profile.get('adaptation_speed', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('generalization_ability', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('knowledge_retention', 0.5)))
        capabilities.append(self._add_dp_noise(raw_profile.get('transfer_efficiency', 0.5)))
        
        # Ensure all values are in [0, 1]
        return [max(0.0, min(1.0, cap)) for cap in capabilities]
    
    def _create_compatibility_hash(self, model_size: ModelSizeCategory, 
                                  arch_type: ArchitectureType, 
                                  performance: PerformanceCategory) -> str:
        """Create compatibility hash for matching"""
        hash_input = f"{model_size.value}:{arch_type.value}:{performance.value}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def _add_dp_noise(self, value: float) -> float:
        """Add differential privacy noise to a value"""
        noise_scale = self.privacy_params.noise_scale / self.privacy_params.dp_epsilon
        return value + np.random.laplace(0, noise_scale)
    
    def _calculate_profile_age(self, client_id: str) -> int:
        """Calculate profile age in hours (simulated)"""
        # In a real implementation, this would track actual registration time
        return random.randint(1, 168)  # 1 hour to 1 week
    
    def _check_k_anonymity(self, profile: AbstractProfile) -> bool:
        """Check if profile satisfies k-anonymity"""
        # Create quasi-identifier tuple
        quasi_id = (
            profile.model_size_category.value,
            profile.architecture_type.value,
            profile.performance_category.value,
            profile.trust_score_quantized,
            profile.specialization_score_quantized
        )
        
        quasi_id_str = str(quasi_id)
        
        # Check if this quasi-identifier has at least k similar profiles
        if quasi_id_str not in self.profile_groups:
            self.profile_groups[quasi_id_str] = []
        
        self.profile_groups[quasi_id_str].append(profile.profile_id)
        
        return len(self.profile_groups[quasi_id_str]) >= self.privacy_params.k_anonymity
    
    def _generalize_for_k_anonymity(self, profile: AbstractProfile) -> AbstractProfile:
        """Generalize profile to achieve k-anonymity"""
        # Apply generalization by reducing specificity
        
        # Generalize performance category
        if profile.performance_category in [PerformanceCategory.POOR, PerformanceCategory.BELOW_AVERAGE]:
            profile.performance_category = PerformanceCategory.BELOW_AVERAGE
        elif profile.performance_category in [PerformanceCategory.ABOVE_AVERAGE, PerformanceCategory.EXCELLENT]:
            profile.performance_category = PerformanceCategory.ABOVE_AVERAGE
        
        # Generalize quantized scores (reduce precision)
        profile.trust_score_quantized = (profile.trust_score_quantized // 2) * 2
        profile.specialization_score_quantized = (profile.specialization_score_quantized // 2) * 2
        profile.collaboration_score_quantized = (profile.collaboration_score_quantized // 2) * 2
        
        logger.info(f"Applied generalization for k-anonymity: {profile.profile_id}")
        return profile
    
    def _create_default_profile(self, client_id: str) -> AbstractProfile:
        """NO DEFAULT PROFILES - must use real data"""
        logger.error(f"Profile sanitization failed for {client_id} - cannot create synthetic default profile")
        raise ValueError(f"Profile sanitization failed for {client_id} - real profile data required")
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy metrics for monitoring"""
        return {
            'k_anonymity_parameter': self.privacy_params.k_anonymity,
            'dp_epsilon': self.privacy_params.dp_epsilon,
            'dp_delta': self.privacy_params.dp_delta,
            'profile_groups': len(self.profile_groups),
            'average_group_size': np.mean([len(group) for group in self.profile_groups.values()]) if self.profile_groups else 0,
            'anonymization_rate': 1.0  # All profiles are anonymized
        }


class PerformanceFeedbackSanitizer:
    """
    Sanitizes performance feedback to ensure CPM only sees aggregate metrics
    """
    
    def __init__(self, privacy_params: PrivacyParameters = None):
        self.privacy_params = privacy_params or PrivacyParameters()
        
    def sanitize_performance_feedback(self, raw_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize performance feedback to only include aggregate metrics
        
        Args:
            raw_feedback: Raw performance feedback from client
            
        Returns:
            Sanitized feedback with only aggregate metrics
        """
        try:
            # Only include aggregate performance metrics
            sanitized = {
                'performance_delta': self._clip_and_noise(
                    raw_feedback.get('performance_delta', 0.0), -0.1, 0.1
                ),
                'exchange_success': bool(raw_feedback.get('exchange_success', False)),
                'trust_change': self._clip_and_noise(
                    raw_feedback.get('trust_change', 0.0), -0.2, 0.2
                ),
                'collaboration_quality': self._quantize_with_dp(
                    raw_feedback.get('collaboration_quality', 0.5), 0, 1
                ),
                'timestamp': time.time()
            }
            
            # Explicitly exclude any detailed information
            excluded_keys = [
                'detailed_logs', 'training_metrics', 'model_parameters',
                'knowledge_content', 'raw_performance', 'step_by_step_metrics'
            ]
            
            for key in excluded_keys:
                if key in raw_feedback:
                    logger.warning(f"Excluded sensitive key '{key}' from feedback")
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing performance feedback: {e}")
            return {
                'performance_delta': 0.0,
                'exchange_success': False,
                'trust_change': 0.0,
                'collaboration_quality': 5,
                'timestamp': time.time()
            }
    
    def _clip_and_noise(self, value: float, min_val: float, max_val: float) -> float:
        """Clip value and add DP noise"""
        clipped = np.clip(value, min_val, max_val)
        noise_scale = self.privacy_params.noise_scale / self.privacy_params.dp_epsilon
        return clipped + np.random.laplace(0, noise_scale)
    
    def _quantize_with_dp(self, value: float, min_val: float, max_val: float) -> int:
        """Quantize value with DP noise"""
        clipped = np.clip(value, min_val, max_val)
        noise_scale = self.privacy_params.noise_scale / self.privacy_params.dp_epsilon
        noisy = clipped + np.random.laplace(0, noise_scale)
        return int(np.clip(noisy * 10, 0, 10))


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sanitizer
    sanitizer = ProfileSanitizer()
    
    # Example raw profile (what client might send)
    raw_profile = {
        'model_size_mb': 160,
        'architecture_type': 'decoder',
        'performance': 0.75,
        'trust_score': 0.85,
        'specialization_score': 0.6,
        'collaboration_quality': 0.7,
        'code_capability': 0.8,
        'reasoning_capability': 0.7,
        'language_capability': 0.9,
        'math_capability': 0.6,
        'adaptation_speed': 0.8,
        'generalization_ability': 0.7,
        'knowledge_retention': 0.8,
        'transfer_efficiency': 0.6,
        'exchange_stats': {'successful': 5, 'total': 8}
    }
    
    # Sanitize profile
    abstract_profile = sanitizer.sanitize_profile(raw_profile, "client_001")
    
    print(f"Abstract Profile ID: {abstract_profile.profile_id}")
    print(f"Model Size Category: {abstract_profile.model_size_category}")
    print(f"Architecture Type: {abstract_profile.architecture_type}")
    print(f"Performance Category: {abstract_profile.performance_category}")
    print(f"Trust Score (quantized): {abstract_profile.trust_score_quantized}")
    print(f"Capability Vector: {abstract_profile.capability_vector}")
    print(f"Profile Vector: {abstract_profile.to_vector()}")
    
    # Test performance feedback sanitization
    feedback_sanitizer = PerformanceFeedbackSanitizer()
    
    raw_feedback = {
        'performance_delta': 0.05,
        'exchange_success': True,
        'trust_change': 0.1,
        'collaboration_quality': 0.8,
        'detailed_logs': 'sensitive training information',  # Should be excluded
        'model_parameters': [1, 2, 3]  # Should be excluded
    }
    
    sanitized_feedback = feedback_sanitizer.sanitize_performance_feedback(raw_feedback)
    print(f"Sanitized Feedback: {sanitized_feedback}")
    
    # Privacy metrics
    privacy_metrics = sanitizer.get_privacy_metrics()
    print(f"Privacy Metrics: {privacy_metrics}")
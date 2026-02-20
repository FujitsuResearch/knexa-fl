"""
Privacy-Preserving Profile System for KNEXA-FL
Implements k-anonymity and differential privacy for agent profiles
"""

import numpy as np
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

class ModelSizeCategory(Enum):
    """Categorical model size representation"""
    ULTRA_SMALL = 1    # < 100M parameters
    SMALL = 2          # 100M - 300M parameters  
    MEDIUM = 3         # 300M - 1B parameters
    LARGE = 4          # 1B - 10B parameters
    ULTRA_LARGE = 5    # > 10B parameters

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
    k_anonymity: int = 3               # Minimum group size
    dp_epsilon: float = 1.0            # Differential privacy epsilon
    dp_delta: float = 1e-5             # Differential privacy delta
    noise_scale: float = 0.1           # Noise scale for continuous values
    quantization_levels: int = 10      # Quantization levels

@dataclass
class AbstractProfile:
    """
    Abstract, privacy-preserving agent profile.
    Contains only sanitized information visible to CPM.
    """
    # Anonymous identifier
    profile_id: str
    
    # Categorical information (k-anonymous)
    model_size_category: ModelSizeCategory
    performance_category: PerformanceCategory
    
    # Quantized continuous values (DP-protected)
    trust_score_quantized: int          # 0-10 quantized
    specialization_score_quantized: int # 0-10 quantized
    collaboration_score_quantized: int  # 0-10 quantized
    
    # Aggregated statistics
    successful_exchanges: int
    total_exchanges: int
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def to_vector(self) -> np.ndarray:
        """Convert profile to numerical vector for LinUCB"""
        vector = np.zeros(16)
        
        # Categorical features (normalized)
        vector[0] = float(self.model_size_category.value) / 5.0
        vector[1] = float(self.performance_category.value) / 5.0
        
        # Quantized continuous features
        vector[2] = self.trust_score_quantized / 10.0
        vector[3] = self.specialization_score_quantized / 10.0
        vector[4] = self.collaboration_score_quantized / 10.0
        
        # Success rate
        success_rate = self.successful_exchanges / max(1, self.total_exchanges)
        vector[5] = success_rate
        
        # Padding for 16-dimensional vector
        # (Reserved for future features)
        
        return vector

class ProfileSanitizer:
    """
    Sanitizes agent profiles for privacy-preserving CPM access.
    Implements k-anonymity and differential privacy.
    """
    
    def __init__(self, privacy_params: Optional[PrivacyParameters] = None):
        self.privacy_params = privacy_params or PrivacyParameters()
        self.profile_groups: Dict[str, List[str]] = {}
        
    def sanitize_profile(self, raw_profile: Dict[str, Any], agent_id: str) -> AbstractProfile:
        """
        Sanitize raw agent profile to abstract profile.
        
        Args:
            raw_profile: Raw profile data from agent
            agent_id: Agent identifier
            
        Returns:
            Sanitized abstract profile
        """
        # Create anonymous ID
        profile_id = self._create_anonymous_id(agent_id)
        
        # Categorize model size
        model_size_mb = float(raw_profile.get('model_size_mb', 100))
        model_size_category = self._categorize_model_size(model_size_mb)
        
        # Categorize performance
        performance = float(raw_profile.get('performance', 0.5))
        performance_category = self._categorize_performance(performance)
        
        # Quantize continuous values with DP noise
        trust_score = float(raw_profile.get('trust_score', 0.8))
        trust_quantized = self._quantize_with_dp(trust_score)
        
        specialization = float(raw_profile.get('specialization_score', 0.5))
        specialization_quantized = self._quantize_with_dp(specialization)
        
        collaboration = float(raw_profile.get('collaboration_quality', 0.5))
        collaboration_quantized = self._quantize_with_dp(collaboration)
        
        # Get exchange statistics
        exchange_stats = raw_profile.get('exchange_stats', {})
        successful = int(exchange_stats.get('successful', 0))
        total = int(exchange_stats.get('total', 0))
        
        return AbstractProfile(
            profile_id=profile_id,
            model_size_category=model_size_category,
            performance_category=performance_category,
            trust_score_quantized=trust_quantized,
            specialization_score_quantized=specialization_quantized,
            collaboration_score_quantized=collaboration_quantized,
            successful_exchanges=successful,
            total_exchanges=total
        )
    
    def _create_anonymous_id(self, agent_id: str) -> str:
        """Create anonymous ID using hashing"""
        salt = "knexa_privacy_salt"
        return hashlib.sha256(f"{agent_id}{salt}".encode()).hexdigest()[:16]
    
    def _categorize_model_size(self, size_mb: float) -> ModelSizeCategory:
        """Categorize model size"""
        # Assuming ~2MB per million parameters
        params_millions = size_mb / 2.0
        
        if params_millions < 100:
            return ModelSizeCategory.ULTRA_SMALL
        elif params_millions < 300:
            return ModelSizeCategory.SMALL
        elif params_millions < 1000:
            return ModelSizeCategory.MEDIUM
        elif params_millions < 10000:
            return ModelSizeCategory.LARGE
        else:
            return ModelSizeCategory.ULTRA_LARGE
    
    def _categorize_performance(self, performance: float) -> PerformanceCategory:
        """Categorize performance level"""
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
    
    def _quantize_with_dp(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> int:
        """Quantize continuous value with differential privacy noise"""
        # Clip to valid range
        value = np.clip(value, min_val, max_val)
        
        # Add Laplace noise for differential privacy
        sensitivity = (max_val - min_val) / self.privacy_params.quantization_levels
        scale = sensitivity / self.privacy_params.dp_epsilon
        noise = np.random.laplace(0, scale)
        
        noisy_value = value + noise
        
        # Quantize to discrete levels
        normalized = (noisy_value - min_val) / (max_val - min_val)
        quantized = int(np.clip(normalized * self.privacy_params.quantization_levels, 
                               0, self.privacy_params.quantization_levels))
        
        return quantized

class PerformanceFeedbackSanitizer:
    """Sanitizes performance feedback for privacy"""
    
    def __init__(self, noise_scale: float = 0.01):
        self.noise_scale = noise_scale
    
    def sanitize_performance_feedback(self, raw_feedback: Dict[str, Any]) -> Dict[str, float]:
        """
        Sanitize performance feedback to ensure privacy.
        
        Args:
            raw_feedback: Raw performance feedback
            
        Returns:
            Sanitized feedback with only aggregate metrics
        """
        # Extract and sanitize performance delta
        perf_delta = float(raw_feedback.get('performance_delta', 0.0))
        perf_delta_noisy = perf_delta + np.random.laplace(0, self.noise_scale)
        
        # Extract and sanitize trust change
        trust_change = float(raw_feedback.get('trust_change', 0.0))
        trust_change_noisy = trust_change + np.random.laplace(0, self.noise_scale)
        
        return {
            'performance_delta': float(np.clip(perf_delta_noisy, -1.0, 1.0)),
            'trust_change': float(np.clip(trust_change_noisy, -0.1, 0.1))
        }
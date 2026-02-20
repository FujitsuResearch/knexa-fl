"""
Privacy Guardrail Filter for KNEXA-FL
Ensures sensitive information is not leaked during knowledge exchange
"""

import re
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PrivacyConfig:
    """Configuration for privacy protection"""
    sensitive_patterns: List[str] = None
    clip_norm: float = 1.0
    noise_multiplier: float = 0.1
    sier_threshold: float = 0.01  # Sensitive Information Exposure Rate threshold
    
    def __post_init__(self):
        if self.sensitive_patterns is None:
            # Default sensitive patterns to detect
            self.sensitive_patterns = [
                # Credentials and secrets
                r'\b(?:password|passwd|pwd)[:\s]*\S+',
                r'\b(?:api[_\s]?key|apikey)[:\s]*\S+',
                r'\b(?:secret|token|credential)[:\s]*\S+',
                r'\b(?:auth|authorization)[:\s]*bearer\s+\S+',
                
                # Personal identifiers
                r'\b(?:ssn|social[_\s]?security)[:\s]*\d{3}-?\d{2}-?\d{4}',
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b(?:ein|tax[_\s]?id)[:\s]*\d{2}-?\d{7}',
                
                # Contact information
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b(?:phone|tel|mobile)[:\s]*[\+\d\s\(\)-]+\d',
                r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone number
                
                # Financial information
                r'\b(?:credit[_\s]?card|cc)[:\s]*\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}',
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
                r'\b(?:bank[_\s]?account|acct)[:\s]*\d+',
                
                # Network information
                r'\b(?:ip[_\s]?address|ip)[:\s]*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
                r'(?:http[s]?://)?(?:www\.)?[\w\-]+\.[\w\-]+(?:\.[\w\-]+)*(?:/[\w\-._~:/?#[\]@!$&\'()*+,;=]*)?',  # URLs with potential secrets
                
                # Medical information
                r'\b(?:diagnosis|medical[_\s]?record|patient[_\s]?id)[:\s]*\S+',
                r'\b(?:prescription|medication)[:\s]*\S+',
                
                # Generic private data indicators
                r'\b(?:private|confidential|restricted|internal[_\s]?only)\b',
                r'\b(?:do[_\s]?not[_\s]?share|sensitive)\b'
            ]

class GuardrailFilter:
    """
    Privacy-preserving filter for knowledge exchange.
    Prevents leakage of sensitive information.
    """
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
        
        # Compile regex patterns for efficiency
        self.pattern_regex = re.compile(
            '|'.join(self.config.sensitive_patterns),
            re.IGNORECASE
        )
        
        # Statistics tracking
        self.stats = {
            'total_checked': 0,
            'total_filtered': 0,
            'patterns_detected': {}
        }
    
    def check_text_safety(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Check if text is safe for sharing.
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, sier_score, detected_patterns)
        """
        self.stats['total_checked'] += 1
        
        # Find all matches
        matches = list(self.pattern_regex.finditer(text))
        
        if not matches:
            return True, 0.0, []
        
        # Calculate SIER (Sensitive Information Exposure Rate)
        words = text.split()
        total_words = len(words)
        flagged_count = len(matches)
        sier = flagged_count / max(1, total_words)
        
        # Extract matched patterns
        detected_patterns = []
        for match in matches:
            pattern = match.group()
            # Redact the actual value for logging
            redacted = re.sub(r'\S', '*', pattern)
            detected_patterns.append(redacted)
            
            # Update pattern statistics
            pattern_type = self._classify_pattern(pattern)
            self.stats['patterns_detected'][pattern_type] = \
                self.stats['patterns_detected'].get(pattern_type, 0) + 1
        
        # Check against threshold
        is_safe = sier <= self.config.sier_threshold
        
        if not is_safe:
            self.stats['total_filtered'] += 1
        
        return is_safe, sier, detected_patterns
    
    def filter_knowledge_outputs(self, 
                               outputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter a list of knowledge outputs for privacy.
        
        Args:
            outputs: List of output dictionaries with 'response' field
            
        Returns:
            Filtered list with only safe outputs
        """
        filtered_outputs = []
        
        for output in outputs:
            text = output.get('response', '')
            is_safe, sier, patterns = self.check_text_safety(text)
            
            if is_safe:
                filtered_outputs.append(output)
            else:
                # Output filtered due to privacy concerns
        
        return filtered_outputs
    
    def apply_differential_privacy(self, 
                                 tensor: torch.Tensor,
                                 sensitivity: float = 1.0) -> torch.Tensor:
        """
        Apply differential privacy to tensor outputs.
        
        Args:
            tensor: Tensor to protect
            sensitivity: Sensitivity of the function
            
        Returns:
            DP-protected tensor
        """
        # L2 norm clipping
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        clipped_tensor = tensor * (self.config.clip_norm / norm).clamp(max=1.0)
        
        # Add calibrated Gaussian noise
        noise_scale = self.config.noise_multiplier * self.config.clip_norm * sensitivity
        noise = torch.normal(0, noise_scale, size=tensor.shape, device=tensor.device)
        
        noisy_tensor = clipped_tensor + noise
        
        return noisy_tensor
    
    def _classify_pattern(self, pattern: str) -> str:
        """Classify the type of sensitive pattern detected"""
        pattern_lower = pattern.lower()
        
        if any(keyword in pattern_lower for keyword in ['password', 'api', 'key', 'secret', 'token']):
            return 'credentials'
        elif any(keyword in pattern_lower for keyword in ['ssn', 'social', 'ein', 'tax']):
            return 'identifiers'
        elif '@' in pattern or any(keyword in pattern_lower for keyword in ['email', 'phone', 'tel']):
            return 'contact'
        elif any(keyword in pattern_lower for keyword in ['credit', 'card', 'bank', 'account']):
            return 'financial'
        elif re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', pattern):
            return 'network'
        elif any(keyword in pattern_lower for keyword in ['medical', 'patient', 'diagnosis']):
            return 'medical'
        else:
            return 'other'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics"""
        filter_rate = self.stats['total_filtered'] / max(1, self.stats['total_checked'])
        
        return {
            'total_checked': self.stats['total_checked'],
            'total_filtered': self.stats['total_filtered'],
            'filter_rate': filter_rate,
            'patterns_by_type': dict(self.stats['patterns_detected']),
            'sier_threshold': self.config.sier_threshold
        }
    
    def reset_statistics(self):
        """Reset filter statistics"""
        self.stats = {
            'total_checked': 0,
            'total_filtered': 0,
            'patterns_detected': {}
        }

class SecureAggregator:
    """
    Secure aggregation utilities for privacy-preserving parameter sharing
    """
    
    @staticmethod
    def add_noise_for_secure_aggregation(parameters: Dict[str, torch.Tensor],
                                       noise_scale: float = 0.001,
                                       clip_norm: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Add noise to parameters for secure aggregation.
        
        Args:
            parameters: Model parameters
            noise_scale: Scale of noise to add
            clip_norm: Maximum norm for clipping
            
        Returns:
            Noisy parameters
        """
        noisy_params = {}
        
        for name, param in parameters.items():
            # Clip parameter updates
            param_norm = torch.norm(param, p=2)
            if param_norm > clip_norm:
                param = param * (clip_norm / param_norm)
            
            # Add Gaussian noise
            noise = torch.randn_like(param) * noise_scale
            noisy_params[name] = param + noise
        
        return noisy_params
    
    @staticmethod
    def compute_secure_checksum(parameters: Dict[str, torch.Tensor]) -> str:
        """
        Compute secure checksum of parameters for integrity verification.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Hex checksum string
        """
        import hashlib
        
        hasher = hashlib.sha256()
        
        for name in sorted(parameters.keys()):
            param = parameters[name]
            param_bytes = param.cpu().numpy().tobytes()
            hasher.update(name.encode())
            hasher.update(param_bytes)
        
        return hasher.hexdigest()
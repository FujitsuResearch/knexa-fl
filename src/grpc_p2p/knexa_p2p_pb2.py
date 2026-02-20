#!/usr/bin/env python3
"""
Generated gRPC message classes for KNEXA-FL P2P system
Simple manual implementation to avoid protobuf version conflicts
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import time

# Message classes for P2P communication

@dataclass
class ChannelEstablishRequest:
    peer_id: str
    public_key: str
    timestamp: int
    nonce: str
    signature: str

@dataclass
class ChannelEstablishResponse:
    success: bool
    public_key: str
    session_id: str
    expiry: int
    error_message: str
    signature: str

@dataclass
class KnowledgeDistillationRequest:
    session_id: str
    sender_id: str
    receiver_id: str
    round_id: int
    encrypted_logits: bytes
    encryption_nonce: bytes
    query_ids: List[str]
    temperature: float
    alpha: float
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    sier_score: float = 0.0

@dataclass
class KnowledgeDistillationResponse:
    success: bool
    session_id: str
    processed_queries: int
    validation_loss: float
    error_message: str

@dataclass
class PEFTModuleRequest:
    session_id: str
    sender_id: str
    receiver_id: str
    round_id: int
    encrypted_delta: bytes
    encryption_nonce: bytes
    module_type: str
    rank: int
    target_modules: List[str]
    lambda_weight: float
    transformation_params: bytes = b''
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5

@dataclass
class PEFTModuleResponse:
    success: bool
    session_id: str
    integration_loss: float
    parameters_updated: int
    error_message: str

@dataclass
class HeartbeatRequest:
    peer_id: str
    timestamp: int
    status: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0

@dataclass
class HeartbeatResponse:
    alive: bool
    timestamp: int
    cpm_status: str

@dataclass
class PerformanceReport:
    session_id: str
    peer_id: str
    round_id: int
    pre_performance: float
    post_performance: float
    delta_performance: float
    transfer_time: float
    bytes_transferred: int
    compression_ratio: float
    trust_score: float
    collaboration_quality: float
    sier_exposure: float

@dataclass
class PerformanceAck:
    acknowledged: bool
    timestamp: int

# CPM Service Messages

@dataclass
class PeerRegistration:
    peer_id: str
    endpoint: str
    public_key: str
    model_family: str
    model_name: str
    model_size_mb: int
    architecture_type: str
    supported_modules: List[str]
    device_info: str
    available_memory: float
    compute_capacity: float
    timestamp: int

@dataclass
class PeerRegistrationResponse:
    success: bool
    peer_id: str
    error_message: str
    policies: List[str]
    heartbeat_interval: int

@dataclass
class ProfileUpdate:
    peer_id: str
    round_id: int
    context_vector: List[float]
    local_performance: float
    transfer_performance: float
    global_performance: float
    trust_score: float
    learning_rate: float
    collaboration_quality: float
    communication_efficiency: float
    data_distribution: Dict[str, float]
    difficulty_distribution: Dict[str, float]
    specialization_score: float
    performance_trend: str
    recent_collaborations: List[str]
    timestamp: int

@dataclass
class ProfileUpdateResponse:
    success: bool
    error_message: str
    timestamp: int

@dataclass
class MatchingRequest:
    available_peers: List[str]
    round_id: int
    max_pairs: int
    system_constraints: Dict[str, float]
    timestamp: int

@dataclass
class MatchingResponse:
    success: bool
    pairings: List['PeerPairing']
    error_message: str
    timestamp: int

@dataclass
class PeerPairing:
    student_id: str
    teacher_id: str
    exchange_type: str
    alpha: float
    temperature: float
    query_samples: List[str]
    lambda_weight: float
    transformation_params: bytes
    expected_reward: float
    confidence_score: float
    priority: int

@dataclass
class TrustUpdate:
    peer_id: str
    partner_id: str
    trust_delta: float
    interaction_type: str
    performance_gain: float
    reason: str
    timestamp: int

@dataclass
class TrustUpdateResponse:
    success: bool
    new_trust_score: float
    error_message: str

@dataclass
class PolicyRequest:
    peer_id: str
    policy_type: str
    policy_data: bytes
    action: str
    timestamp: int

@dataclass
class PolicyResponse:
    compliant: bool
    action_taken: bool
    violation_details: str
    recommendations: List[str]
    error_message: str

@dataclass
class PeerDiscoveryRequest:
    requester_id: str
    capabilities_needed: List[str]
    model_family_preference: str
    min_trust_score: float
    max_results: int
    timestamp: int

@dataclass
class PeerDiscoveryResponse:
    success: bool
    available_peers: List['PeerInfo']
    error_message: str
    timestamp: int

@dataclass
class PeerInfo:
    peer_id: str
    endpoint: str
    model_family: str
    model_name: str
    trust_score: float
    capabilities: List[str]
    status: str
    performance_rating: float
    last_seen: int
#!/usr/bin/env python3
"""
Central Profiler/Matchmaker (CPM) Service for KNEXA-FL
Non-aggregating orchestrator for P2P peer matching and policy enforcement
"""

import asyncio
import time
import threading
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import grpc
from grpc import aio

from .service_registry import ServiceRegistry, PeerInfo
from .privacy_profile import ProfileSanitizer, PerformanceFeedbackSanitizer, AbstractProfile, PrivacyParameters
from ..bandit import LinUCB
from . import knexa_p2p_pb2 as pb2
from . import knexa_p2p_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)

@dataclass
class PeerPairing:
    """Information about a peer pairing decision"""
    student_id: str
    teacher_id: str
    exchange_type: str  # "knowledge_distillation", "peft_exchange"
    
    # Knowledge Distillation parameters
    alpha: float = 0.5
    temperature: float = 2.0
    query_samples: List[str] = field(default_factory=list)
    
    # PEFT Exchange parameters
    lambda_weight: float = 0.1
    transformation_params: bytes = b''
    
    # Quality metrics
    expected_reward: float = 0.0
    confidence_score: float = 0.0
    priority: int = 1
    
    # Metadata
    round_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PolicyRule:
    """Policy rule for governance"""
    rule_id: str
    rule_type: str  # "privacy", "security", "governance"
    condition: str
    action: str
    priority: int = 1
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class CPMService(pb2_grpc.CPMServiceServicer):
    """
    Central Profiler/Matchmaker Service - CONTENT-BLIND VERSION
    Non-aggregating orchestrator that only sees abstract, sanitized profiles
    
    PRIVACY GUARANTEES:
    - NEVER has access to knowledge transfer content (logits, PEFT deltas)
    - NEVER has access to detailed training logs or metrics
    - NEVER has access to raw model parameters or gradients
    - ONLY sees k-anonymous, DP-protected abstract profiles
    - ONLY receives aggregate performance deltas as feedback
    """
    
    def __init__(self, cpm_id: str = "cpm_main", endpoint: str = "localhost:8000"):
        self.cpm_id = cpm_id
        self.endpoint = endpoint
        
        # Core components - PRIVACY-PRESERVING
        self.service_registry = ServiceRegistry(f"registry_{cpm_id}")
        
        # Privacy-preserving profile management
        self.profile_sanitizer = ProfileSanitizer()
        self.feedback_sanitizer = PerformanceFeedbackSanitizer()
        
        # Abstract profile storage (NO raw client data)
        self.abstract_profiles: Dict[str, AbstractProfile] = {}
        self.profile_lock = threading.RLock()
        
        # LinUCB bandit for abstract profile matching
        self.bandit = LinUCB(d=32)  # 16*2 for pairwise abstract vectors
        
        # Policy management (high-level only)
        self.policies: Dict[str, PolicyRule] = {}
        self.policy_lock = threading.RLock()
        
        # Performance tracking (AGGREGATE METRICS ONLY)
        self.matching_history: List[Dict[str, Any]] = []
        self.aggregate_feedback: Dict[str, List[float]] = {}  # Only performance deltas
        
        # Configuration
        self.max_pairs_per_round = 10
        self.min_trust_score = 0.3
        self.matching_timeout = 30.0  # seconds
        
        # Privacy parameters
        self.privacy_params = PrivacyParameters(
            k_anonymity=3,
            dp_epsilon=1.0,
            dp_delta=1e-5
        )
        
        # gRPC server
        self.server = None
        self.running = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=16)
        
        logger.info(f"CONTENT-BLIND CPM Service initialized: {cpm_id} at {endpoint}")
        logger.info(f"Privacy parameters: k={self.privacy_params.k_anonymity}, ε={self.privacy_params.dp_epsilon}")
        logger.info("⚠️  CPM has ZERO access to knowledge transfer content or detailed training data")
    
    async def start_server(self):
        """Start the CPM gRPC server"""
        self.server = aio.server(self.executor)
        
        # Add service implementations
        pb2_grpc.add_CPMServiceServicer_to_server(self, self.server)
        
        listen_addr = self.endpoint
        self.server.add_insecure_port(listen_addr)
        
        await self.server.start()
        self.running = True
        
        logger.info(f"CPM Service started on {listen_addr}")
        
        # Start background tasks
        asyncio.create_task(self._performance_monitoring_loop())
        asyncio.create_task(self._policy_enforcement_loop())
        asyncio.create_task(self._bandit_update_loop())
    
    async def stop_server(self):
        """Stop the CPM gRPC server"""
        if self.server:
            await self.server.stop(grace=5)
            self.running = False
            logger.info("CPM Service stopped")
    
    # gRPC Service Implementation
    async def RegisterPeer(self, request: pb2.PeerRegistration, context) -> pb2.PeerRegistrationResponse:
        """
        Register a new peer in the service registry
        """
        try:
            # Create PeerInfo from request
            peer_info = PeerInfo(
                peer_id=request.peer_id,
                endpoint=request.endpoint,
                public_key=request.public_key,
                model_family=request.model_family,
                model_name=request.model_name,
                model_size_mb=request.model_size_mb,
                architecture_type=request.architecture_type,
                supported_modules=list(request.supported_modules),
                device_info=request.device_info,
                available_memory=request.available_memory,
                compute_capacity=request.compute_capacity
            )
            
            # Register peer
            success = self.service_registry.register_peer(peer_info)
            
            if success:
                # Initialize client profile
                self._initialize_client_profile(peer_info)
                
                # Get applicable policies
                policies = self._get_applicable_policies(peer_info)
                
                response = pb2.PeerRegistrationResponse(
                    success=True,
                    peer_id=request.peer_id,
                    error_message='',
                    policies=policies,
                    heartbeat_interval=30
                )
                
                logger.info(f"Peer registered successfully: {request.peer_id}")
            else:
                response = pb2.PeerRegistrationResponse(
                    success=False,
                    peer_id=request.peer_id,
                    error_message='Registration failed',
                    policies=[],
                    heartbeat_interval=30
                )
                
                logger.error(f"Failed to register peer: {request.peer_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RegisterPeer: {e}")
            return pb2.PeerRegistrationResponse(
                success=False,
                peer_id=request.peer_id,
                error_message=str(e),
                policies=[],
                heartbeat_interval=30
            )
    
    async def UpdateProfile(self, request: pb2.ProfileUpdate, context) -> pb2.ProfileUpdateResponse:
        """
        Update peer profile information - CONTENT-BLIND VERSION
        Only accepts sanitized, abstract profile information
        """
        try:
            peer_id = request.peer_id
            round_id = request.round_id
            
            # PRIVACY-PRESERVING: Extract only minimal, sanitized information
            raw_profile = {
                'model_size_mb': request.global_performance * 1000,  # Simulate model size from performance
                'architecture_type': 'decoder',  # Default architecture
                'performance': request.global_performance,
                'trust_score': request.trust_score,
                'specialization_score': request.specialization_score,
                'collaboration_quality': request.collaboration_quality,
                'exchange_stats': {
                    'successful': len([c for c in request.recent_collaborations if c]),
                    'total': len(request.recent_collaborations)
                }
            }
            
            # CRITICAL: Sanitize the profile to ensure privacy
            abstract_profile = self.profile_sanitizer.sanitize_profile(raw_profile, peer_id)
            
            # Store only the abstract profile
            with self.profile_lock:
                self.abstract_profiles[peer_id] = abstract_profile
            
            # Update service registry with minimal information
            peer_info = self.service_registry.get_peer_info(peer_id)
            if peer_info:
                peer_info.last_heartbeat = time.time()
                peer_info.status = "online"
            
            response = pb2.ProfileUpdateResponse(
                success=True,
                error_message='',
                timestamp=int(time.time())
            )
            
            logger.info(f"Abstract profile updated for peer {peer_id}: category={abstract_profile.performance_category}")
            logger.info(f"Privacy preserved: k-anonymity={self.privacy_params.k_anonymity}, DP-protected")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in UpdateProfile: {e}")
            return pb2.ProfileUpdateResponse(
                success=False,
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    async def RequestMatching(self, request: pb2.MatchingRequest, context) -> pb2.MatchingResponse:
        """
        Request peer matching using LinUCB bandit
        """
        try:
            available_peers = list(request.available_peers)
            round_id = request.round_id
            max_pairs = min(request.max_pairs, self.max_pairs_per_round)
            
            logger.info(f"Matching request for round {round_id}: {len(available_peers)} peers, max {max_pairs} pairs")
            
            # Perform intelligent matching
            pairings = await self._perform_intelligent_matching(
                available_peers, round_id, max_pairs
            )
            
            # Convert to response format
            response_pairings = []
            for pairing in pairings:
                pb_pairing = pb2.PeerPairing(
                    student_id=pairing.student_id,
                    teacher_id=pairing.teacher_id,
                    exchange_type=pairing.exchange_type,
                    alpha=pairing.alpha,
                    temperature=pairing.temperature,
                    query_samples=pairing.query_samples,
                    lambda_weight=pairing.lambda_weight,
                    transformation_params=pairing.transformation_params,
                    expected_reward=pairing.expected_reward,
                    confidence_score=pairing.confidence_score,
                    priority=pairing.priority
                )
                response_pairings.append(pb_pairing)
            
            response = pb2.MatchingResponse(
                success=True,
                pairings=response_pairings,
                error_message='',
                timestamp=int(time.time())
            )
            
            logger.info(f"Generated {len(pairings)} peer pairings for round {round_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in RequestMatching: {e}")
            return pb2.MatchingResponse(
                success=False,
                pairings=[],
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    async def UpdateTrustScore(self, request: pb2.TrustUpdate, context) -> pb2.TrustUpdateResponse:
        """
        Update trust score for a peer - CONTENT-BLIND VERSION
        Only accepts sanitized performance feedback
        """
        try:
            peer_id = request.peer_id
            partner_id = request.partner_id
            
            # PRIVACY-PRESERVING: Sanitize the feedback first
            raw_feedback = {
                'performance_delta': request.performance_gain,
                'trust_change': request.trust_delta,
                'exchange_success': request.trust_delta > 0,
                'collaboration_quality': 0.8 if request.trust_delta > 0 else 0.4
            }
            
            # CRITICAL: Sanitize feedback to ensure no sensitive information
            sanitized_feedback = self.feedback_sanitizer.sanitize_performance_feedback(raw_feedback)
            
            # Store only aggregate feedback
            if peer_id not in self.aggregate_feedback:
                self.aggregate_feedback[peer_id] = []
            
            self.aggregate_feedback[peer_id].append(sanitized_feedback['performance_delta'])
            
            # Update LinUCB bandit with sanitized reward
            if peer_id in self.abstract_profiles and partner_id in self.abstract_profiles:
                profile_i = self.abstract_profiles[peer_id]
                profile_j = self.abstract_profiles[partner_id]
                
                # Build pairwise context from abstract profiles
                context_i = profile_i.to_vector()
                context_j = profile_j.to_vector()
                pairwise_context = np.concatenate([context_i, context_j])
                
                # Update bandit with sanitized reward
                reward = sanitized_feedback['performance_delta']
                self.bandit.update(pairwise_context, reward, len(self.aggregate_feedback[peer_id]))
            
            # Update abstract profile trust score
            if peer_id in self.abstract_profiles:
                abstract_profile = self.abstract_profiles[peer_id]
                # Update quantized trust score
                trust_change = sanitized_feedback['trust_change']
                new_trust = abstract_profile.trust_score_quantized + int(trust_change * 10)
                abstract_profile.trust_score_quantized = max(0, min(10, new_trust))
                
                response = pb2.TrustUpdateResponse(
                    success=True,
                    new_trust_score=abstract_profile.trust_score_quantized / 10.0,
                    error_message=''
                )
                
                logger.info(f"Trust score updated for {peer_id}: Δ={trust_change:.3f} (sanitized)")
                logger.info(f"PRIVACY: Only aggregate performance delta stored: {sanitized_feedback['performance_delta']:.4f}")
            else:
                response = pb2.TrustUpdateResponse(
                    success=False,
                    new_trust_score=0.0,
                    error_message=f'Abstract profile for {peer_id} not found'
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in UpdateTrustScore: {e}")
            return pb2.TrustUpdateResponse(
                success=False,
                new_trust_score=0.0,
                error_message=str(e)
            )
    
    async def EnforcePolicy(self, request: pb2.PolicyRequest, context) -> pb2.PolicyResponse:
        """
        Enforce policy rules
        """
        try:
            peer_id = request.peer_id
            policy_type = request.policy_type
            policy_data = request.policy_data
            action = request.action
            
            # Get applicable policies
            applicable_policies = self._get_policies_by_type(policy_type)
            
            compliant = True
            violations = []
            recommendations = []
            
            # Check compliance
            for policy in applicable_policies:
                if policy.enabled:
                    policy_compliant, violation_details = self._check_policy_compliance(
                        peer_id, policy, policy_data
                    )
                    
                    if not policy_compliant:
                        compliant = False
                        violations.append(violation_details)
                        recommendations.extend(self._get_policy_recommendations(policy))
            
            # Take action if not compliant
            action_taken = False
            if not compliant and action == "enforce":
                action_taken = self._take_policy_action(peer_id, violations)
            
            response = pb2.PolicyResponse(
                compliant=compliant,
                action_taken=action_taken,
                violation_details='; '.join(violations),
                recommendations=recommendations,
                error_message=''
            )
            
            logger.info(f"Policy enforcement for {peer_id}: compliant={compliant}, action_taken={action_taken}")
            return response
            
        except Exception as e:
            logger.error(f"Error in EnforcePolicy: {e}")
            return pb2.PolicyResponse(
                compliant=False,
                action_taken=False,
                violation_details=str(e),
                recommendations=[],
                error_message=str(e)
            )
    
    async def DiscoverPeers(self, request: pb2.PeerDiscoveryRequest, context) -> pb2.PeerDiscoveryResponse:
        """
        Discover available peers based on requirements
        """
        try:
            requester_id = request.requester_id
            capabilities_needed = list(request.capabilities_needed)
            model_family_preference = request.model_family_preference
            min_trust_score = request.min_trust_score
            max_results = request.max_results
            
            # Discover peers
            discovered_peers = self.service_registry.discover_peers(
                requester_id=requester_id,
                capabilities_needed=capabilities_needed,
                model_family_preference=model_family_preference,
                min_trust_score=min_trust_score,
                max_results=max_results
            )
            
            # Convert to response format
            peer_infos = []
            for peer in discovered_peers:
                peer_info = pb2.PeerInfo(
                    peer_id=peer.peer_id,
                    endpoint=peer.endpoint,
                    model_family=peer.model_family,
                    model_name=peer.model_name,
                    trust_score=peer.trust_score,
                    capabilities=peer.supported_modules,
                    status=peer.status,
                    performance_rating=peer.performance_rating,
                    last_seen=int(peer.last_heartbeat)
                )
                peer_infos.append(peer_info)
            
            response = pb2.PeerDiscoveryResponse(
                success=True,
                available_peers=peer_infos,
                error_message='',
                timestamp=int(time.time())
            )
            
            logger.info(f"Discovered {len(peer_infos)} peers for requester {requester_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in DiscoverPeers: {e}")
            return pb2.PeerDiscoveryResponse(
                success=False,
                available_peers=[],
                error_message=str(e),
                timestamp=int(time.time())
            )
    
    # Core matching logic
    async def _perform_intelligent_matching(self, available_peers: List[str], 
                                          round_id: int, max_pairs: int) -> List[PeerPairing]:
        """
        Perform intelligent peer matching using LinUCB bandit - CONTENT-BLIND VERSION
        Uses only abstract profiles, no access to knowledge content
        """
        try:
            pairings = []
            used_peers = set()
            
            # Generate all possible pairings using ONLY abstract profiles
            candidate_pairings = []
            
            with self.profile_lock:
                for i, student_id in enumerate(available_peers):
                    for j, teacher_id in enumerate(available_peers):
                        if (i != j and student_id not in used_peers and teacher_id not in used_peers and
                            student_id in self.abstract_profiles and teacher_id in self.abstract_profiles):
                            
                            # Build pairwise context from ABSTRACT profiles only
                            student_profile = self.abstract_profiles[student_id]
                            teacher_profile = self.abstract_profiles[teacher_id]
                            
                            # Create pairwise context vector
                            context_student = student_profile.to_vector()
                            context_teacher = teacher_profile.to_vector()
                            pairwise_context = np.concatenate([context_student, context_teacher])
                            
                            # Get UCB score from bandit
                            ucb_score = self.bandit.get_ucb_score(pairwise_context)
                            
                            # Determine exchange type based on abstract compatibility
                            exchange_type = self._determine_exchange_type_abstract(student_profile, teacher_profile)
                            
                            # Generate parameters from abstract profiles
                            alpha, temperature = self._generate_kd_parameters_abstract(student_profile, teacher_profile)
                            lambda_weight = self._generate_peft_parameters_abstract(student_profile, teacher_profile)
                            
                            candidate_pairings.append({
                                'student_id': student_id,
                                'teacher_id': teacher_id,
                                'exchange_type': exchange_type,
                                'context': pairwise_context,
                                'ucb_score': ucb_score,
                                'alpha': alpha,
                                'temperature': temperature,
                                'lambda_weight': lambda_weight
                            })
            
            # Sort by UCB score
            candidate_pairings.sort(key=lambda x: x['ucb_score'], reverse=True)
            
            # Select top pairings
            for candidate in candidate_pairings:
                if len(pairings) >= max_pairs:
                    break
                
                student_id = candidate['student_id']
                teacher_id = candidate['teacher_id']
                
                if student_id not in used_peers and teacher_id not in used_peers:
                    pairing = PeerPairing(
                        student_id=student_id,
                        teacher_id=teacher_id,
                        exchange_type=candidate['exchange_type'],
                        alpha=candidate['alpha'],
                        temperature=candidate['temperature'],
                        lambda_weight=candidate['lambda_weight'],
                        expected_reward=candidate['ucb_score'],
                        confidence_score=min(1.0, candidate['ucb_score'] / 2.0),
                        priority=len(pairings) + 1,
                        round_id=round_id
                    )
                    
                    pairings.append(pairing)
                    used_peers.add(student_id)
                    used_peers.add(teacher_id)
            
            # Store matching history (no sensitive information)
            matching_record = {
                'round_id': round_id,
                'available_peers': available_peers,
                'num_pairings': len(pairings),
                'avg_ucb_score': np.mean([p.expected_reward for p in pairings]) if pairings else 0.0,
                'timestamp': time.time()
            }
            self.matching_history.append(matching_record)
            
            logger.info(f"PRIVACY: Intelligent matching completed using only abstract profiles")
            logger.info(f"Generated {len(pairings)} pairings with zero knowledge content access")
            
            return pairings
            
        except Exception as e:
            logger.error(f"Error in intelligent matching: {e}")
            return []
    
    def _initialize_client_profile(self, peer_info: PeerInfo):
        """Initialize client profile in context builder"""
        try:
            # Create initial profile data
            three_tier_results = {
                'local_pass@1': 0.0,
                'transfer_pass@1': None,
                'global_pass@1': 0.0
            }
            
            model_info = {
                'family': peer_info.model_family,
                'size_mb': peer_info.model_size_mb,
                'architecture': peer_info.architecture_type
            }
            
            data_info = {
                'type_distribution': {},
                'difficulty_distribution': {'easy': 0.33, 'medium': 0.33, 'hard': 0.34},
                'specialization_score': 0.5
            }
            
            # Initialize profile data (context builder not needed for abstract profiles)
            # Profile will be created when client sends profile update
            
            logger.info(f"Initialized client profile for {peer_info.peer_id}")
            
        except Exception as e:
            logger.error(f"Error initializing client profile for {peer_info.peer_id}: {e}")
    
    def _determine_exchange_type(self, student_id: str, teacher_id: str) -> str:
        """Determine the best exchange type for a peer pair"""
        try:
            student_info = self.service_registry.get_peer_info(student_id)
            teacher_info = self.service_registry.get_peer_info(teacher_id)
            
            if not student_info or not teacher_info:
                return "knowledge_distillation"
            
            # Check if PEFT exchange is possible (compatible architectures)
            if (student_info.architecture_type == teacher_info.architecture_type and
                student_info.model_family == teacher_info.model_family):
                
                # Check if both support compatible PEFT modules
                common_modules = set(student_info.supported_modules) & set(teacher_info.supported_modules)
                if common_modules:
                    return "peft_exchange"
            
            return "knowledge_distillation"
            
        except Exception as e:
            logger.error(f"Error determining exchange type: {e}")
            return "knowledge_distillation"
    
    def _generate_kd_parameters(self, student_id: str, teacher_id: str) -> Tuple[float, float]:
        """Generate knowledge distillation parameters"""
        try:
            student_info = self.service_registry.get_peer_info(student_id)
            teacher_info = self.service_registry.get_peer_info(teacher_id)
            
            if not student_info or not teacher_info:
                return 0.5, 2.0
            
            # Adaptive parameters based on model sizes and performance
            size_ratio = student_info.model_size_mb / teacher_info.model_size_mb
            performance_gap = abs(student_info.performance_rating - teacher_info.performance_rating)
            
            # Alpha: higher for larger performance gaps
            alpha = 0.3 + 0.4 * min(1.0, performance_gap * 2.0)
            
            # Temperature: higher for larger size ratios
            temperature = 1.5 + 1.0 * min(1.0, size_ratio)
            
            return alpha, temperature
            
        except Exception as e:
            logger.error(f"Error generating KD parameters: {e}")
            return 0.5, 2.0
    
    def _generate_peft_parameters(self, student_id: str, teacher_id: str) -> float:
        """Generate PEFT exchange parameters"""
        try:
            student_info = self.service_registry.get_peer_info(student_id)
            teacher_info = self.service_registry.get_peer_info(teacher_id)
            
            if not student_info or not teacher_info:
                return 0.1
            
            # Lambda weight based on trust and performance
            trust_factor = min(student_info.trust_score, teacher_info.trust_score)
            performance_factor = min(student_info.performance_rating, teacher_info.performance_rating)
            
            lambda_weight = 0.05 + 0.15 * (0.6 * trust_factor + 0.4 * performance_factor)
            
            return lambda_weight
            
        except Exception as e:
            logger.error(f"Error generating PEFT parameters: {e}")
            return 0.1
    
    def _get_applicable_policies(self, peer_info: PeerInfo) -> List[str]:
        """Get applicable policies for a peer"""
        policies = []
        
        with self.policy_lock:
            for policy in self.policies.values():
                if policy.enabled:
                    policies.append(f"{policy.rule_type}:{policy.rule_id}")
        
        return policies
    
    def _get_policies_by_type(self, policy_type: str) -> List[PolicyRule]:
        """Get policies by type"""
        with self.policy_lock:
            return [p for p in self.policies.values() if p.rule_type == policy_type and p.enabled]
    
    def _check_policy_compliance(self, peer_id: str, policy: PolicyRule, 
                                policy_data: bytes) -> Tuple[bool, str]:
        """Check if peer complies with policy"""
        try:
            # Basic policy compliance check
            # In a real implementation, this would evaluate the policy condition
            # against the peer's data and behavior
            
            if policy.rule_type == "privacy":
                # Check privacy compliance
                return True, ""
            elif policy.rule_type == "security":
                # Check security compliance
                return True, ""
            elif policy.rule_type == "governance":
                # Check governance compliance
                return True, ""
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error checking policy compliance: {e}")
            return False, str(e)
    
    def _get_policy_recommendations(self, policy: PolicyRule) -> List[str]:
        """Get recommendations for policy compliance"""
        return [f"Comply with policy {policy.rule_id}"]
    
    def _take_policy_action(self, peer_id: str, violations: List[str]) -> bool:
        """Take policy enforcement action"""
        try:
            # In a real implementation, this would take specific actions
            # based on the policy violations
            logger.warning(f"Policy violations for {peer_id}: {violations}")
            return True
            
        except Exception as e:
            logger.error(f"Error taking policy action: {e}")
            return False
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Every minute
                
                # Monitor peer performance
                online_peers = self.service_registry.get_online_peers()
                
                for peer in online_peers:
                    if peer.peer_id not in self.aggregate_feedback:
                        self.aggregate_feedback[peer.peer_id] = []
                    
                    # Add current performance
                    self.aggregate_feedback[peer.peer_id].append(peer.performance_rating)
                    
                    # Keep only recent history
                    if len(self.aggregate_feedback[peer.peer_id]) > 100:
                        self.aggregate_feedback[peer.peer_id] = self.aggregate_feedback[peer.peer_id][-100:]
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    async def _policy_enforcement_loop(self):
        """Background policy enforcement"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Check policy compliance for all peers
                online_peers = self.service_registry.get_online_peers()
                
                for peer in online_peers:
                    # Check compliance (simplified)
                    # In a real implementation, this would perform comprehensive checks
                    pass
                
            except Exception as e:
                logger.error(f"Error in policy enforcement: {e}")
    
    async def _bandit_update_loop(self):
        """Background bandit model updates"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Update bandit based on recent performance feedback
                # This would incorporate rewards from completed exchanges
                
            except Exception as e:
                logger.error(f"Error in bandit update: {e}")
    
    def _determine_exchange_type_abstract(self, student_profile: AbstractProfile, teacher_profile: AbstractProfile) -> str:
        """
        Determine exchange type based on abstract profile compatibility
        """
        # Per our paper, we are only focusing on knowledge distillation.
        # The PEFT exchange mechanism is disabled.
        return "knowledge_distillation"
    
    def _generate_kd_parameters_abstract(self, student_profile: AbstractProfile, teacher_profile: AbstractProfile) -> Tuple[float, float]:
        """
        Generate KD parameters based on abstract profile features
        """
        # Performance gap-based alpha
        perf_gap = abs(student_profile.performance_category.value - teacher_profile.performance_category.value)
        alpha = 0.3 + 0.2 * (perf_gap / 4.0)  # Normalized to [0.3, 0.5]
        
        # Model size-based temperature
        size_ratio = student_profile.model_size_category.value / teacher_profile.model_size_category.value
        temperature = 1.5 + 1.0 * min(1.0, size_ratio)
        
        return alpha, temperature
    
    def _generate_peft_parameters_abstract(self, student_profile: AbstractProfile, teacher_profile: AbstractProfile) -> float:
        """
        Generate PEFT parameters based on abstract profile features
        """
        # Trust-based lambda weight
        trust_factor = min(student_profile.trust_score_quantized, teacher_profile.trust_score_quantized) / 10.0
        collab_factor = min(student_profile.collaboration_score_quantized, teacher_profile.collaboration_score_quantized) / 10.0
        
        lambda_weight = 0.05 + 0.15 * (0.6 * trust_factor + 0.4 * collab_factor)
        
        return lambda_weight
    
    def get_cpm_statistics(self) -> Dict[str, Any]:
        """Get CPM statistics - CONTENT-BLIND VERSION"""
        registry_stats = self.service_registry.get_registry_statistics()
        privacy_metrics = self.profile_sanitizer.get_privacy_metrics()
        
        return {
            'cpm_id': self.cpm_id,
            'endpoint': self.endpoint,
            'uptime': time.time() - getattr(self, 'start_time', time.time()),
            'registry_stats': registry_stats,
            'matching_history': len(self.matching_history),
            'abstract_profiles': len(self.abstract_profiles),
            'aggregate_feedback': {
                peer_id: len(feedback) for peer_id, feedback in self.aggregate_feedback.items()
            },
            'privacy_metrics': privacy_metrics,
            'active_policies': len([p for p in self.policies.values() if p.enabled]),
            'bandit_state': {
                'context_dim': self.bandit.d,
                'num_arms_played': getattr(self.bandit, 'num_arms_played', 0)
            },
            'privacy_guarantees': {
                'content_blind': True,
                'k_anonymity': self.privacy_params.k_anonymity,
                'dp_epsilon': self.privacy_params.dp_epsilon,
                'knowledge_content_access': False,
                'training_logs_access': False,
                'model_parameters_access': False
            }
        }


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create CPM service
        cpm = CPMService("cpm_main", "localhost:8000")
        
        # Start server
        await cpm.start_server()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await cpm.stop_server()
    
    asyncio.run(main())
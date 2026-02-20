#!/usr/bin/env python3
"""
P2P Service Registry and Discovery for KNEXA-FL
Implements distributed service discovery, peer registration, and health monitoring
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import socket
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Information about a registered peer"""
    peer_id: str
    endpoint: str  # "ip:port"
    public_key: str
    
    # Capability information
    model_family: str
    model_name: str
    model_size_mb: int
    architecture_type: str
    supported_modules: List[str]
    
    # Resource information
    device_info: str
    available_memory: float
    compute_capacity: float
    
    # Dynamic state
    status: str = "offline"  # "online", "busy", "offline"
    trust_score: float = 0.8
    performance_rating: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)
    
    # Statistics
    successful_exchanges: int = 0
    failed_exchanges: int = 0
    total_bytes_transferred: int = 0
    average_response_time: float = 0.0
    
    # Policy compliance
    privacy_level: str = "standard"  # "minimal", "standard", "high"
    governance_policies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'peer_id': self.peer_id,
            'endpoint': self.endpoint,
            'public_key': self.public_key,
            'model_family': self.model_family,
            'model_name': self.model_name,
            'model_size_mb': self.model_size_mb,
            'architecture_type': self.architecture_type,
            'supported_modules': self.supported_modules,
            'device_info': self.device_info,
            'available_memory': self.available_memory,
            'compute_capacity': self.compute_capacity,
            'status': self.status,
            'trust_score': self.trust_score,
            'performance_rating': self.performance_rating,
            'last_heartbeat': self.last_heartbeat,
            'successful_exchanges': self.successful_exchanges,
            'failed_exchanges': self.failed_exchanges,
            'total_bytes_transferred': self.total_bytes_transferred,
            'average_response_time': self.average_response_time,
            'privacy_level': self.privacy_level,
            'governance_policies': self.governance_policies
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PeerInfo':
        """Create from dictionary"""
        return cls(
            peer_id=data['peer_id'],
            endpoint=data['endpoint'],
            public_key=data['public_key'],
            model_family=data['model_family'],
            model_name=data['model_name'],
            model_size_mb=data['model_size_mb'],
            architecture_type=data['architecture_type'],
            supported_modules=data['supported_modules'],
            device_info=data['device_info'],
            available_memory=data['available_memory'],
            compute_capacity=data['compute_capacity'],
            status=data.get('status', 'offline'),
            trust_score=data.get('trust_score', 0.8),
            performance_rating=data.get('performance_rating', 0.0),
            last_heartbeat=data.get('last_heartbeat', time.time()),
            successful_exchanges=data.get('successful_exchanges', 0),
            failed_exchanges=data.get('failed_exchanges', 0),
            total_bytes_transferred=data.get('total_bytes_transferred', 0),
            average_response_time=data.get('average_response_time', 0.0),
            privacy_level=data.get('privacy_level', 'standard'),
            governance_policies=data.get('governance_policies', [])
        )


class ServiceRegistry:
    """
    Distributed service registry for P2P peer discovery
    Maintains peer information, handles registration, and provides discovery services
    """
    
    def __init__(self, registry_id: str = "cpm_registry"):
        self.registry_id = registry_id
        self.peers: Dict[str, PeerInfo] = {}
        self.peer_lock = threading.RLock()
        
        # Configuration
        self.heartbeat_timeout = 30.0  # seconds
        self.cleanup_interval = 60.0  # seconds
        self.max_peers = 1000
        
        # Statistics
        self.total_registrations = 0
        self.active_sessions = 0
        self.total_discoveries = 0
        
        # Start background tasks
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Service Registry {registry_id} initialized")
    
    def register_peer(self, peer_info: PeerInfo) -> bool:
        """
        Register a new peer or update existing peer information
        
        Args:
            peer_info: Peer information
            
        Returns:
            True if successful, False otherwise
        """
        with self.peer_lock:
            try:
                # Check if registry is full
                if len(self.peers) >= self.max_peers and peer_info.peer_id not in self.peers:
                    logger.warning(f"Registry full, cannot register peer {peer_info.peer_id}")
                    return False
                
                # Validate peer information
                if not self._validate_peer_info(peer_info):
                    logger.error(f"Invalid peer information for {peer_info.peer_id}")
                    return False
                
                # Update status and heartbeat
                peer_info.status = "online"
                peer_info.last_heartbeat = time.time()
                
                # Store peer information
                is_new_peer = peer_info.peer_id not in self.peers
                self.peers[peer_info.peer_id] = peer_info
                
                if is_new_peer:
                    self.total_registrations += 1
                    logger.info(f"Registered new peer {peer_info.peer_id} at {peer_info.endpoint}")
                else:
                    logger.info(f"Updated peer {peer_info.peer_id} information")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to register peer {peer_info.peer_id}: {e}")
                return False
    
    def unregister_peer(self, peer_id: str) -> bool:
        """
        Unregister a peer from the registry
        
        Args:
            peer_id: Peer identifier
            
        Returns:
            True if successful, False otherwise
        """
        with self.peer_lock:
            if peer_id in self.peers:
                del self.peers[peer_id]
                logger.info(f"Unregistered peer {peer_id}")
                return True
            return False
    
    def update_heartbeat(self, peer_id: str, status: str = "online", 
                        resource_info: Optional[Dict] = None) -> bool:
        """
        Update peer heartbeat and status
        
        Args:
            peer_id: Peer identifier
            status: Peer status ("online", "busy", "idle")
            resource_info: Optional resource information
            
        Returns:
            True if successful, False otherwise
        """
        with self.peer_lock:
            if peer_id not in self.peers:
                return False
            
            peer = self.peers[peer_id]
            peer.status = status
            peer.last_heartbeat = time.time()
            
            # Update resource information if provided
            if resource_info:
                peer.available_memory = resource_info.get('available_memory', peer.available_memory)
                peer.compute_capacity = resource_info.get('compute_capacity', peer.compute_capacity)
            
            return True
    
    def discover_peers(self, requester_id: str, capabilities_needed: List[str] = None,
                      model_family_preference: str = None, min_trust_score: float = 0.0,
                      max_results: int = 10) -> List[PeerInfo]:
        """
        Discover peers based on requirements
        
        Args:
            requester_id: ID of the requesting peer
            capabilities_needed: List of required capabilities
            model_family_preference: Preferred model family
            min_trust_score: Minimum trust score
            max_results: Maximum number of results
            
        Returns:
            List of matching peers
        """
        with self.peer_lock:
            self.total_discoveries += 1
            
            # Get all online peers except the requester
            candidates = [
                peer for peer in self.peers.values()
                if peer.peer_id != requester_id 
                and peer.status == "online"
                and peer.trust_score >= min_trust_score
                and self._is_peer_alive(peer)
            ]
            
            # Filter by capabilities if specified
            if capabilities_needed:
                candidates = [
                    peer for peer in candidates
                    if any(cap in peer.supported_modules for cap in capabilities_needed)
                ]
            
            # Filter by model family if specified
            if model_family_preference:
                preferred_candidates = [
                    peer for peer in candidates
                    if peer.model_family == model_family_preference
                ]
                if preferred_candidates:
                    candidates = preferred_candidates
            
            # Sort by trust score and performance rating
            candidates.sort(
                key=lambda p: (p.trust_score * 0.6 + p.performance_rating * 0.4),
                reverse=True
            )
            
            # Return top results
            results = candidates[:max_results]
            
            logger.info(f"Discovered {len(results)} peers for requester {requester_id}")
            return results
    
    def get_peer_info(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer"""
        with self.peer_lock:
            return self.peers.get(peer_id)
    
    def get_all_peers(self) -> List[PeerInfo]:
        """Get information about all registered peers"""
        with self.peer_lock:
            return list(self.peers.values())
    
    def get_online_peers(self) -> List[PeerInfo]:
        """Get all online peers"""
        with self.peer_lock:
            return [
                peer for peer in self.peers.values()
                if peer.status == "online" and self._is_peer_alive(peer)
            ]
    
    def update_peer_statistics(self, peer_id: str, exchange_success: bool,
                              bytes_transferred: int, response_time: float) -> bool:
        """
        Update peer statistics after an exchange
        
        Args:
            peer_id: Peer identifier
            exchange_success: Whether the exchange was successful
            bytes_transferred: Number of bytes transferred
            response_time: Response time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        with self.peer_lock:
            if peer_id not in self.peers:
                return False
            
            peer = self.peers[peer_id]
            
            if exchange_success:
                peer.successful_exchanges += 1
            else:
                peer.failed_exchanges += 1
            
            peer.total_bytes_transferred += bytes_transferred
            
            # Update average response time (exponential moving average)
            if peer.average_response_time == 0:
                peer.average_response_time = response_time
            else:
                peer.average_response_time = 0.7 * peer.average_response_time + 0.3 * response_time
            
            # Update performance rating
            total_exchanges = peer.successful_exchanges + peer.failed_exchanges
            if total_exchanges > 0:
                success_rate = peer.successful_exchanges / total_exchanges
                # Combine success rate with response time performance
                response_performance = max(0, 1.0 - (peer.average_response_time / 10.0))
                peer.performance_rating = 0.7 * success_rate + 0.3 * response_performance
            
            return True
    
    def update_trust_score(self, peer_id: str, trust_delta: float, reason: str) -> bool:
        """
        Update peer trust score
        
        Args:
            peer_id: Peer identifier
            trust_delta: Change in trust score
            reason: Reason for trust change
            
        Returns:
            True if successful, False otherwise
        """
        with self.peer_lock:
            if peer_id not in self.peers:
                return False
            
            peer = self.peers[peer_id]
            old_trust = peer.trust_score
            peer.trust_score = max(0.0, min(1.0, peer.trust_score + trust_delta))
            
            logger.info(f"Updated trust score for {peer_id}: {old_trust:.3f} -> {peer.trust_score:.3f} ({reason})")
            return True
    
    def get_registry_statistics(self) -> Dict:
        """Get registry statistics"""
        with self.peer_lock:
            online_peers = len(self.get_online_peers())
            return {
                'registry_id': self.registry_id,
                'total_peers': len(self.peers),
                'online_peers': online_peers,
                'offline_peers': len(self.peers) - online_peers,
                'total_registrations': self.total_registrations,
                'total_discoveries': self.total_discoveries,
                'active_sessions': self.active_sessions,
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            }
    
    def _validate_peer_info(self, peer_info: PeerInfo) -> bool:
        """Validate peer information"""
        required_fields = ['peer_id', 'endpoint', 'public_key', 'model_family', 'model_name']
        
        for field in required_fields:
            if not getattr(peer_info, field):
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate endpoint format
        try:
            host, port = peer_info.endpoint.split(':')
            socket.inet_aton(host)  # Validate IP address
            int(port)  # Validate port number
        except (ValueError, socket.error):
            logger.error(f"Invalid endpoint format: {peer_info.endpoint}")
            return False
        
        return True
    
    def _is_peer_alive(self, peer: PeerInfo) -> bool:
        """Check if peer is alive based on heartbeat"""
        return time.time() - peer.last_heartbeat < self.heartbeat_timeout
    
    def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_dead_peers()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_dead_peers(self):
        """Remove dead peers from registry"""
        with self.peer_lock:
            current_time = time.time()
            dead_peers = [
                peer_id for peer_id, peer in self.peers.items()
                if current_time - peer.last_heartbeat > self.heartbeat_timeout
            ]
            
            for peer_id in dead_peers:
                logger.info(f"Removing dead peer {peer_id}")
                del self.peers[peer_id]
    
    def shutdown(self):
        """Shutdown the registry"""
        self.running = False
        logger.info(f"Service Registry {self.registry_id} shutting down")


class DistributedServiceRegistry:
    """
    Distributed service registry implementation
    Supports multiple registry nodes with eventual consistency
    """
    
    def __init__(self, node_id: str, registry_nodes: List[str] = None):
        self.node_id = node_id
        self.local_registry = ServiceRegistry(f"node_{node_id}")
        self.registry_nodes = registry_nodes or []
        
        # Synchronization
        self.sync_interval = 30.0  # seconds
        self.last_sync = time.time()
        
        # Start synchronization loop
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        logger.info(f"Distributed Service Registry node {node_id} initialized")
    
    def register_peer(self, peer_info: PeerInfo) -> bool:
        """Register peer in local registry and propagate to other nodes"""
        success = self.local_registry.register_peer(peer_info)
        
        if success:
            # Propagate to other nodes asynchronously
            threading.Thread(
                target=self._propagate_registration,
                args=(peer_info,),
                daemon=True
            ).start()
        
        return success
    
    def _propagate_registration(self, peer_info: PeerInfo):
        """Propagate peer registration to other registry nodes"""
        # Implementation would depend on the specific distributed protocol
        # For now, just log the action
        logger.info(f"Propagating registration of {peer_info.peer_id} to {len(self.registry_nodes)} nodes")
    
    def _sync_loop(self):
        """Synchronization loop with other registry nodes"""
        while self.local_registry.running:
            try:
                time.sleep(self.sync_interval)
                self._sync_with_nodes()
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
    
    def _sync_with_nodes(self):
        """Synchronize with other registry nodes"""
        # Implementation would depend on the specific distributed protocol
        # Could use gossip protocol, consensus algorithms, etc.
        current_time = time.time()
        if current_time - self.last_sync > self.sync_interval:
            logger.debug(f"Synchronizing with {len(self.registry_nodes)} registry nodes")
            self.last_sync = current_time


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create service registry
    registry = ServiceRegistry("test_registry")
    
    # Create sample peer
    peer_info = PeerInfo(
        peer_id="peer1",
        endpoint="192.168.1.100:9000",
        public_key="test_public_key",
        model_family="pythia",
        model_name="EleutherAI/pythia-160m",
        model_size_mb=160,
        architecture_type="decoder",
        supported_modules=["lora", "dora"],
        device_info="NVIDIA A100 80GB",
        available_memory=75000.0,
        compute_capacity=100.0
    )
    
    # Test registration
    success = registry.register_peer(peer_info)
    print(f"Registration success: {success}")
    
    # Test discovery
    discovered = registry.discover_peers("peer2", max_results=5)
    print(f"Discovered peers: {len(discovered)}")
    
    # Test statistics
    stats = registry.get_registry_statistics()
    print(f"Registry statistics: {stats}")
    
    # Shutdown
    registry.shutdown()
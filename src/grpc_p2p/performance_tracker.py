#!/usr/bin/env python3
"""
Performance Tracker for KNEXA-FL Two-Phase Training
Tracks local baseline vs P2P enhancement performance with clear attribution
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ClientPerformanceRecord:
    """Individual client performance record for a single round"""
    client_id: str
    round_id: int
    
    # Phase 1: Local Training Baseline
    pre_local_performance: float = 0.0
    post_local_performance: float = 0.0
    local_improvement: float = 0.0
    local_training_time: float = 0.0
    
    # Phase 2: P2P Collaboration Enhancement
    pre_p2p_performance: float = 0.0
    post_p2p_performance: float = 0.0
    p2p_improvement: float = 0.0
    p2p_exchanges_participated: int = 0
    p2p_roles: List[str] = field(default_factory=list)
    p2p_training_time: float = 0.0
    
    # Overall metrics
    total_improvement: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.local_improvement = self.post_local_performance - self.pre_local_performance
        self.p2p_improvement = self.post_p2p_performance - self.pre_p2p_performance
        self.total_improvement = self.post_p2p_performance - self.pre_local_performance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'client_id': self.client_id,
            'round_id': self.round_id,
            'local_baseline': {
                'pre_performance': self.pre_local_performance,
                'post_performance': self.post_local_performance,
                'improvement': self.local_improvement,
                'training_time': self.local_training_time
            },
            'p2p_enhancement': {
                'pre_performance': self.pre_p2p_performance,
                'post_performance': self.post_p2p_performance,
                'improvement': self.p2p_improvement,
                'exchanges_participated': self.p2p_exchanges_participated,
                'roles': self.p2p_roles,
                'training_time': self.p2p_training_time
            },
            'overall': {
                'total_improvement': self.total_improvement,
                'timestamp': self.timestamp
            }
        }


@dataclass
class SystemPerformanceSnapshot:
    """System-wide performance snapshot for a round"""
    round_id: int
    
    # Local training baseline statistics
    local_baseline_mean: float = 0.0
    local_baseline_std: float = 0.0
    local_baseline_min: float = 0.0
    local_baseline_max: float = 0.0
    
    # P2P enhancement statistics
    p2p_enhancement_mean: float = 0.0
    p2p_enhancement_std: float = 0.0
    p2p_enhancement_min: float = 0.0
    p2p_enhancement_max: float = 0.0
    
    # Overall performance statistics
    overall_performance_mean: float = 0.0
    overall_performance_std: float = 0.0
    overall_performance_min: float = 0.0
    overall_performance_max: float = 0.0
    
    # Participation statistics
    total_clients: int = 0
    clients_with_p2p: int = 0
    total_p2p_exchanges: int = 0
    p2p_participation_rate: float = 0.0
    
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'round_id': self.round_id,
            'local_baseline': {
                'mean': self.local_baseline_mean,
                'std': self.local_baseline_std,
                'min': self.local_baseline_min,
                'max': self.local_baseline_max
            },
            'p2p_enhancement': {
                'mean': self.p2p_enhancement_mean,
                'std': self.p2p_enhancement_std,
                'min': self.p2p_enhancement_min,
                'max': self.p2p_enhancement_max
            },
            'overall_performance': {
                'mean': self.overall_performance_mean,
                'std': self.overall_performance_std,
                'min': self.overall_performance_min,
                'max': self.overall_performance_max
            },
            'participation': {
                'total_clients': self.total_clients,
                'clients_with_p2p': self.clients_with_p2p,
                'total_p2p_exchanges': self.total_p2p_exchanges,
                'p2p_participation_rate': self.p2p_participation_rate
            },
            'timestamp': self.timestamp
        }


class PerformanceTracker:
    """
    Comprehensive performance tracker for KNEXA-FL two-phase training
    
    Tracks performance across:
    1. Local training baseline (all clients)
    2. P2P collaboration enhancement (participating clients)
    3. Overall system performance and comparisons
    """
    
    def __init__(self, num_clients: int):
        """
        Initialize performance tracker
        
        Args:
            num_clients: Total number of clients in the system
        """
        self.num_clients = num_clients
        self.client_records: Dict[str, Dict[int, ClientPerformanceRecord]] = {}
        self.round_snapshots: Dict[int, SystemPerformanceSnapshot] = {}
        self.experiment_metadata = {
            'num_clients': num_clients,
            'start_time': time.time(),
            'total_rounds': 0
        }
        
        # Initialize client records
        for client_id in range(num_clients):
            self.client_records[f"client_{client_id}"] = {}
        
        logger.info(f"Performance tracker initialized for {num_clients} clients")
    
    def record_local_baseline(self, 
                            client_id: str, 
                            round_id: int,
                            pre_performance: float,
                            post_performance: float,
                            training_time: float = 0.0):
        """
        Record local training baseline performance
        
        Args:
            client_id: Client identifier
            round_id: Federated learning round
            pre_performance: Performance before local training
            post_performance: Performance after local training
            training_time: Time taken for local training
        """
        # Get or create client record for this round
        if client_id not in self.client_records:
            self.client_records[client_id] = {}
        
        if round_id not in self.client_records[client_id]:
            self.client_records[client_id][round_id] = ClientPerformanceRecord(
                client_id=client_id,
                round_id=round_id
            )
        
        record = self.client_records[client_id][round_id]
        record.pre_local_performance = pre_performance
        record.post_local_performance = post_performance
        record.local_training_time = training_time
        record.pre_p2p_performance = post_performance  # P2P starts from local result
        
        # Update derived metrics
        record.__post_init__()
        
        logger.debug(f"Recorded local baseline for {client_id} round {round_id}: "
                    f"{pre_performance:.6f} -> {post_performance:.6f} "
                    f"(+{record.local_improvement:.6f})")
    
    def record_p2p_enhancement(self,
                             client_id: str,
                             round_id: int,
                             pre_p2p_performance: float,
                             post_p2p_performance: float,
                             exchanges_participated: int,
                             roles: List[str],
                             training_time: float = 0.0):
        """
        Record P2P collaboration enhancement performance
        
        Args:
            client_id: Client identifier
            round_id: Federated learning round
            pre_p2p_performance: Performance before P2P collaboration
            post_p2p_performance: Performance after P2P collaboration
            exchanges_participated: Number of P2P exchanges participated in
            roles: List of roles played (teacher, student, self_distillation)
            training_time: Time taken for P2P collaboration
        """
        if client_id not in self.client_records:
            self.client_records[client_id] = {}
        
        if round_id not in self.client_records[client_id]:
            logger.warning(f"No local baseline recorded for {client_id} round {round_id}")
            self.client_records[client_id][round_id] = ClientPerformanceRecord(
                client_id=client_id,
                round_id=round_id
            )
        
        record = self.client_records[client_id][round_id]
        record.pre_p2p_performance = pre_p2p_performance
        record.post_p2p_performance = post_p2p_performance
        record.p2p_exchanges_participated = exchanges_participated
        record.p2p_roles = roles.copy()
        record.p2p_training_time = training_time
        
        # Update derived metrics
        record.__post_init__()
        
        logger.debug(f"Recorded P2P enhancement for {client_id} round {round_id}: "
                    f"{pre_p2p_performance:.6f} -> {post_p2p_performance:.6f} "
                    f"(+{record.p2p_improvement:.6f})")
    
    def finalize_round(self, round_id: int):
        """
        Finalize a round and compute system-wide statistics
        
        Args:
            round_id: Federated learning round to finalize
        """
        logger.info(f"Finalizing performance statistics for round {round_id}")
        
        # Collect all client records for this round
        round_records = []
        for client_id in self.client_records:
            if round_id in self.client_records[client_id]:
                round_records.append(self.client_records[client_id][round_id])
        
        if not round_records:
            logger.warning(f"No client records found for round {round_id}")
            return
        
        # Compute system-wide statistics
        snapshot = SystemPerformanceSnapshot(round_id=round_id)
        
        # Local baseline statistics
        local_improvements = [r.local_improvement for r in round_records]
        local_post_performances = [r.post_local_performance for r in round_records]
        
        if local_improvements:
            snapshot.local_baseline_mean = np.mean(local_improvements)
            snapshot.local_baseline_std = np.std(local_improvements)
            snapshot.local_baseline_min = np.min(local_improvements)
            snapshot.local_baseline_max = np.max(local_improvements)
        
        # P2P enhancement statistics
        p2p_improvements = [r.p2p_improvement for r in round_records]
        p2p_post_performances = [r.post_p2p_performance for r in round_records]
        
        if p2p_improvements:
            snapshot.p2p_enhancement_mean = np.mean(p2p_improvements)
            snapshot.p2p_enhancement_std = np.std(p2p_improvements)
            snapshot.p2p_enhancement_min = np.min(p2p_improvements)
            snapshot.p2p_enhancement_max = np.max(p2p_improvements)
        
        # Overall performance statistics
        total_improvements = [r.total_improvement for r in round_records]
        
        if total_improvements:
            snapshot.overall_performance_mean = np.mean(total_improvements)
            snapshot.overall_performance_std = np.std(total_improvements)
            snapshot.overall_performance_min = np.min(total_improvements)
            snapshot.overall_performance_max = np.max(total_improvements)
        
        # Participation statistics
        snapshot.total_clients = len(round_records)
        snapshot.clients_with_p2p = len([r for r in round_records if r.p2p_exchanges_participated > 0])
        snapshot.total_p2p_exchanges = sum(r.p2p_exchanges_participated for r in round_records)
        snapshot.p2p_participation_rate = snapshot.clients_with_p2p / max(1, snapshot.total_clients)
        
        # Store snapshot
        self.round_snapshots[round_id] = snapshot
        
        # Update experiment metadata
        self.experiment_metadata['total_rounds'] = max(self.experiment_metadata['total_rounds'], round_id + 1)
        
        logger.info(f"Round {round_id} statistics:")
        logger.info(f"  Local baseline: {snapshot.local_baseline_mean:.6f} ± {snapshot.local_baseline_std:.6f}")
        logger.info(f"  P2P enhancement: {snapshot.p2p_enhancement_mean:.6f} ± {snapshot.p2p_enhancement_std:.6f}")
        logger.info(f"  Overall improvement: {snapshot.overall_performance_mean:.6f} ± {snapshot.overall_performance_std:.6f}")
        logger.info(f"  P2P participation: {snapshot.clients_with_p2p}/{snapshot.total_clients} ({snapshot.p2p_participation_rate:.1%})")
    
    def get_client_performance_history(self, client_id: str) -> List[Dict[str, Any]]:
        """Get performance history for a specific client"""
        if client_id not in self.client_records:
            return []
        
        return [record.to_dict() for record in self.client_records[client_id].values()]
    
    def get_round_performance_summary(self, round_id: int) -> Optional[Dict[str, Any]]:
        """Get performance summary for a specific round"""
        if round_id not in self.round_snapshots:
            return None
        
        return self.round_snapshots[round_id].to_dict()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        summary = {
            'metadata': self.experiment_metadata.copy(),
            'round_summaries': {},
            'client_summaries': {},
            'overall_statistics': {}
        }
        
        # Add round summaries
        for round_id, snapshot in self.round_snapshots.items():
            summary['round_summaries'][round_id] = snapshot.to_dict()
        
        # Add client summaries
        for client_id, records in self.client_records.items():
            client_summary = {
                'total_rounds': len(records),
                'local_improvements': [r.local_improvement for r in records.values()],
                'p2p_improvements': [r.p2p_improvement for r in records.values()],
                'total_improvements': [r.total_improvement for r in records.values()],
                'p2p_participation': sum(1 for r in records.values() if r.p2p_exchanges_participated > 0)
            }
            
            # Calculate client-level statistics
            if client_summary['local_improvements']:
                client_summary['avg_local_improvement'] = np.mean(client_summary['local_improvements'])
                client_summary['avg_p2p_improvement'] = np.mean(client_summary['p2p_improvements'])
                client_summary['avg_total_improvement'] = np.mean(client_summary['total_improvements'])
            
            summary['client_summaries'][client_id] = client_summary
        
        # Calculate overall statistics
        if self.round_snapshots:
            all_local_means = [s.local_baseline_mean for s in self.round_snapshots.values()]
            all_p2p_means = [s.p2p_enhancement_mean for s in self.round_snapshots.values()]
            all_overall_means = [s.overall_performance_mean for s in self.round_snapshots.values()]
            
            summary['overall_statistics'] = {
                'avg_local_baseline_improvement': np.mean(all_local_means),
                'avg_p2p_enhancement': np.mean(all_p2p_means),
                'avg_overall_improvement': np.mean(all_overall_means),
                'total_rounds': len(self.round_snapshots),
                'avg_p2p_participation_rate': np.mean([s.p2p_participation_rate for s in self.round_snapshots.values()])
            }
        
        return summary
    
    def generate_performance_report(self) -> str:
        """Generate human-readable performance report"""
        summary = self.get_experiment_summary()
        
        report = []
        report.append("="*60)
        report.append("KNEXA-FL PERFORMANCE REPORT")
        report.append("="*60)
        
        # Experiment metadata
        metadata = summary['metadata']
        report.append(f"Experiment Duration: {metadata['total_rounds']} rounds")
        report.append(f"Total Clients: {metadata['num_clients']}")
        report.append("")
        
        # Overall statistics
        if 'overall_statistics' in summary and summary['overall_statistics']:
            stats = summary['overall_statistics']
            report.append("OVERALL PERFORMANCE STATISTICS")
            report.append("-" * 40)
            report.append(f"Average Local Baseline Improvement: {stats['avg_local_baseline_improvement']:.6f}")
            report.append(f"Average P2P Enhancement:           {stats['avg_p2p_enhancement']:.6f}")
            report.append(f"Average Overall Improvement:       {stats['avg_overall_improvement']:.6f}")
            report.append(f"Average P2P Participation Rate:    {stats['avg_p2p_participation_rate']:.1%}")
            report.append("")
        
        # Round-by-round breakdown
        report.append("ROUND-BY-ROUND BREAKDOWN")
        report.append("-" * 40)
        report.append(f"{'Round':<6} {'Local':<10} {'P2P':<10} {'Overall':<10} {'P2P Part.':<10}")
        report.append("-" * 46)
        
        for round_id in sorted(summary['round_summaries'].keys()):
            round_data = summary['round_summaries'][round_id]
            local_mean = round_data['local_baseline']['mean']
            p2p_mean = round_data['p2p_enhancement']['mean']
            overall_mean = round_data['overall_performance']['mean']
            p2p_rate = round_data['participation']['p2p_participation_rate']
            
            report.append(f"{round_id:<6} {local_mean:<10.6f} {p2p_mean:<10.6f} {overall_mean:<10.6f} {p2p_rate:<10.1%}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)


# Global performance tracker instance
_global_tracker = None


def get_global_performance_tracker(num_clients: int) -> PerformanceTracker:
    """Get or create global performance tracker singleton"""
    global _global_tracker
    
    if _global_tracker is None:
        _global_tracker = PerformanceTracker(num_clients)
    
    return _global_tracker


def reset_global_performance_tracker():
    """Reset global performance tracker (for testing)"""
    global _global_tracker
    _global_tracker = None


if __name__ == "__main__":
    # Test the performance tracker
    logging.basicConfig(level=logging.INFO)
    
    # Create tracker
    tracker = PerformanceTracker(4)
    
    # Simulate performance data
    for round_id in range(3):
        for client_id in range(4):
            client_str = f"client_{client_id}"
            
            # Record local baseline
            pre_local = 0.1 + np.random.normal(0, 0.01)
            post_local = pre_local + 0.02 + np.random.normal(0, 0.005)
            tracker.record_local_baseline(client_str, round_id, pre_local, post_local)
            
            # Record P2P enhancement
            pre_p2p = post_local
            post_p2p = pre_p2p + 0.01 + np.random.normal(0, 0.003)
            tracker.record_p2p_enhancement(client_str, round_id, pre_p2p, post_p2p, 2, ['teacher', 'student'])
        
        # Finalize round
        tracker.finalize_round(round_id)
    
    # Generate report
    print("\n" + tracker.generate_performance_report())
#!/usr/bin/env python3
"""
Comprehensive Federated Learning Metrics Tracker for KNEXA-FL
Research-grade metrics collection system for rigorous FL evaluation

This module provides comprehensive tracking of all metrics essential for 
federated learning research, ensuring fair comparison with baselines and 
meeting reviewer expectations for rigorous experimental evaluation.

Key Features:
- Standard FL metrics (communication, convergence, fairness)
- Realistic network simulation for local experiments
- Statistical heterogeneity analysis
- KNEXA-FL specific metrics (P2P efficiency, knowledge quality)
- Publication-ready reporting and visualization
- Integration with existing performance tracking
"""

import time
import numpy as np
import torch
import pickle
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from collections import defaultdict, deque
# import matplotlib.pyplot as plt  # Removed to avoid plot generation
# import seaborn as sns  # Removed to avoid plot generation
from scipy import stats
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class CommunicationMetrics:
    """Communication overhead and efficiency metrics"""
    # Per-round communication costs
    bytes_sent_per_round: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    bytes_received_per_round: Dict[int, Dict[int, int]] = field(default_factory=lambda: defaultdict(dict))
    
    # Knowledge transfer payload sizes
    knowledge_payload_sizes: List[int] = field(default_factory=list)
    logit_payload_sizes: List[int] = field(default_factory=list)
    text_payload_sizes: List[int] = field(default_factory=list)
    
    # Network simulation
    simulated_latency_ms: List[float] = field(default_factory=list)
    simulated_bandwidth_mbps: float = 100.0  # Realistic enterprise bandwidth
    
    # Communication rounds and efficiency
    total_communication_rounds: int = 0
    total_bytes_transferred: int = 0
    communication_efficiency: float = 0.0  # performance_gain / total_bytes
    
    # P2P specific metrics
    p2p_pairing_overhead_ms: List[float] = field(default_factory=list)
    cpm_decision_time_ms: List[float] = field(default_factory=list)

@dataclass
class SystemMetrics:
    """System resource usage and overhead metrics"""
    # Memory usage tracking
    gpu_memory_usage_mb: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    peak_memory_usage_mb: Dict[int, float] = field(default_factory=dict)
    memory_efficiency: float = 0.0
    
    # Compute overhead
    model_loading_time_s: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    knowledge_distillation_time_s: List[float] = field(default_factory=list)
    privacy_computation_time_s: List[float] = field(default_factory=list)
    
    # GPU utilization
    gpu_utilization_timeline: List[float] = field(default_factory=list)
    compute_efficiency: float = 0.0
    
    # Energy consumption estimation (for green AI metrics)
    estimated_energy_kwh: float = 0.0

@dataclass
class ConvergenceMetrics:
    """Convergence and learning efficiency metrics"""
    # Performance tracking
    global_performance_history: List[float] = field(default_factory=list)
    client_performance_history: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Convergence analysis
    convergence_round: Optional[int] = None
    convergence_threshold: float = 0.01
    performance_plateau_rounds: int = 0
    
    # Learning efficiency
    sample_efficiency: float = 0.0  # performance / total_samples
    time_to_convergence_s: float = 0.0
    
    # Stability metrics
    performance_variance: float = 0.0
    convergence_stability: float = 0.0

@dataclass
class FairnessMetrics:
    """Client fairness and equity metrics"""
    # Performance fairness
    client_performance_variance: float = 0.0
    performance_gini_coefficient: float = 0.0
    worst_client_performance: float = 0.0
    best_client_performance: float = 0.0
    
    # Participation fairness
    participation_rates: Dict[int, float] = field(default_factory=dict)
    resource_allocation_fairness: float = 0.0
    
    # Knowledge sharing fairness
    knowledge_received_per_client: Dict[int, int] = field(default_factory=dict)
    knowledge_contributed_per_client: Dict[int, int] = field(default_factory=dict)
    contribution_fairness_score: float = 0.0

@dataclass
class HeterogeneityMetrics:
    """Statistical and system heterogeneity analysis"""
    # Data heterogeneity
    data_distribution_divergence: Dict[Tuple[int, int], float] = field(default_factory=dict)
    statistical_heterogeneity_score: float = 0.0
    
    # Model heterogeneity
    model_architecture_diversity: float = 0.0
    model_size_variance: float = 0.0
    parameter_count_diversity: List[int] = field(default_factory=list)
    
    # System heterogeneity
    compute_capability_variance: float = 0.0
    memory_heterogeneity_score: float = 0.0

@dataclass
class KnexaFLMetrics:
    """KNEXA-FL specific metrics"""
    # Knowledge transfer quality
    text_transfer_success_rate: float = 0.0
    logit_transfer_success_rate: float = 0.0
    hybrid_transfer_efficiency: float = 0.0
    
    # Knowledge distillation effectiveness
    kd_performance_gain: List[float] = field(default_factory=list)
    temperature_optimization_history: List[float] = field(default_factory=list)
    alpha_optimization_history: List[float] = field(default_factory=list)
    
    # P2P efficiency
    pairing_quality_scores: List[float] = field(default_factory=list)
    bandit_exploration_rate: float = 0.0
    bandit_regret: List[float] = field(default_factory=list)
    
    # Privacy preservation
    differential_privacy_epsilon: float = 0.0
    privacy_budget_consumption: List[float] = field(default_factory=list)
    sier_safety_scores: List[float] = field(default_factory=list)

class NetworkSimulator:
    """Realistic network simulation for local experiments"""
    
    def __init__(self, 
                 base_latency_ms: float = 50.0,
                 bandwidth_mbps: float = 100.0,
                 jitter_factor: float = 0.1):
        self.base_latency_ms = base_latency_ms
        self.bandwidth_mbps = bandwidth_mbps
        self.jitter_factor = jitter_factor
        
    def simulate_transfer_time(self, payload_size_bytes: int) -> float:
        """Simulate realistic network transfer time"""
        # Transmission time based on bandwidth
        transmission_time_ms = (payload_size_bytes * 8) / (self.bandwidth_mbps * 1000)
        
        # Add latency with jitter
        jitter = np.random.normal(0, self.base_latency_ms * self.jitter_factor)
        total_latency_ms = self.base_latency_ms + jitter
        
        return max(0, transmission_time_ms + total_latency_ms)
    
    def get_current_bandwidth(self) -> float:
        """Simulate variable bandwidth conditions"""
        # Add realistic bandwidth variation (±20%)
        variation = np.random.normal(1.0, 0.2)
        return max(10.0, self.bandwidth_mbps * variation)

class FederatedMetricsTracker:
    """
    Comprehensive federated learning metrics tracker for KNEXA-FL
    
    This class provides research-grade metrics collection and analysis
    for rigorous evaluation of federated learning experiments.
    """
    
    def __init__(self, 
                 num_clients: int,
                 save_dir: str = "federated_metrics",
                 enable_network_simulation: bool = True):
        self.num_clients = num_clients
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize metric containers
        self.communication = CommunicationMetrics()
        self.system = SystemMetrics()
        self.convergence = ConvergenceMetrics()
        self.fairness = FairnessMetrics()
        self.heterogeneity = HeterogeneityMetrics()
        self.knexa_fl = KnexaFLMetrics()
        
        # Network simulation
        self.network_sim = NetworkSimulator() if enable_network_simulation else None
        
        # Experiment tracking
        self.experiment_start_time = time.time()
        self.current_round = 0
        self.round_start_times: Dict[int, float] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Baseline comparison data
        self.baseline_results: Dict[str, Any] = {}
        
        logger.info(f"Initialized FederatedMetricsTracker for {num_clients} clients")
        logger.info(f"Metrics will be saved to: {self.save_dir}")
    
    def start_round(self, round_id: int):
        """Mark the start of a federated learning round"""
        with self._lock:
            self.current_round = round_id
            self.round_start_times[round_id] = time.time()
            self.communication.total_communication_rounds += 1
            logger.debug(f"Started tracking round {round_id}")
    
    def record_communication(self,
                           sender_id: int,
                           receiver_id: int,
                           payload_size_bytes: int,
                           transfer_type: str = "knowledge"):
        """Record communication between clients"""
        with self._lock:
            round_id = self.current_round
            
            # Record bytes transferred
            if sender_id not in self.communication.bytes_sent_per_round[round_id]:
                self.communication.bytes_sent_per_round[round_id][sender_id] = 0
            if receiver_id not in self.communication.bytes_received_per_round[round_id]:
                self.communication.bytes_received_per_round[round_id][receiver_id] = 0
            
            self.communication.bytes_sent_per_round[round_id][sender_id] += payload_size_bytes
            self.communication.bytes_received_per_round[round_id][receiver_id] += payload_size_bytes
            self.communication.total_bytes_transferred += payload_size_bytes
            
            # Record payload sizes by type
            if transfer_type == "logits":
                self.communication.logit_payload_sizes.append(payload_size_bytes)
            elif transfer_type == "text":
                self.communication.text_payload_sizes.append(payload_size_bytes)
            else:
                self.communication.knowledge_payload_sizes.append(payload_size_bytes)
            
            # Simulate network transfer if enabled
            if self.network_sim:
                transfer_time_ms = self.network_sim.simulate_transfer_time(payload_size_bytes)
                self.communication.simulated_latency_ms.append(transfer_time_ms)
            
            logger.debug(f"Recorded communication: {sender_id} → {receiver_id}, "
                        f"{payload_size_bytes} bytes ({transfer_type})")
    
    def record_system_metrics(self,
                            client_id: int,
                            gpu_memory_mb: float,
                            operation_time_s: float,
                            operation_type: str):
        """Record system resource usage"""
        with self._lock:
            # GPU memory tracking
            self.system.gpu_memory_usage_mb[client_id].append(gpu_memory_mb)
            if client_id not in self.system.peak_memory_usage_mb:
                self.system.peak_memory_usage_mb[client_id] = gpu_memory_mb
            else:
                self.system.peak_memory_usage_mb[client_id] = max(
                    self.system.peak_memory_usage_mb[client_id], gpu_memory_mb
                )
            
            # Operation timing
            if operation_type == "model_loading":
                self.system.model_loading_time_s[client_id].append(operation_time_s)
            elif operation_type == "knowledge_distillation":
                self.system.knowledge_distillation_time_s.append(operation_time_s)
            elif operation_type == "privacy_computation":
                self.system.privacy_computation_time_s.append(operation_time_s)
            
            logger.debug(f"Recorded system metrics: client {client_id}, "
                        f"{operation_type} took {operation_time_s:.3f}s, "
                        f"GPU memory: {gpu_memory_mb:.1f}MB")
    
    def record_performance(self,
                         client_id: Optional[int],
                         performance: float,
                         is_global: bool = False):
        """Record client or global performance"""
        with self._lock:
            if is_global:
                self.convergence.global_performance_history.append(performance)
                self._check_convergence(performance)
            else:
                if client_id is not None:
                    self.convergence.client_performance_history[client_id].append(performance)
            
            logger.debug(f"Recorded performance: "
                        f"{'global' if is_global else f'client {client_id}'} = {performance:.6f}")
    
    def record_knowledge_transfer(self,
                                student_id: int,
                                teacher_id: int,
                                transfer_method: str,
                                success: bool,
                                quality_score: float,
                                kd_params: Dict[str, float]):
        """Record knowledge distillation transfer details"""
        with self._lock:
            # Update transfer success rates
            if transfer_method == "text":
                successes = sum(1 for s in [success] if s)
                total = len([success])
                self.knexa_fl.text_transfer_success_rate = successes / max(1, total)
            elif transfer_method == "logits":
                successes = sum(1 for s in [success] if s)
                total = len([success])
                self.knexa_fl.logit_transfer_success_rate = successes / max(1, total)
            
            # Record quality and parameters
            if success:
                self.knexa_fl.pairing_quality_scores.append(quality_score)
                
                if "temperature" in kd_params:
                    self.knexa_fl.temperature_optimization_history.append(kd_params["temperature"])
                if "alpha" in kd_params:
                    self.knexa_fl.alpha_optimization_history.append(kd_params["alpha"])
            
            logger.debug(f"Recorded knowledge transfer: {teacher_id} → {student_id}, "
                        f"method={transfer_method}, success={success}, quality={quality_score:.3f}")
    
    def record_privacy_metrics(self,
                             epsilon: float,
                             sier_score: float,
                             privacy_computation_time_s: float):
        """Record privacy preservation metrics"""
        with self._lock:
            self.knexa_fl.differential_privacy_epsilon = epsilon
            self.knexa_fl.sier_safety_scores.append(sier_score)
            self.system.privacy_computation_time_s.append(privacy_computation_time_s)
            
            logger.debug(f"Recorded privacy metrics: ε={epsilon}, SIER={sier_score:.4f}")
    
    def record_p2p_metrics(self,
                         pairing_time_ms: float,
                         cpm_decision_time_ms: float,
                         bandit_reward: float):
        """Record P2P specific metrics"""
        with self._lock:
            self.communication.p2p_pairing_overhead_ms.append(pairing_time_ms)
            self.communication.cpm_decision_time_ms.append(cpm_decision_time_ms)
            self.knexa_fl.bandit_regret.append(max(0, 1.0 - bandit_reward))  # Simple regret
            
            logger.debug(f"Recorded P2P metrics: pairing={pairing_time_ms:.1f}ms, "
                        f"cpm={cpm_decision_time_ms:.1f}ms, reward={bandit_reward:.3f}")
    
    def _check_convergence(self, current_performance: float):
        """Check if the system has converged"""
        if len(self.convergence.global_performance_history) < 5:
            return
        
        recent_performance = self.convergence.global_performance_history[-5:]
        performance_change = max(recent_performance) - min(recent_performance)
        
        if performance_change < self.convergence.convergence_threshold:
            if self.convergence.convergence_round is None:
                self.convergence.convergence_round = self.current_round
                self.convergence.time_to_convergence_s = time.time() - self.experiment_start_time
                logger.info(f"Convergence detected at round {self.current_round}")
    
    def compute_derived_metrics(self):
        """Compute derived metrics from collected data"""
        with self._lock:
            self._compute_communication_efficiency()
            self._compute_fairness_metrics()
            self._compute_heterogeneity_metrics()
            self._compute_knexa_fl_metrics()
            
            logger.info("Computed all derived metrics")
    
    def _compute_communication_efficiency(self):
        """Compute communication efficiency metrics"""
        if self.communication.total_bytes_transferred > 0 and self.convergence.global_performance_history:
            final_performance = self.convergence.global_performance_history[-1]
            initial_performance = self.convergence.global_performance_history[0]
            performance_gain = final_performance - initial_performance
            
            # Communication efficiency: performance gain per MB transferred
            bytes_mb = self.communication.total_bytes_transferred / (1024 * 1024)
            self.communication.communication_efficiency = performance_gain / max(0.001, bytes_mb)
    
    def _compute_fairness_metrics(self):
        """Compute fairness and equity metrics"""
        if not self.convergence.client_performance_history:
            return
        
        # Get final performance for each client
        final_performances = []
        for client_id in range(self.num_clients):
            if client_id in self.convergence.client_performance_history:
                client_history = self.convergence.client_performance_history[client_id]
                if client_history:
                    final_performances.append(client_history[-1])
        
        if len(final_performances) >= 2:
            # Performance variance
            self.fairness.client_performance_variance = np.var(final_performances)
            
            # Gini coefficient for performance inequality
            self.fairness.performance_gini_coefficient = self._compute_gini_coefficient(final_performances)
            
            # Best and worst client performance
            self.fairness.worst_client_performance = min(final_performances)
            self.fairness.best_client_performance = max(final_performances)
    
    def _compute_heterogeneity_metrics(self):
        """Compute statistical and system heterogeneity"""
        # Model size variance from globals
        from src.globals import MODEL_MAP, LLM_REGISTRY
        
        model_sizes = []
        for client_id in range(self.num_clients):
            if client_id in MODEL_MAP:
                model_name = MODEL_MAP[client_id]
                if model_name in LLM_REGISTRY:
                    size_str = LLM_REGISTRY[model_name]["params"]
                    # Extract numeric value (e.g., "160M" -> 160)
                    size_value = float(size_str.replace("M", "").replace("B", ""))
                    if "B" in size_str:
                        size_value *= 1000  # Convert B to M
                    model_sizes.append(size_value)
        
        if len(model_sizes) >= 2:
            self.heterogeneity.model_size_variance = np.var(model_sizes)
            self.heterogeneity.parameter_count_diversity = model_sizes
            
            # Architecture diversity (count unique architectures)
            architectures = set()
            for client_id in range(self.num_clients):
                if client_id in MODEL_MAP:
                    model_name = MODEL_MAP[client_id]
                    if model_name in LLM_REGISTRY:
                        arch = LLM_REGISTRY[model_name]["arch"]
                        architectures.add(arch)
            
            self.heterogeneity.model_architecture_diversity = len(architectures) / len(model_sizes)
    
    def _compute_knexa_fl_metrics(self):
        """Compute KNEXA-FL specific metrics"""
        # Hybrid transfer efficiency
        if self.knexa_fl.text_transfer_success_rate > 0 and self.knexa_fl.logit_transfer_success_rate > 0:
            self.knexa_fl.hybrid_transfer_efficiency = (
                self.knexa_fl.text_transfer_success_rate + self.knexa_fl.logit_transfer_success_rate
            ) / 2.0
        
        # Bandit exploration rate
        if self.knexa_fl.bandit_regret:
            recent_regret = self.knexa_fl.bandit_regret[-10:] if len(self.knexa_fl.bandit_regret) >= 10 else self.knexa_fl.bandit_regret
            self.knexa_fl.bandit_exploration_rate = np.mean(recent_regret)
    
    def _compute_gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        # Handle edge case where all values are zero
        sum_values = np.sum(sorted_values)
        if sum_values == 0:
            return 0.0  # Perfect equality when all values are zero
        
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * sum_values) - (n + 1) / n
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        self.compute_derived_metrics()
        
        report = {
            "experiment_info": {
                "num_clients": self.num_clients,
                "total_rounds": self.current_round,
                "experiment_duration_s": time.time() - self.experiment_start_time,
                "convergence_round": self.convergence.convergence_round,
                "time_to_convergence_s": self.convergence.time_to_convergence_s
            },
            "communication": asdict(self.communication),
            "system": asdict(self.system),
            "convergence": asdict(self.convergence),
            "fairness": asdict(self.fairness),
            "heterogeneity": asdict(self.heterogeneity),
            "knexa_fl": asdict(self.knexa_fl)
        }
        
        # Save report
        report_path = self.save_dir / f"federated_metrics_report_round_{self.current_round}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated comprehensive metrics report: {report_path}")
        return report
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create publication-ready summary table"""
        summary_data = {
            "Metric": [
                "Total Communication (MB)",
                "Communication Efficiency",
                "Convergence Round", 
                "Time to Convergence (s)",
                "Performance Gini Coefficient",
                "Model Size Variance",
                "Text Transfer Success Rate",
                "Hybrid Transfer Efficiency",
                "Average Privacy ε",
                "Peak GPU Memory (GB)"
            ],
            "Value": [
                f"{self.communication.total_bytes_transferred / (1024*1024):.2f}",
                f"{self.communication.communication_efficiency:.4f}",
                f"{self.convergence.convergence_round or 'N/A'}",
                f"{self.convergence.time_to_convergence_s:.1f}",
                f"{self.fairness.performance_gini_coefficient:.4f}",
                f"{self.heterogeneity.model_size_variance:.2f}",
                f"{self.knexa_fl.text_transfer_success_rate:.3f}",
                f"{self.knexa_fl.hybrid_transfer_efficiency:.3f}",
                f"{self.knexa_fl.differential_privacy_epsilon:.2f}",
                f"{max(self.system.peak_memory_usage_mb.values()) / 1024:.1f}" if self.system.peak_memory_usage_mb else "0.0"
            ]
        }
        
        df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = self.save_dir / f"summary_table_round_{self.current_round}.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Created summary table: {csv_path}")
        return df
    
    def plot_convergence_analysis(self):
        """Placeholder - plot generation disabled"""
        logger.info("Plot generation disabled - convergence data saved in JSON format")
        return
        '''plt.figure(figsize=(12, 8))
        
        # Global performance convergence
        plt.subplot(2, 2, 1)
        plt.plot(self.convergence.global_performance_history, 'b-', linewidth=2, label='Global Performance')
        if self.convergence.convergence_round:
            plt.axvline(x=self.convergence.convergence_round, color='r', linestyle='--', 
                       label=f'Convergence (Round {self.convergence.convergence_round})')
        plt.xlabel('Round')
        plt.ylabel('Performance')
        plt.title('Global Performance Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Client performance comparison
        plt.subplot(2, 2, 2)
        for client_id, history in self.convergence.client_performance_history.items():
            plt.plot(history, label=f'Client {client_id}', alpha=0.7)
        plt.xlabel('Round')
        plt.ylabel('Performance')
        plt.title('Client Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Communication overhead
        plt.subplot(2, 2, 3)
        rounds = list(range(len(self.convergence.global_performance_history)))
        comm_overhead = [self.communication.total_bytes_transferred / ((r+1) * 1024*1024) for r in rounds]
        plt.plot(rounds, comm_overhead, 'g-', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Cumulative Communication (MB)')
        plt.title('Communication Overhead')
        plt.grid(True, alpha=0.3)
        
        # Knowledge transfer quality
        plt.subplot(2, 2, 4)
        if self.knexa_fl.pairing_quality_scores:
            plt.plot(self.knexa_fl.pairing_quality_scores, 'o-', markersize=4)
            plt.xlabel('Transfer Instance')
            plt.ylabel('Quality Score')
            plt.title('Knowledge Transfer Quality')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / f"convergence_analysis_round_{self.current_round}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated convergence analysis plot: {plot_path}")'''
    
    def compare_with_baselines(self, baseline_results: Dict[str, Dict[str, float]]):
        """Compare KNEXA-FL results with baseline methods"""
        self.baseline_results = baseline_results
        
        # Create comparison table
        comparison_data = {
            "Method": ["KNEXA-FL"],
            "Convergence Round": [self.convergence.convergence_round or float('inf')],
            "Communication (MB)": [self.communication.total_bytes_transferred / (1024*1024)],
            "Final Performance": [self.convergence.global_performance_history[-1] if self.convergence.global_performance_history else 0],
            "Fairness (Gini)": [self.fairness.performance_gini_coefficient]
        }
        
        # Add baseline results
        for method, results in baseline_results.items():
            comparison_data["Method"].append(method)
            comparison_data["Convergence Round"].append(results.get("convergence_round", float('inf')))
            comparison_data["Communication (MB)"].append(results.get("communication_mb", 0))
            comparison_data["Final Performance"].append(results.get("final_performance", 0))
            comparison_data["Fairness (Gini)"].append(results.get("fairness_gini", 0))
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison
        comparison_path = self.save_dir / f"baseline_comparison_round_{self.current_round}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"Generated baseline comparison: {comparison_path}")
        return comparison_df

# Global instance for easy access
_global_metrics_tracker: Optional[FederatedMetricsTracker] = None

def initialize_global_tracker(num_clients: int, save_dir: str = "federated_metrics") -> FederatedMetricsTracker:
    """Initialize global metrics tracker"""
    global _global_metrics_tracker
    _global_metrics_tracker = FederatedMetricsTracker(num_clients, save_dir)
    return _global_metrics_tracker

def get_global_tracker() -> Optional[FederatedMetricsTracker]:
    """Get global metrics tracker instance"""
    return _global_metrics_tracker

def record_communication(sender_id: int, receiver_id: int, payload_size: int, transfer_type: str = "knowledge"):
    """Convenience function to record communication via global tracker"""
    if _global_metrics_tracker:
        _global_metrics_tracker.record_communication(sender_id, receiver_id, payload_size, transfer_type)

def record_performance(client_id: Optional[int], performance: float, is_global: bool = False):
    """Convenience function to record performance via global tracker"""
    if _global_metrics_tracker:
        _global_metrics_tracker.record_performance(client_id, performance, is_global)
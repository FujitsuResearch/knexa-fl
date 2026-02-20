"""
Main Experiment Runner for KNEXA-FL LinUCB Validation

Runs comprehensive experiments comparing different pairing strategies
across various settings.

Author: Inderjeet Singh
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import argparse
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_environment import SyntheticEnvironment
from reward_models import GroundTruthReward, PerformanceUpdateModel
from bandit_engines.random_baseline import RandomBaselineEngine
from bandit_engines.heterogeneity_greedy import HeterogeneityGreedyEngine
from bandit_engines.linucb_basic import LinUCBBasicEngine
from bandit_engines.linucb_enhanced import LinUCBEnhancedEngine
from bandit_engines.oracle_engine import OracleEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    exp_id: str
    num_clients: int
    num_rounds: int
    heterogeneity_level: str
    seed: int
    methods: List[str]
    
    # Reward model parameters
    gamma: float = 10.0      # Increased for meaningful improvements
    delta: float = 0.00001   # Reduced to prevent dominating the reward
    noise_std: float = 0.02  # Reduced for more stable results
    
    # Update model parameters
    learning_rate_base: float = 0.05
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RoundResult:
    """Results from a single round"""
    round_id: int
    method: str
    pairs: List[Tuple[int, int]]
    rewards: List[float]
    cumulative_reward: float
    avg_performance: Dict[str, float]
    heterogeneity_scores: List[float]
    computation_time: float


class ExperimentRunner:
    """Runs simulation experiments"""
    
    def __init__(self, config: ExperimentConfig, output_dir: str):
        """
        Initialize experiment runner
        
        Args:
            config: Experiment configuration
            output_dir: Directory for outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = SyntheticEnvironment(
            num_clients=config.num_clients,
            heterogeneity_level=config.heterogeneity_level,
            seed=config.seed
        )
        
        # Initialize models
        self.reward_model = GroundTruthReward(
            gamma=config.gamma,
            delta=config.delta,
            noise_std=config.noise_std
        )
        self.update_model = PerformanceUpdateModel(
            learning_rate_base=config.learning_rate_base
        )
        
        # Results storage
        self.results = {method: [] for method in config.methods}
        self.round_results = {method: [] for method in config.methods}
        
    def initialize_engines(self) -> Dict[str, Any]:
        """Initialize all pairing engines"""
        engines = {}
        
        for method in self.config.methods:
            if method == 'random':
                engines[method] = RandomBaselineEngine(seed=self.config.seed)
            elif method == 'hetero_greedy':
                engines[method] = HeterogeneityGreedyEngine()
            elif method == 'linucb_basic':
                engines[method] = LinUCBBasicEngine()
            elif method == 'linucb_enhanced':
                engines[method] = LinUCBEnhancedEngine()
            elif method == 'oracle':
                engines[method] = OracleEngine()
            else:
                raise ValueError(f"Unknown method: {method}")
                
        return engines
    
    def run_experiment(self):
        """Run the complete experiment"""
        logger.info(f"Starting experiment {self.config.exp_id}")
        logger.info(f"Config: {self.config.to_dict()}")
        
        # Generate clients
        clients = self.env.generate_clients()
        logger.info(f"Generated {len(clients)} clients")
        
        # Initialize engines
        engines = self.initialize_engines()
        
        # Track metrics
        metrics_history = {method: {
            'cumulative_rewards': [],
            'round_rewards': [],
            'avg_pass_at_1': [],
            'avg_pass_at_10': [],
            'avg_codebleu': [],
            'heterogeneity_utilization': [],
            'computation_times': []
        } for method in self.config.methods}
        
        # Run rounds
        for round_id in tqdm(range(self.config.num_rounds), desc="Rounds"):
            logger.info(f"\n--- Round {round_id + 1}/{self.config.num_rounds} ---")
            
            for method in self.config.methods:
                round_start = time.time()
                
                # Select pairs
                engine = engines[method]
                k_pairs = min(self.config.num_clients // 2, 10)  # Max 10 pairs
                selected_pairs = engine.select_pairs(clients, k_pairs, round_id)
                
                # Execute collaborations
                round_rewards = []
                heterogeneity_scores = []
                
                for client_i_id, client_j_id, metadata in selected_pairs:
                    client_i = clients[client_i_id]
                    client_j = clients[client_j_id]
                    
                    # Compute reward
                    reward, components = self.reward_model.compute_reward(
                        client_i, client_j, 
                        metadata.get('method', 'kd'),
                        round_id
                    )
                    
                    # Update engine
                    engine.update(client_i_id, client_j_id, reward, round_id)
                    
                    # Update client performance
                    perf_deltas_i = self.update_model.update_performance(
                        client_i, reward, client_j, metadata.get('method', 'kd')
                    )
                    perf_deltas_j = self.update_model.update_performance(
                        client_j, reward, client_i, metadata.get('method', 'kd')
                    )
                    
                    self.env.update_client_performance(client_i_id, perf_deltas_i)
                    self.env.update_client_performance(client_j_id, perf_deltas_j)
                    
                    # Update collaboration metrics
                    comm_kb = components.get('communication_kb', 2048)
                    collaboration_success = reward > 0.5
                    self.env.update_collaboration_metrics(
                        client_i_id, client_j_id, comm_kb, collaboration_success
                    )
                    
                    round_rewards.append(reward)
                    heterogeneity_scores.append(components.get('data_heterogeneity', 0))
                
                round_time = time.time() - round_start
                
                # Calculate round metrics
                avg_pass_at_1 = np.mean([c.local_pass_at_1 for c in clients])
                avg_pass_at_10 = np.mean([c.local_pass_at_10 for c in clients])
                avg_codebleu = np.mean([c.local_codebleu for c in clients])
                
                # Store metrics
                cumulative_reward = sum(engine.reward_history)
                metrics_history[method]['cumulative_rewards'].append(cumulative_reward)
                metrics_history[method]['round_rewards'].append(np.mean(round_rewards))
                metrics_history[method]['avg_pass_at_1'].append(avg_pass_at_1)
                metrics_history[method]['avg_pass_at_10'].append(avg_pass_at_10)
                metrics_history[method]['avg_codebleu'].append(avg_codebleu)
                metrics_history[method]['heterogeneity_utilization'].append(
                    np.mean(heterogeneity_scores) if heterogeneity_scores else 0
                )
                metrics_history[method]['computation_times'].append(round_time)
                
                # Store detailed round result
                round_result = RoundResult(
                    round_id=round_id,
                    method=method,
                    pairs=[(p[0], p[1]) for p in selected_pairs],
                    rewards=round_rewards,
                    cumulative_reward=cumulative_reward,
                    avg_performance={
                        'pass_at_1': avg_pass_at_1,
                        'pass_at_10': avg_pass_at_10,
                        'codebleu': avg_codebleu
                    },
                    heterogeneity_scores=heterogeneity_scores,
                    computation_time=round_time
                )
                self.round_results[method].append(round_result)
                
                logger.info(f"{method}: avg_reward={np.mean(round_rewards):.3f}, "
                          f"cumulative={cumulative_reward:.3f}, "
                          f"pass@1={avg_pass_at_1:.3f}")
        
        # Save results
        self._save_results(metrics_history)
        
        # Calculate final statistics
        final_stats = self._calculate_final_statistics(metrics_history, engines)
        self._save_final_statistics(final_stats)
        
        return metrics_history, final_stats
    
    def _save_results(self, metrics_history: Dict[str, Dict]):
        """Save experiment results"""
        # Save raw results
        results_file = self.output_dir / f"{self.config.exp_id}_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'config': self.config.to_dict(),
                'metrics_history': metrics_history,
                'round_results': self.round_results
            }, f)
        
        # Save metrics as CSV for easy analysis
        for method in self.config.methods:
            df = pd.DataFrame(metrics_history[method])
            df['round'] = range(len(df))
            csv_file = self.output_dir / f"{self.config.exp_id}_{method}_metrics.csv"
            df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _calculate_final_statistics(self, metrics_history: Dict, 
                                   engines: Dict) -> Dict[str, Any]:
        """Calculate final statistics for all methods"""
        stats = {}
        
        # Calculate oracle regret if oracle was run
        oracle_cumulative = None
        if 'oracle' in metrics_history:
            oracle_cumulative = metrics_history['oracle']['cumulative_rewards'][-1]
        
        for method in self.config.methods:
            method_stats = {
                'method': method,
                'final_cumulative_reward': metrics_history[method]['cumulative_rewards'][-1],
                'avg_round_reward': np.mean(metrics_history[method]['round_rewards']),
                'final_pass_at_1': metrics_history[method]['avg_pass_at_1'][-1],
                'final_pass_at_10': metrics_history[method]['avg_pass_at_10'][-1],
                'final_codebleu': metrics_history[method]['avg_codebleu'][-1],
                'avg_heterogeneity': np.mean(metrics_history[method]['heterogeneity_utilization']),
                'total_computation_time': sum(metrics_history[method]['computation_times']),
                'avg_round_time': np.mean(metrics_history[method]['computation_times'])
            }
            
            # Calculate regret if oracle available
            if oracle_cumulative is not None and method != 'oracle':
                method_stats['regret'] = oracle_cumulative - method_stats['final_cumulative_reward']
                method_stats['regret_ratio'] = method_stats['regret'] / oracle_cumulative
            
            # Add engine-specific statistics
            if method in engines:
                engine_stats = engines[method].get_statistics()
                method_stats.update({f"engine_{k}": v for k, v in engine_stats.items()})
            
            stats[method] = method_stats
        
        return stats
    
    def _save_final_statistics(self, stats: Dict[str, Any]):
        """Save final statistics"""
        # Save as JSON
        stats_file = self.output_dir / f"{self.config.exp_id}_final_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save summary table
        summary_df = pd.DataFrame(stats).T
        summary_file = self.output_dir / f"{self.config.exp_id}_summary.csv"
        summary_df.to_csv(summary_file)
        
        # Print summary
        logger.info("\n=== EXPERIMENT SUMMARY ===")
        for method, method_stats in stats.items():
            logger.info(f"\n{method}:")
            logger.info(f"  Final Reward: {method_stats['final_cumulative_reward']:.3f}")
            logger.info(f"  Final Pass@1: {method_stats['final_pass_at_1']:.3f}")
            if 'regret' in method_stats:
                logger.info(f"  Regret: {method_stats['regret']:.3f} "
                          f"({method_stats['regret_ratio']*100:.1f}%)")


def run_all_experiments(output_base_dir: str = "results"):
    """Run all planned experiments"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment configurations
    client_scales = [8, 32, 64]
    heterogeneity_levels = ['low', 'medium', 'high']
    methods = ['random', 'hetero_greedy', 'linucb_basic', 'linucb_enhanced', 'oracle']
    num_seeds = 5  # Reduced for testing, use 25 for final
    num_rounds = 50
    
    all_results = []
    
    for num_clients in client_scales:
        for hetero_level in heterogeneity_levels:
            for seed in range(num_seeds):
                exp_id = f"n{num_clients}_h{hetero_level}_s{seed}"
                
                config = ExperimentConfig(
                    exp_id=exp_id,
                    num_clients=num_clients,
                    num_rounds=num_rounds,
                    heterogeneity_level=hetero_level,
                    seed=seed,
                    methods=methods
                )
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Running experiment: {exp_id}")
                logger.info(f"{'='*60}")
                
                runner = ExperimentRunner(config, str(output_dir))
                metrics_history, final_stats = runner.run_experiment()
                
                # Store summary
                for method, stats in final_stats.items():
                    result = {
                        'num_clients': num_clients,
                        'heterogeneity': hetero_level,
                        'seed': seed,
                        'method': method,
                        **stats
                    }
                    all_results.append(result)
    
    # Save aggregate results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "all_results.csv", index=False)
    
    # Calculate aggregate statistics
    agg_stats = results_df.groupby(['num_clients', 'heterogeneity', 'method']).agg({
        'final_cumulative_reward': ['mean', 'std'],
        'final_pass_at_1': ['mean', 'std'],
        'regret': ['mean', 'std']
    }).round(3)
    
    agg_stats.to_csv(output_dir / "aggregate_statistics.csv")
    print(f"\nAll experiments completed. Results saved to {output_dir}")
    print("\nAggregate Statistics:")
    print(agg_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KNEXA-FL LinUCB validation experiments")
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test with reduced parameters')
    
    args = parser.parse_args()
    
    if args.quick_test:
        # Quick test configuration
        config = ExperimentConfig(
            exp_id="quick_test",
            num_clients=8,
            num_rounds=10,
            heterogeneity_level='high',
            seed=42,
            methods=['random', 'linucb_basic', 'linucb_enhanced']
        )
        
        runner = ExperimentRunner(config, args.output_dir)
        metrics_history, final_stats = runner.run_experiment()
    else:
        # Run all experiments
        run_all_experiments(args.output_dir)
#!/usr/bin/env python3
"""
FedAvg Results Analysis and Visualization
Publication-ready analysis tools for FedAvg baseline results

Author: Inderjeet Singh
Academic Standard: Research-grade analysis for peer review
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class FedAvgResultsAnalyzer:
    """
    Comprehensive analysis and visualization for FedAvg experiment results
    """
    
    def __init__(self, experiments_dir: str = "experiments_fedavg"):
        self.experiments_dir = Path(experiments_dir)
        self.runs_dir = self.experiments_dir / "runs"
        self.plots_dir = self.experiments_dir / "plots"
        self.reports_dir = self.experiments_dir / "reports"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plotting
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        sns.set_palette("husl")
        
    def load_experiment_data(self, experiment_id: str) -> Dict[str, Any]:
        """Load all data for a specific experiment"""
        exp_dir = self.runs_dir / experiment_id
        
        if not exp_dir.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
        
        data = {}
        
        # Load metadata
        metadata_file = exp_dir / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data['metadata'] = json.load(f)
        
        # Load final results
        final_results_file = exp_dir / "final_results.json"
        if final_results_file.exists():
            with open(final_results_file, 'r') as f:
                data['final_results'] = json.load(f)
        
        # Load round results
        round_results = []
        round_dir = exp_dir / "round_results"
        if round_dir.exists():
            for round_file in sorted(round_dir.glob("round_*.json")):
                try:
                    with open(round_file, 'r') as f:
                        round_data = json.load(f)
                        round_results.append(round_data)
                except Exception as e:
                    logger.warning(f"Failed to load {round_file}: {e}")
        
        data['round_results'] = round_results
        
        return data
    
    def create_performance_plot(self, experiment_id: str, save_path: Optional[str] = None) -> str:
        """Create performance over rounds plot"""
        data = self.load_experiment_data(experiment_id)
        round_results = data.get('round_results', [])
        
        if not round_results:
            logger.warning(f"No round results found for {experiment_id}")
            return None
        
        # Extract performance data
        rounds = []
        avg_performances = []
        client_performances = {}
        
        for round_data in round_results:
            rounds.append(round_data.get('round', 0))
            
            # Get average performance
            if 'aggregated_metrics' in round_data and 'avg_pass@1' in round_data['aggregated_metrics']:
                avg_performances.append(round_data['aggregated_metrics']['avg_pass@1'])
            elif 'avg_delta_perf' in round_data:
                avg_performances.append(round_data['avg_delta_perf'])
            else:
                avg_performances.append(0.0)
            
            # Get individual client performances
            if 'client_metrics' in round_data:
                for client_id, metrics in round_data['client_metrics'].items():
                    if client_id not in client_performances:
                        client_performances[client_id] = []
                    
                    perf = metrics.get('performance', metrics.get('delta_perf', 0.0))
                    client_performances[client_id].append(perf)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Average Performance
        ax1.plot(rounds, avg_performances, 'b-', linewidth=2, marker='o', label='Average Performance')
        ax1.set_xlabel('Federated Round')
        ax1.set_ylabel('Performance (pass@1)')
        ax1.set_title(f'FedAvg Performance Over Rounds - {experiment_id}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Individual Client Performances
        for client_id, performances in client_performances.items():
            if len(performances) == len(rounds):
                ax2.plot(rounds, performances, marker='s', alpha=0.7, 
                        label=f'Client {client_id.split("_")[-1]}')
        
        ax2.set_xlabel('Federated Round')
        ax2.set_ylabel('Client Performance')
        ax2.set_title('Individual Client Performance')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"{experiment_id}_performance.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plot saved to: {save_path}")
        return str(save_path)
    
    def create_convergence_analysis(self, experiment_id: str, save_path: Optional[str] = None) -> str:
        """Create convergence analysis plot"""
        data = self.load_experiment_data(experiment_id)
        round_results = data.get('round_results', [])
        
        if not round_results:
            return None
        
        # Extract convergence data
        rounds = []
        performances = []
        improvements = []
        
        for i, round_data in enumerate(round_results):
            rounds.append(round_data.get('round', i))
            
            if 'aggregated_metrics' in round_data and 'avg_pass@1' in round_data['aggregated_metrics']:
                perf = round_data['aggregated_metrics']['avg_pass@1']
            else:
                perf = round_data.get('avg_delta_perf', 0.0)
            
            performances.append(perf)
            
            if i > 0:
                improvement = perf - performances[i-1]
                improvements.append(improvement)
            else:
                improvements.append(0.0)
        
        # Create convergence plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot 1: Performance trajectory
        ax1.plot(rounds, performances, 'g-', linewidth=2, marker='o')
        ax1.set_ylabel('Performance')
        ax1.set_title(f'FedAvg Convergence Analysis - {experiment_id}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Round-to-round improvement
        ax2.bar(rounds[1:], improvements[1:], alpha=0.7, color='orange')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Round-to-Round Improvement')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence indicator (moving average of improvements)
        if len(improvements) > 3:
            window_size = min(5, len(improvements) // 2)
            moving_avg = pd.Series(improvements).rolling(window=window_size).mean()
            ax3.plot(rounds, moving_avg, 'r-', linewidth=2, label=f'{window_size}-round moving average')
            ax3.axhline(y=0.001, color='green', linestyle='--', alpha=0.7, label='Convergence threshold')
            ax3.set_xlabel('Federated Round')
            ax3.set_ylabel('Moving Avg Improvement')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"{experiment_id}_convergence.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence analysis saved to: {save_path}")
        return str(save_path)
    
    def create_client_analysis(self, experiment_id: str, save_path: Optional[str] = None) -> str:
        """Create detailed client analysis"""
        data = self.load_experiment_data(experiment_id)
        round_results = data.get('round_results', [])
        metadata = data.get('metadata', {})
        
        if not round_results:
            return None
        
        # Extract client data
        client_data = {}
        for round_data in round_results:
            if 'client_metrics' in round_data:
                for client_id, metrics in round_data['client_metrics'].items():
                    if client_id not in client_data:
                        client_data[client_id] = {
                            'performances': [],
                            'improvements': [],
                            'training_samples': []
                        }
                    
                    client_data[client_id]['performances'].append(
                        metrics.get('performance', 0.0)
                    )
                    client_data[client_id]['improvements'].append(
                        metrics.get('delta_perf', 0.0)
                    )
                    client_data[client_id]['training_samples'].append(
                        metrics.get('num_examples', 0)
                    )
        
        if not client_data:
            return None
        
        # Create client analysis plots
        num_clients = len(client_data)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Final performance by client
        client_ids = list(client_data.keys())
        final_performances = [client_data[cid]['performances'][-1] if client_data[cid]['performances'] else 0 
                            for cid in client_ids]
        
        axes[0, 0].bar([cid.split('_')[-1] for cid in client_ids], final_performances, alpha=0.7)
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Final Performance')
        axes[0, 0].set_title('Final Performance by Client')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training samples by client
        training_samples = [client_data[cid]['training_samples'][0] if client_data[cid]['training_samples'] else 0 
                           for cid in client_ids]
        
        axes[0, 1].bar([cid.split('_')[-1] for cid in client_ids], training_samples, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Client ID')
        axes[0, 1].set_ylabel('Training Samples')
        axes[0, 1].set_title('Training Data Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance correlation with data size
        axes[1, 0].scatter(training_samples, final_performances, alpha=0.7, s=100)
        axes[1, 0].set_xlabel('Training Samples')
        axes[1, 0].set_ylabel('Final Performance')
        axes[1, 0].set_title('Performance vs Data Size')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(training_samples) > 1 and len(final_performances) > 1:
            correlation = np.corrcoef(training_samples, final_performances)[0, 1]
            axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                           transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor='white'))
        
        # Plot 4: Total improvement by client
        total_improvements = [sum(client_data[cid]['improvements']) for cid in client_ids]
        
        axes[1, 1].bar([cid.split('_')[-1] for cid in client_ids], total_improvements, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Client ID')
        axes[1, 1].set_ylabel('Total Improvement')
        axes[1, 1].set_title('Cumulative Improvement by Client')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'FedAvg Client Analysis - {experiment_id}', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.plots_dir / f"{experiment_id}_client_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Client analysis saved to: {save_path}")
        return str(save_path)
    
    def generate_comprehensive_report(self, experiment_id: str) -> str:
        """Generate comprehensive analysis report with all visualizations"""
        data = self.load_experiment_data(experiment_id)
        
        # Create all visualizations
        performance_plot = self.create_performance_plot(experiment_id)
        convergence_plot = self.create_convergence_analysis(experiment_id)
        client_plot = self.create_client_analysis(experiment_id)
        
        # Calculate detailed statistics
        stats = self._calculate_detailed_statistics(data)
        
        # Generate comprehensive markdown report
        report_content = f"""# FedAvg Experiment Analysis Report

## Experiment Overview
- **Experiment ID**: {experiment_id}
- **Algorithm**: FedAvg (Federated Averaging)
- **Analysis Generated**: {datetime.now().isoformat()}

## Configuration Summary
"""
        
        if 'metadata' in data and 'config' in data['metadata']:
            config = data['metadata']['config']
            report_content += f"""
- **Rounds**: {config.get('num_rounds', 'N/A')}
- **Clients**: {config.get('num_clients', 'N/A')}
- **Local Learning Rate**: {config.get('learning_rate_local', 'N/A')}
- **Local Epochs**: {config.get('local_epochs', 'N/A')}
- **Batch Size**: {config.get('batch_size_local', 'N/A')}
- **Model Configuration**: {config.get('model_configuration', 'N/A')}
- **Random Seed**: {config.get('seed', 'N/A')}
"""
        
        report_content += f"""
## Performance Analysis

### Key Metrics
- **Final Performance**: {stats.get('final_performance', 'N/A'):.4f}
- **Best Performance**: {stats.get('best_performance', 'N/A'):.4f} (Round {stats.get('best_round', 'N/A')})
- **Average Performance**: {stats.get('avg_performance', 'N/A'):.4f}
- **Total Improvement**: {stats.get('total_improvement', 'N/A'):.4f}
- **Convergence**: {'Yes' if stats.get('converged', False) else 'No'}

### Statistical Summary
- **Performance Standard Deviation**: {stats.get('performance_std', 'N/A'):.4f}
- **Improvement Variance**: {stats.get('improvement_variance', 'N/A'):.6f}
- **Rounds to Best Performance**: {stats.get('rounds_to_best', 'N/A')}

## Client Analysis
"""
        
        if 'client_stats' in stats:
            client_stats = stats['client_stats']
            report_content += f"""
- **Most Improved Client**: Client {client_stats.get('most_improved_client', 'N/A')} ({client_stats.get('max_improvement', 'N/A'):.4f})
- **Best Performing Client**: Client {client_stats.get('best_client', 'N/A')} ({client_stats.get('max_performance', 'N/A'):.4f})
- **Data-Performance Correlation**: {client_stats.get('data_perf_correlation', 'N/A'):.3f}
"""
        
        report_content += f"""
## Visualizations

### Performance Over Time
![Performance Analysis]({performance_plot})

### Convergence Analysis  
![Convergence Analysis]({convergence_plot})

### Client Analysis
![Client Analysis]({client_plot})

## Algorithm Details
- **Aggregation Method**: Weighted averaging by dataset size
- **Parameter Sharing**: Full model parameters (via PEFT modules)
- **Communication**: Model parameters only (no raw data)
- **Privacy**: Standard FedAvg privacy guarantees

## Reproducibility Information
All experiment parameters and results are logged for full reproducibility. See experiment metadata for complete system information.

## Notes
This analysis provides a comprehensive view of the FedAvg baseline performance. Results are suitable for comparison with other federated learning algorithms and for academic publication.

---
**Analysis Author**: Inderjeet Singh  
**Report Generated**: {datetime.now().isoformat()}
"""
        
        # Save comprehensive report
        report_path = self.reports_dir / f"{experiment_id}_comprehensive_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Comprehensive analysis report saved to: {report_path}")
        return str(report_path)
    
    def _calculate_detailed_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed statistics from experiment data"""
        round_results = data.get('round_results', [])
        
        if not round_results:
            return {}
        
        # Extract performance data
        performances = []
        improvements = []
        
        for round_data in round_results:
            if 'aggregated_metrics' in round_data and 'avg_pass@1' in round_data['aggregated_metrics']:
                perf = round_data['aggregated_metrics']['avg_pass@1']
            else:
                perf = round_data.get('avg_delta_perf', 0.0)
            
            performances.append(perf)
            
            if len(performances) > 1:
                improvements.append(perf - performances[-2])
        
        stats = {}
        
        if performances:
            stats['final_performance'] = performances[-1]
            stats['best_performance'] = max(performances)
            stats['best_round'] = performances.index(max(performances))
            stats['avg_performance'] = sum(performances) / len(performances)
            stats['total_improvement'] = performances[-1] - performances[0]
            stats['performance_std'] = np.std(performances)
            stats['rounds_to_best'] = stats['best_round']
            
            # Convergence analysis
            if len(improvements) > 3:
                recent_improvements = improvements[-3:]
                stats['converged'] = all(abs(imp) < 0.001 for imp in recent_improvements)
                stats['improvement_variance'] = np.var(improvements)
        
        # Client-specific statistics
        client_stats = self._calculate_client_statistics(round_results)
        if client_stats:
            stats['client_stats'] = client_stats
        
        return stats
    
    def _calculate_client_statistics(self, round_results: List[Dict]) -> Dict[str, Any]:
        """Calculate client-specific statistics"""
        client_data = {}
        
        for round_data in round_results:
            if 'client_metrics' in round_data:
                for client_id, metrics in round_data['client_metrics'].items():
                    if client_id not in client_data:
                        client_data[client_id] = {
                            'performances': [],
                            'improvements': [],
                            'training_samples': []
                        }
                    
                    client_data[client_id]['performances'].append(
                        metrics.get('performance', 0.0)
                    )
                    client_data[client_id]['improvements'].append(
                        metrics.get('delta_perf', 0.0)
                    )
                    client_data[client_id]['training_samples'].append(
                        metrics.get('num_examples', 0)
                    )
        
        if not client_data:
            return {}
        
        stats = {}
        
        # Find best performing and most improved clients
        final_performances = {}
        total_improvements = {}
        training_samples = {}
        
        for client_id, data in client_data.items():
            if data['performances']:
                final_performances[client_id] = data['performances'][-1]
                total_improvements[client_id] = sum(data['improvements'])
                training_samples[client_id] = data['training_samples'][0] if data['training_samples'] else 0
        
        if final_performances:
            best_client = max(final_performances, key=final_performances.get)
            stats['best_client'] = best_client.split('_')[-1]
            stats['max_performance'] = final_performances[best_client]
        
        if total_improvements:
            most_improved_client = max(total_improvements, key=total_improvements.get)
            stats['most_improved_client'] = most_improved_client.split('_')[-1]
            stats['max_improvement'] = total_improvements[most_improved_client]
        
        # Calculate correlation between data size and performance
        if len(training_samples) > 1 and len(final_performances) > 1:
            sample_sizes = list(training_samples.values())
            performances = [final_performances[cid] for cid in training_samples.keys()]
            
            if len(sample_sizes) == len(performances):
                correlation = np.corrcoef(sample_sizes, performances)[0, 1]
                stats['data_perf_correlation'] = correlation
        
        return stats

# Global instance for easy access
fedavg_analyzer = FedAvgResultsAnalyzer()
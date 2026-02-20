#!/usr/bin/env python3
"""
Comprehensive Metrics Formatter for KNEXA-FL
Ensures all metrics are saved in structured text/CSV/markdown formats
"""
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsFormatter:
    """Formats and saves all experiment metrics in multiple formats"""
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.metrics_dir = self.experiment_dir / "formatted_metrics"
        self.metrics_dir.mkdir(exist_ok=True)
    
    def save_round_metrics(self, round_id: int, metrics: Dict[str, Any]):
        """Save per-round metrics in multiple formats"""
        # Extract key metrics
        client_metrics = {}
        
        # Process local results for training losses
        if 'local_results' in metrics:
            for result in metrics['local_results']:
                client_id = result.get('client_id')
                if client_id is not None:
                    client_metrics[client_id] = {
                        'local_improvement': result.get('local_improvement', 0.0),
                        'training_loss': result.get('avg_loss', 0.0),
                        'validation_loss': result.get('val_loss', 0.0)
                    }
        
        # Process final results for performance metrics
        if 'final_results' in metrics:
            for result in metrics['final_results']:
                client_id = result.get('client_id')
                if client_id is not None and client_id in client_metrics:
                    client_metrics[client_id]['final_performance'] = result.get('final_performance', 0.0)
        
        # Process strategic pass@k results
        if 'strategic_pass_at_k' in metrics:
            if 'end' in metrics['strategic_pass_at_k']:
                for client_id, pass_at_k in metrics['strategic_pass_at_k']['end'].items():
                    if client_id in client_metrics:
                        client_metrics[client_id].update({
                            'pass_at_1': pass_at_k.get('pass@1', 0.0),
                            'pass_at_5': pass_at_k.get('pass@5', 0.0),
                            'pass_at_10': pass_at_k.get('pass@10', 0.0),
                            'codebleu': pass_at_k.get('codebleu', 0.0)
                        })
        
        # Save as CSV
        csv_file = self.metrics_dir / f"round_{round_id:03d}_metrics.csv"
        if client_metrics:
            df = pd.DataFrame.from_dict(client_metrics, orient='index')
            df.index.name = 'client_id'
            df.to_csv(csv_file)
            logger.info(f"Saved round {round_id} metrics to {csv_file}")
        
        # Save as JSON
        json_file = self.metrics_dir / f"round_{round_id:03d}_metrics.json"
        with open(json_file, 'w') as f:
            json.dump({
                'round_id': round_id,
                'timestamp': datetime.now().isoformat(),
                'client_metrics': client_metrics,
                'summary': self._compute_round_summary(client_metrics)
            }, f, indent=2)
        
        # Save as Markdown
        md_file = self.metrics_dir / f"round_{round_id:03d}_metrics.md"
        self._write_round_markdown(round_id, client_metrics, md_file)
    
    def save_training_logs(self, round_id: int, client_id: int, 
                          training_losses: List[float], validation_losses: List[float]):
        """Save detailed training logs for each client"""
        log_dir = self.metrics_dir / "training_logs"
        log_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_file = log_dir / f"client_{client_id}_round_{round_id:03d}_training.csv"
        df = pd.DataFrame({
            'step': range(len(training_losses)),
            'training_loss': training_losses,
            'validation_loss': validation_losses[:len(training_losses)] if validation_losses else [None] * len(training_losses)
        })
        df.to_csv(csv_file, index=False)
        
        # Save summary
        summary_file = log_dir / f"client_{client_id}_round_{round_id:03d}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'client_id': client_id,
                'round_id': round_id,
                'num_steps': len(training_losses),
                'initial_loss': training_losses[0] if training_losses else None,
                'final_loss': training_losses[-1] if training_losses else None,
                'avg_loss': sum(training_losses) / len(training_losses) if training_losses else None,
                'min_loss': min(training_losses) if training_losses else None,
                'max_loss': max(training_losses) if training_losses else None
            }, f, indent=2)
    
    def save_codebleu_metrics(self, round_id: int, codebleu_scores: Dict[int, float]):
        """Save CodeBLEU metrics separately for easy access"""
        codebleu_dir = self.metrics_dir / "codebleu"
        codebleu_dir.mkdir(exist_ok=True)
        
        # Save as CSV
        csv_file = codebleu_dir / f"round_{round_id:03d}_codebleu.csv"
        df = pd.DataFrame(list(codebleu_scores.items()), columns=['client_id', 'codebleu_score'])
        df.to_csv(csv_file, index=False)
        
        # Save summary statistics
        summary_file = codebleu_dir / f"round_{round_id:03d}_codebleu_summary.json"
        scores = list(codebleu_scores.values())
        with open(summary_file, 'w') as f:
            json.dump({
                'round_id': round_id,
                'num_clients': len(scores),
                'mean_codebleu': sum(scores) / len(scores) if scores else 0.0,
                'min_codebleu': min(scores) if scores else 0.0,
                'max_codebleu': max(scores) if scores else 0.0,
                'std_codebleu': pd.Series(scores).std() if scores else 0.0
            }, f, indent=2)
    
    def save_final_summary(self, all_rounds_data: Dict[int, Dict[str, Any]]):
        """Save comprehensive summary across all rounds"""
        # Compile metrics across rounds
        summary_data = []
        for round_id, round_data in sorted(all_rounds_data.items()):
            round_summary = {
                'round': round_id,
                'avg_performance': 0.0,
                'avg_pass_at_1': 0.0,
                'avg_codebleu': 0.0,
                'avg_training_loss': 0.0
            }
            
            # Extract averages from round data
            if 'client_metrics' in round_data:
                client_metrics = round_data['client_metrics']
                if client_metrics:
                    perfs = [m.get('final_performance', 0) for m in client_metrics.values()]
                    pass1s = [m.get('pass_at_1', 0) for m in client_metrics.values()]
                    codebleus = [m.get('codebleu', 0) for m in client_metrics.values()]
                    losses = [m.get('training_loss', 0) for m in client_metrics.values()]
                    
                    round_summary['avg_performance'] = sum(perfs) / len(perfs) if perfs else 0
                    round_summary['avg_pass_at_1'] = sum(pass1s) / len(pass1s) if pass1s else 0
                    round_summary['avg_codebleu'] = sum(codebleus) / len(codebleus) if codebleus else 0
                    round_summary['avg_training_loss'] = sum(losses) / len(losses) if losses else 0
            
            summary_data.append(round_summary)
        
        # Save as CSV
        csv_file = self.metrics_dir / "experiment_summary.csv"
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        
        # Save as comprehensive markdown report
        md_file = self.metrics_dir / "experiment_report.md"
        self._write_experiment_report(summary_data, md_file)
    
    def _compute_round_summary(self, client_metrics: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        """Compute summary statistics for a round"""
        if not client_metrics:
            return {}
        
        summary = {}
        metric_names = ['final_performance', 'pass_at_1', 'pass_at_5', 'pass_at_10', 
                       'codebleu', 'training_loss', 'validation_loss']
        
        for metric in metric_names:
            values = [m.get(metric, 0) for m in client_metrics.values() if metric in m]
            if values:
                summary[f'avg_{metric}'] = sum(values) / len(values)
                summary[f'min_{metric}'] = min(values)
                summary[f'max_{metric}'] = max(values)
        
        return summary
    
    def _write_round_markdown(self, round_id: int, client_metrics: Dict[int, Dict[str, float]], 
                             file_path: Path):
        """Write round metrics as markdown"""
        with open(file_path, 'w') as f:
            f.write(f"# Round {round_id} Metrics\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Client metrics table
            f.write("## Client Metrics\n\n")
            f.write("| Client | Performance | Pass@1 | Pass@5 | Pass@10 | CodeBLEU | Train Loss | Val Loss |\n")
            f.write("|--------|-------------|--------|--------|---------|----------|------------|----------|\n")
            
            for client_id in sorted(client_metrics.keys()):
                m = client_metrics[client_id]
                f.write(f"| {client_id} | "
                       f"{m.get('final_performance', 0):.4f} | "
                       f"{m.get('pass_at_1', 0):.4f} | "
                       f"{m.get('pass_at_5', 0):.4f} | "
                       f"{m.get('pass_at_10', 0):.4f} | "
                       f"{m.get('codebleu', 0):.4f} | "
                       f"{m.get('training_loss', 0):.4f} | "
                       f"{m.get('validation_loss', 0):.4f} |\n")
            
            # Summary statistics
            f.write("\n## Summary Statistics\n\n")
            summary = self._compute_round_summary(client_metrics)
            for key, value in summary.items():
                f.write(f"- **{key}**: {value:.4f}\n")
    
    def _write_experiment_report(self, summary_data: List[Dict[str, float]], file_path: Path):
        """Write comprehensive experiment report as markdown"""
        with open(file_path, 'w') as f:
            f.write("# KNEXA-FL Experiment Report\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Progress over rounds
            f.write("## Progress Over Rounds\n\n")
            f.write("| Round | Avg Performance | Avg Pass@1 | Avg CodeBLEU | Avg Train Loss |\n")
            f.write("|-------|-----------------|------------|--------------|----------------|\n")
            
            for round_data in summary_data:
                f.write(f"| {round_data['round']} | "
                       f"{round_data['avg_performance']:.4f} | "
                       f"{round_data['avg_pass_at_1']:.4f} | "
                       f"{round_data['avg_codebleu']:.4f} | "
                       f"{round_data['avg_training_loss']:.4f} |\n")
            
            # Final results
            if summary_data:
                final = summary_data[-1]
                f.write("\n## Final Results\n\n")
                f.write(f"- **Final Performance**: {final['avg_performance']:.4f}\n")
                f.write(f"- **Final Pass@1**: {final['avg_pass_at_1']:.4f}\n")
                f.write(f"- **Final CodeBLEU**: {final['avg_codebleu']:.4f}\n")
                f.write(f"- **Final Training Loss**: {final['avg_training_loss']:.4f}\n")
"""
Performance Presentation Module for KNEXA-FL
Clean terminal output and tracking for academic research
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PerformancePresenter:
    """Clean, structured presentation of client performance metrics"""
    
    def __init__(self):
        self.baselines = {}  # client_id -> baseline metrics
        
    def format_performance_table(self, client_id: int, model_name: str, round_id: int, 
                                performance: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
        """Format performance metrics in a clean table"""
        
        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Header
        header = f"🤖 CLIENT {client_id} [{model_short}] - ROUND {round_id} PERFORMANCE"
        separator = "=" * 80
        
        # Dataset descriptions
        dataset_descriptions = [
            "\n📊 EVALUATION METRICS BY DATASET TYPE:",
            "─" * 80,
            "• LOCAL VALIDATION: Client's own validation split (local data distribution)",
            "• GLOBAL TEST SET: Shared test set across all clients (global distribution)",
            "• TRANSFER SET: Knowledge distillation questions (KD Loss = training loss from distillation)",
            "• TRAINING SET: Sample from client's training data (overfitting monitor)",
            "─" * 80
        ]
        
        # Performance table
        table_lines = [
            "┌─────────────────────────┬──────────┬──────────┬─────────────┬──────────────┬──────────────┐",
            "│ Dataset                 │ Loss     │ Pass@1   │ Pass@5      │ Pass@10     │ CodeBLEU     │",
            "├─────────────────────────┼──────────┼──────────┼─────────────┼──────────────┼──────────────┤"
        ]
        
        # Add rows for each dataset with clear labels
        datasets = ['train_sample', 'local_val', 'global_val', 'kd_transfer']
        dataset_names = [
            'Training Set (Sample)',
            'Local Validation Set',
            'Global Test Set',
            'Transfer Set (KD Loss)'
        ]
        
        for dataset, display_name in zip(datasets, dataset_names):
            if dataset in performance:
                metrics = performance[dataset]
                # CRITICAL: Never use default values that could be misinterpreted as real metrics
                loss = f"{metrics['loss']:.3f}" if 'loss' in metrics and metrics['loss'] is not None else "   N/A   "
                pass_1 = f"{metrics['pass_at_1']:.3f}" if 'pass_at_1' in metrics and metrics['pass_at_1'] is not None else "   N/A   "
                pass_5 = f"{metrics['pass_at_5']:.3f}" if 'pass_at_5' in metrics and metrics['pass_at_5'] is not None else "   N/A   "
                pass_10 = f"{metrics['pass_at_10']:.3f}" if 'pass_at_10' in metrics and metrics['pass_at_10'] is not None else "   N/A   "
                codebleu = f"{metrics['codebleu']:.3f}" if 'codebleu' in metrics and metrics['codebleu'] is not None else "   N/A   "
                
                row = f"│ {display_name:<23} │ {loss:>8} │ {pass_1:>8} │ {pass_5:>11} │ {pass_10:>12} │ {codebleu:>12} │"
                table_lines.append(row)
        
        table_lines.append("└─────────────────────────┴──────────┴──────────┴─────────────┴──────────────┴──────────────┘")
        
        # Improvement summary if baseline available
        improvement_text = ""
        if baseline:
            improvements = self._calculate_improvements(performance, baseline)
            if improvements:
                improvement_parts = []
                if 'local_val' in improvements:
                    local_imp = improvements['local_val']
                    if 'loss' in local_imp:
                        improvement_parts.append(f"Val Loss {local_imp['loss']:+.3f}")
                    if 'pass_at_1' in local_imp:
                        improvement_parts.append(f"Pass@1 {local_imp['pass_at_1']:+.3f}")
                    if 'codebleu' in local_imp:
                        improvement_parts.append(f"CodeBLEU {local_imp['codebleu']:+.3f}")
                
                if improvement_parts:
                    improvement_text = f"📈 FROM BASELINE: {' | '.join(improvement_parts)}"
        
        # Combine all parts
        result = f"\n{separator}\n{header}\n{separator}"
        result += "\n".join(dataset_descriptions)
        result += "\n" + "\n".join(table_lines)
        if improvement_text:
            result += f"\n{improvement_text}"
        result += f"\n{separator}\n"
        
        return result
    
    def _calculate_improvements(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate improvements from baseline"""
        improvements = {}
        
        for dataset in ['local_val', 'global_val', 'kd_transfer']:
            if dataset in current and dataset in baseline:
                curr_metrics = current[dataset]
                base_metrics = baseline[dataset]
                dataset_improvements = {}
                
                # Calculate improvements for each metric
                for metric in ['loss', 'pass_at_1', 'pass_at_5', 'codebleu']:
                    if metric in curr_metrics and metric in base_metrics:
                        curr_val = curr_metrics[metric]
                        base_val = base_metrics[metric]
                        if curr_val is not None and base_val is not None:
                            if metric == 'loss':
                                # For loss, improvement is negative (lower is better)
                                dataset_improvements[metric] = curr_val - base_val
                            else:
                                # For other metrics, improvement is positive (higher is better)
                                dataset_improvements[metric] = curr_val - base_val
                
                if dataset_improvements:
                    improvements[dataset] = dataset_improvements
        
        return improvements
    
    def format_training_stage_summary(self, client_id: int, stage: str, 
                                     pre_metrics: Dict[str, Any], post_metrics: Dict[str, Any]) -> str:
        """Format before/after summary for training stages"""
        
        model_name = f"Client {client_id}"
        stage_emoji = "🏋️" if stage == "LOCAL_TRAINING" else "🤝"
        
        summary_lines = [
            f"\n{stage_emoji} {stage} RESULTS - {model_name}",
            "=" * 50
        ]
        
        # Key metrics comparison
        if 'local_val' in pre_metrics and 'local_val' in post_metrics:
            # Ensure we have actual metrics, not defaults
            if 'loss' not in pre_metrics['local_val'] or 'loss' not in post_metrics['local_val']:
                summary_lines.append("⚠️ LOSS METRICS UNAVAILABLE")
                return "\n".join(summary_lines)
                
            pre_loss = pre_metrics['local_val']['loss']
            post_loss = post_metrics['local_val']['loss']
            pre_pass1 = pre_metrics['local_val'].get('pass_at_1')  # Can be None
            post_pass1 = post_metrics['local_val'].get('pass_at_1')  # Can be None
            
            loss_change = post_loss - pre_loss
            
            summary_lines.extend([
                f"📊 Validation Loss: {pre_loss:.4f} → {post_loss:.4f} ({loss_change:+.4f})"
            ])
            
            # Only report Pass@1 if we have actual values
            if pre_pass1 is not None and post_pass1 is not None:
                pass1_change = post_pass1 - pre_pass1
                summary_lines.append(
                    f"📈 Pass@1 Score:   {pre_pass1:.3f} → {post_pass1:.3f} ({pass1_change:+.3f})"
                )
            else:
                summary_lines.append("📈 Pass@1 Score:   Not evaluated")
            
            # Status indicator
            if loss_change < 0 and (pre_pass1 is None or post_pass1 is None or pass1_change > 0):
                summary_lines.append("✅ SUCCESSFUL IMPROVEMENT!")
            elif abs(loss_change) < 0.001 and (pre_pass1 is None or post_pass1 is None or abs(pass1_change) < 0.001):
                summary_lines.append("➖ NO SIGNIFICANT CHANGE")
            else:
                summary_lines.append("⚠️ MIXED RESULTS")
        
        summary_lines.append("=" * 50)
        return "\n".join(summary_lines)
    
    def format_round_summary(self, round_id: int, client_performances: Dict[int, Dict[str, Any]], 
                           training_integrity: Dict[int, bool]) -> str:
        """Format comprehensive round summary"""
        
        summary_lines = [
            f"\n🔄 ROUND {round_id} SUMMARY",
            "=" * 60
        ]
        
        for client_id in sorted(client_performances.keys()):
            perf = client_performances[client_id]
            integrity = training_integrity.get(client_id, True)
            
            # Client summary line
            local_val = perf.get('local_val', {})
            # Never use 0 as default - it could be misinterpreted as actual metric
            loss = local_val.get('loss')
            pass1 = local_val.get('pass_at_1')
            
            # Format with proper handling of missing values
            loss_str = f"{loss:.3f}" if loss is not None else "N/A"
            pass1_str = f"{pass1:.3f}" if pass1 is not None else "N/A"
            
            integrity_status = "✅" if integrity else "⚠️"
            summary_lines.append(
                f"Client {client_id}: Loss {loss_str} | Pass@1 {pass1_str} | Training {integrity_status}"
            )
        
        summary_lines.append("=" * 60)
        return "\n".join(summary_lines)
    
    def save_performance_summary(self, experiment_dir: Path, client_performances: Dict[int, Dict[str, Any]], 
                               baselines: Dict[int, Dict[str, Any]]) -> None:
        """Save focused performance summary to JSON"""
        
        summary = {}
        
        for client_id in client_performances:
            current = client_performances[client_id]
            baseline = baselines.get(client_id, {})
            
            client_summary = {
                "baseline": baseline,
                "final": current,
                "improvement": self._calculate_improvements(current, baseline) if baseline else {}
            }
            
            summary[f"client_{client_id}"] = client_summary
        
        # Save to focused location
        summary_file = experiment_dir / "performance_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Performance summary saved to: {summary_file}")
    
    def verify_training_integrity(self, old_params: List, new_params: List, client_id: int) -> bool:
        """Simple verification that training actually updated parameters"""
        try:
            import torch
            
            if not old_params or not new_params:
                return False
            
            if len(old_params) != len(new_params):
                return False
            
            # Check if any parameters changed
            changes_detected = False
            total_change = 0.0
            
            for old_p, new_p in zip(old_params, new_params):
                if not torch.allclose(old_p, new_p, atol=1e-8):
                    changes_detected = True
                    total_change += torch.norm(new_p - old_p).item()
            
            if changes_detected:
                logger.info(f"✅ Client {client_id}: Training integrity verified (parameter change: {total_change:.6f})")
                return True
            else:
                logger.warning(f"⚠️ Client {client_id}: No parameter changes detected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Client {client_id}: Training integrity check failed: {e}")
            return False
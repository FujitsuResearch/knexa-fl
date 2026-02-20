#!/usr/bin/env python3
"""
Structured Logging Utility for KNEXA-FL
Provides organized, visually clear logging with proper round tracking
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import threading
from enum import Enum

class LogLevel(Enum):
    """Log levels with visual indicators"""
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    PROGRESS = "🔄"
    PERFORMANCE = "📊"
    MEMORY = "💾"
    ROUND = "🔥"
    PHASE = "🎯"
    EXCHANGE = "🤝"
    TRAINING = "🏋️"
    CLIENT = "🤖"
    CPM = "🧠"
    BENCHMARK = "📈"

class LossType(Enum):
    """Categorization of loss types according to KNEXA-FL paper equations"""
    LOCAL_TASK = "local_task"          # L_i(W_0, φ_i; D_i) - Equation 225
    KD_LANGUAGE_MODEL = "kd_lm"        # L_LM(X_u, y_j) - Part of Equation 252-253  
    KD_COMBINED = "kd_combined"        # (1-α_kd)L_i(D_i) + α_kd L_LM(X_u, y_j) - Equation 252-253
    VALIDATION = "validation"          # Performance evaluation losses
    REWARD_COMPUTATION = "reward"      # Pre/post loss differences - Equation 297

class DataSource(Enum):
    """Data sources for loss computation"""
    PRIVATE_DATA = "private_data"      # Agent's private dataset D_i
    TRANSFER_SET = "transfer_set"      # Shared transfer set X_u  
    TEACHER_RESPONSES = "teacher_responses"  # Generated responses y_j
    VALIDATION_SET = "validation_set"  # Validation data

class KnexaLogger:
    """
    Structured logger for KNEXA-FL with proper round tracking and visual formatting
    """
    
    SECTION_WIDTH = 120
    INDENT = "    "
    
    def __init__(self, name: str = "KNEXA-FL", log_level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Thread-safe round tracking
        self._lock = threading.Lock()
        self._current_round = -1
        self._current_phase = ""
        self._phase_start_time = None
        self._round_start_time = None
        
        # Basic formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Clear existing handlers and add new one
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        self.logger.propagate = True  # Allow propagation to capture in file logs
        
    def info(self, message: str, indent_level: int = 0):
        """Log info message"""
        indent = self.INDENT * indent_level
        self.logger.info(f"{indent}{message}")
        
    def warning(self, message: str, indent_level: int = 0):
        """Log warning message"""
        indent = self.INDENT * indent_level
        self.logger.warning(f"{indent}{message}")
        
    def error(self, message: str, exception: Exception = None, indent_level: int = 0):
        """Log error message"""
        indent = self.INDENT * indent_level
        if exception:
            self.logger.error(f"{indent}{message}: {exception}")
        else:
            self.logger.error(f"{indent}{message}")
        
    def debug(self, message: str, indent_level: int = 0):
        """Log debug message"""
        indent = self.INDENT * indent_level
        self.logger.debug(f"{indent}{message}")

    def round_start(self, round_id: int, total_rounds: int, **kwargs):
        """Start a new round with comprehensive setup"""
        with self._lock:
            self._current_round = round_id
            self._round_start_time = time.time()
        
        self.logger.info("=" * self.SECTION_WIDTH)
        self.logger.info(f"{LogLevel.ROUND.value}  FEDERATED ROUND {round_id:02d} / {total_rounds}            (true P2P, eval=strategic)")
        self.logger.info("=" * self.SECTION_WIDTH)
        
        if kwargs:
            config_str = ", ".join([f"{k}: {v}" for k, v in kwargs.items()])
            self.logger.info(f"📋 Config: {config_str}")
            self.logger.info("")

    def round_end(self, round_id: int, total_rounds: int, summary: Dict[str, Any] = None):
        """End current round with summary"""
        with self._lock:
            if self._round_start_time:
                round_time = time.time() - self._round_start_time
            else:
                round_time = 0
            self._current_round = -1
        
        self.logger.info("")
        self.logger.info(f"{LogLevel.SUCCESS.value} ROUND {round_id:02d} COMPLETED in {round_time:.2f}s")
        
        if summary:
            for key, value in summary.items():
                if isinstance(value, float):
                    self.logger.info(f"   {key}: {value:.6f}")
                else:
                    self.logger.info(f"   {key}: {value}")
        
        self.logger.info("=" * self.SECTION_WIDTH)

    def phase_start(self, phase_name: str, description: str = ""):
        """Start a new phase within current round"""
        with self._lock:
            self._current_phase = phase_name
            self._phase_start_time = time.time()
        
        self.logger.info("")
        self.logger.info(f"PHASE: {phase_name} {LogLevel.PHASE.value}")
        if description:
            self.logger.info(description)

    def phase_end(self, phase_name: str, results: Dict[str, Any] = None):
        """End current phase with results"""
        with self._lock:
            if self._phase_start_time:
                phase_time = time.time() - self._phase_start_time
            else:
                phase_time = 0
            self._current_phase = ""
        
        if results:
            result_str = ", ".join([f"{k}={v}" for k, v in results.items()])
            self.logger.info(f"{LogLevel.SUCCESS.value} {phase_name} completed in {phase_time:.2f}s ({result_str})")

    def performance_metrics(self, title: str, metrics: Dict[str, Any], indent_level: int = 1):
        """Log performance metrics in organized format"""
        indent = self.INDENT * indent_level
        self.logger.info(f"{indent}📊 {title}")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 0 <= value <= 1:
                    formatted_value = f"{value:.6f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            self.logger.info(f"{indent}   {key}: {formatted_value}")

    def loss_report(self, loss_type: LossType, data_source: DataSource, loss_value: float, 
                   round_num: int, client_id: Optional[int] = None, **kwargs):
        """Report loss values with context"""
        client_info = f"[Client {client_id}]" if client_id is not None else ""
        
        # Standardized data source descriptions
        source_descriptions = {
            DataSource.PRIVATE_DATA: "LOCAL TRAIN DATA",
            DataSource.TRANSFER_SET: "KNOWLEDGE TRANSFER SET (KD)",
            DataSource.TEACHER_RESPONSES: "TEACHER-GENERATED DATA",
            DataSource.VALIDATION_SET: "LOCAL VAL DATA"
        }
        
        source_desc = source_descriptions.get(data_source, data_source.value)
        msg = f"[R{round_num:02d}]{client_info} {loss_type.value} loss on {source_desc}: {loss_value:.6f}"
        
        if kwargs:
            info_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            msg += f" [{info_str}]"
            
        self.info(msg, 1)
    
    def training_loss_report(self, client_id: int, round_num: int, epoch: int, 
                           avg_loss: float, steps: int, data_source: str = "LOCAL TRAIN DATA", **kwargs):
        """Report training loss for a specific client/epoch"""
        msg = f"[R{round_num:02d}][Client {client_id}][Epoch {epoch}] Training loss on {data_source}: {avg_loss:.6f} (steps: {steps})"
        
        if kwargs:
            info_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in kwargs.items()])
            msg += f" [{info_str}]"
            
        self.info(msg, 2)
    
    def codebleu_report(self, client_id: int, round_num: int, codebleu_score: float, 
                       dataset_type: str = "validation", **kwargs):
        """Report CodeBLEU metrics for a client"""
        # Standardize dataset type descriptions
        dataset_descriptions = {
            "validation": "LOCAL VAL DATA",
            "local_val": "LOCAL VAL DATA",
            "global_test": "GLOBAL TEST SET",
            "transfer": "KNOWLEDGE TRANSFER SET (KD)",
            "transfer_set": "KNOWLEDGE TRANSFER SET (KD)",
            "train": "LOCAL TRAIN DATA",
            "training": "LOCAL TRAIN DATA"
        }
        
        dataset_desc = dataset_descriptions.get(dataset_type.lower(), dataset_type.upper())
        msg = f"[R{round_num:02d}][Client {client_id}] CodeBLEU on {dataset_desc}: {codebleu_score:.6f}"
        
        if kwargs:
            info_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in kwargs.items()])
            msg += f" [{info_str}]"
            
        self.info(msg, 2)
    
    def pass_at_k_report(self, client_id: int, round_num: int, pass_at_k_scores: Dict[str, float],
                        dataset_type: str = "validation", **kwargs):
        """Report Pass@k metrics for a client with clear data source"""
        # Standardize dataset type descriptions
        dataset_descriptions = {
            "validation": "LOCAL VAL DATA",
            "local_val": "LOCAL VAL DATA", 
            "global_test": "GLOBAL TEST SET",
            "transfer": "KNOWLEDGE TRANSFER SET (KD)",
            "transfer_set": "KNOWLEDGE TRANSFER SET (KD)",
            "train": "LOCAL TRAIN DATA",
            "training": "LOCAL TRAIN DATA"
        }
        
        dataset_desc = dataset_descriptions.get(dataset_type.lower(), dataset_type.upper())
        
        # Format Pass@k scores
        scores_str = ", ".join([f"{k}={v:.3f}" for k, v in pass_at_k_scores.items()])
        msg = f"[R{round_num:02d}][Client {client_id}] Pass@k on {dataset_desc}: {scores_str}"
        
        if kwargs:
            info_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            msg += f" [{info_str}]"
            
        self.info(msg, 2)

    def loss_validation_check(self, loss_value: float, client_id: Optional[int] = None, 
                            loss_type: str = "training", threshold: float = 100.0,
                            expected_range: Optional[tuple] = None):
        """Validate loss values for academic integrity
        
        Args:
            loss_value: The numeric loss to validate.
            client_id: Optional identifier of the client being validated.
            loss_type: Type/description of the loss value (e.g. ``training``, ``validation``).
            threshold: Legacy upper bound to compare against when ``expected_range`` is *not* supplied.
            expected_range: Optional ``(lower, upper)`` tuple that overrides the default
                bounds.  When provided the loss is checked against these limits instead
                of the implicit ``[0, threshold]`` interval.
        """
        client_info = f"[Client {client_id}]" if client_id is not None else ""
        
        # Resolve validation bounds
        if expected_range and len(expected_range) == 2:
            lower, upper = expected_range
        else:
            lower, upper = 0.0, threshold
        
        # Perform sanity checks
        if loss_value < lower:
            self.error(f"INVALID LOSS{client_info}: {loss_type} loss below lower bound {lower}: {loss_value}")
        elif loss_value > upper:
            self.warning(f"HIGH LOSS{client_info}: {loss_type} loss {loss_value:.6f} exceeds upper bound {upper}")
        elif loss_value == 0.0:
            self.warning(f"ZERO LOSS{client_info}: {loss_type} loss is exactly 0.0 (suspicious)")

    def enhanced_progress_tracking(self, phase_name: str, current: int, total: int, 
                                 client_id: Optional[int] = None, **metrics):
        """Enhanced progress tracking with metrics"""
        client_info = f"[Client {client_id}]" if client_id is not None else ""
        progress = (current / total) * 100 if total > 0 else 0
        
        msg = f"{LogLevel.PROGRESS.value} {phase_name}{client_info}: {current}/{total} ({progress:.1f}%)"
        
        if metrics:
            metric_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items()])
            msg += f" [{metric_str}]"
        
        self.info(msg, 1)

    def comprehensive_round_summary(self, round_id: int, metrics: Dict[str, Any], 
                                  previous_round_metrics: Optional[Dict[str, Any]] = None):
        """Generate comprehensive round summary"""
        self.logger.info("")
        self.logger.info(f"{LogLevel.BENCHMARK.value} COMPREHENSIVE ROUND {round_id} SUMMARY")
        self.logger.info("-" * 80)
        
        # Current round metrics
        if "client_metrics" in metrics:
            for client_id, client_metrics in metrics["client_metrics"].items():
                self.logger.info(f"Client {client_id} Metrics:")
                for metric, value in client_metrics.items():
                    if isinstance(value, float):
                        self.logger.info(f"   {metric}: {value:.6f}")
                    else:
                        self.logger.info(f"   {metric}: {value}")
        
        # Global metrics
        if "global_metrics" in metrics:
            self.logger.info("Global Metrics:")
            for metric, value in metrics["global_metrics"].items():
                if isinstance(value, float):
                    self.logger.info(f"   {metric}: {value:.6f}")
                else:
                    self.logger.info(f"   {metric}: {value}")

    def round_summary_table(self, round_id: int, client_metrics: Dict[int, Dict[str, Any]], 
                          global_metrics: Optional[Dict[str, Any]] = None):
        """Generate round summary table"""
        self.logger.info("")
        self.logger.info(f"📋 ROUND {round_id} SUMMARY TABLE")
        self.logger.info("-" * 60)
        
        # Client summary
        for client_id, metrics in client_metrics.items():
            self.logger.info(f"Client {client_id}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"   {metric}: {value:.6f}")
                else:
                    self.logger.info(f"   {metric}: {value}")
        
        # Global summary
        if global_metrics:
            self.logger.info("Global:")
            for metric, value in global_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"   {metric}: {value:.6f}")
                else:
                    self.logger.info(f"   {metric}: {value}")

    def knowledge_exchange(self, teacher_id: int, student_id: int, 
                         exchange_type: str = "distillation", **metrics):
        """Log knowledge exchange details"""
        msg = f"{LogLevel.EXCHANGE.value} Knowledge Exchange: Teacher {teacher_id} → Student {student_id} ({exchange_type})"
        
        if metrics:
            metric_str = ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in metrics.items()])
            msg += f" [{metric_str}]"
        
        self.info(msg, 1)

    def cpm_pairing_visualization(self, round_id: int, pairings: List[tuple], 
                                estimated_rewards: Optional[List] = None):
        """Log CPM pairing decisions"""
        self.logger.info(f"{LogLevel.CPM.value} CPM Pairings for Round {round_id}:")
        
        for i, pairing in enumerate(pairings):
            if len(pairing) == 4:
                # Full pairing: (student_id, teacher_id, alpha, temperature)
                student, teacher, alpha, temperature = pairing
                reward_str = ""
                if estimated_rewards and i < len(estimated_rewards):
                    reward = estimated_rewards[i]
                    reward_str = f" (estimated reward: {reward:.6f})"
                self.logger.info(f"   Pairing {i+1}: Teacher {teacher} → Student {student} (α={alpha:.3f}, T={temperature:.1f}){reward_str}")
            elif len(pairing) == 2:
                # Simple pairing: (teacher, student)
                teacher, student = pairing
                reward_str = ""
                if estimated_rewards and i < len(estimated_rewards):
                    reward = estimated_rewards[i]
                    reward_str = f" (estimated reward: {reward:.6f})"
                self.logger.info(f"   Pairing {i+1}: Teacher {teacher} → Student {student}{reward_str}")
            else:
                self.logger.info(f"   Pairing {i+1}: {pairing}")

    def cpm_decision(self, decision: str, details: Optional[Dict[str, Any]] = None):
        """Log CPM decision with details"""
        msg = f"{LogLevel.CPM.value} CPM Decision: {decision}"
        
        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            msg += f" [{detail_str}]"
        
        self.info(msg, 1)

    def memory_usage(self, memory_info: Dict[str, Any], context: str = ""):
        """Log memory usage information"""
        context_str = f" ({context})" if context else ""
        self.logger.info(f"{LogLevel.MEMORY.value} Memory Usage{context_str}:")
        
        for key, value in memory_info.items():
            if isinstance(value, float):
                self.logger.info(f"   {key}: {value:.2f} MB")
            else:
                self.logger.info(f"   {key}: {value}")

    def metrics_comparison_table(self, title: str, current_metrics: Dict[str, Any], 
                               previous_metrics: Optional[Dict[str, Any]] = None):
        """Generate metrics comparison table"""
        self.logger.info(f"📊 {title}")
        
        for key, current_value in current_metrics.items():
            if previous_metrics and key in previous_metrics:
                prev_value = previous_metrics[key]
                if isinstance(current_value, float) and isinstance(prev_value, float):
                    change = current_value - prev_value
                    self.logger.info(f"   {key}: {current_value:.6f} (Δ{change:+.6f})")
                else:
                    self.logger.info(f"   {key}: {current_value} (was {prev_value})")
            else:
                if isinstance(current_value, float):
                    self.logger.info(f"   {key}: {current_value:.6f}")
                else:
                    self.logger.info(f"   {key}: {current_value}")

    def loss_improvement_tracker(self, client_id: int, loss_type: str, 
                               initial_loss: float, current_loss: float):
        """Track loss improvements over time"""
        improvement = initial_loss - current_loss
        improvement_pct = (improvement / initial_loss * 100) if initial_loss > 0 else 0
        
        self.logger.info(f"📉 Client {client_id} {loss_type} Loss Improvement:")
        self.logger.info(f"   Initial: {initial_loss:.6f}")
        self.logger.info(f"   Current: {current_loss:.6f}")
        self.logger.info(f"   Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")

    def detect_synthetic_performance_gains(self, performance_gains: List[float], 
                                         client_id: Optional[int] = None,
                                         threshold_patterns: bool = True) -> bool:
        """
        Enhanced detection of synthetic performance gain patterns
        Returns True if synthetic patterns are detected, False otherwise
        """
        if not performance_gains:
            return False
            
        client_info = f" [Client {client_id}]" if client_id is not None else ""
        
        # Check for exact zero values (common in placeholders)
        zero_count = sum(1 for gain in performance_gains if gain == 0.0)
        if zero_count > len(performance_gains) * 0.5:  # More than 50% zeros
            self.error(f"SYNTHETIC DATA DETECTED{client_info}: {zero_count}/{len(performance_gains)} performance gains are exactly 0.0")
            return True
            
        # Check for mathematical progression patterns (synthetic generation)
        if len(performance_gains) >= 3:
            # Check for arithmetic progression
            diffs = [performance_gains[i+1] - performance_gains[i] for i in range(len(performance_gains)-1)]
            if len(set(diffs)) == 1 and diffs[0] != 0:  # Constant difference, non-zero
                self.error(f"SYNTHETIC DATA DETECTED{client_info}: Performance gains follow arithmetic progression (diff={diffs[0]:.6f})")
                return True
        
        return False

# Global instance for easy access
_global_logger = None

def get_structured_logger(name: str = "KNEXA-FL") -> KnexaLogger:
    """Get the global structured logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = KnexaLogger(name)
    return _global_logger

def init_structured_logging(name: str = "KNEXA-FL", level: int = logging.INFO) -> KnexaLogger:
    """Initialize structured logging for KNEXA-FL with noise reduction"""
    global _global_logger
    _global_logger = KnexaLogger(name, level)
    
    # Reduce noise from external libraries
    external_loggers = [
        'datasets',
        'transformers',
        'tokenizers',
        'huggingface_hub',
        'urllib3',
        'requests',
        'matplotlib',
        'PIL'
    ]
    
    for logger_name in external_loggers:
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Special case for transformers to reduce cache warnings
    transformers_logger = logging.getLogger('transformers.tokenization_utils_base')
    transformers_logger.setLevel(logging.ERROR)
    
    return _global_logger

# Export key enums for use in other modules
__all__ = [
    'KnexaLogger', 
    'LossType', 
    'DataSource',
    'get_structured_logger', 
    'init_structured_logging'
]

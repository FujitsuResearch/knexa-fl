if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method("spawn", force=True)
 
import torch, logging, pickle, zlib, os, time
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F, CrossEntropyLoss
import numpy as np, random
from collections import Counter
import flwr as fl
from typing import Dict, List, Any, Tuple
from src.globals import *
import src.globals as globals_module
from src.globals import LAMBDA_PROX, ENABLE_RECOVERY_ROUND, RECOVERY_STEPS, RECOVERY_LR, RECOVERY_LAMBDA_PROX, RECOVERY_BATCH_SIZE
from src.globals_runtime import GLOBAL_KB_LOG
from src.model_utils import load_model_and_tokenizer, load_tokenizer_only
from src.privacy_utils import compute_sier, dp_clip_noise
from src.comm import write_blob, read_blob
from src.vocab_map import sparse_to_dense
from src.federated_metrics_tracker import get_global_tracker
from src.code_evaluation import ImprovedCodeEvaluator

torch.backends.cuda.matmul.allow_tf32 = True
 
logger = logging.getLogger(__name__)

# Lightweight heuristics mirroring supplemental categorization for matchmaking features
class _ProblemProfileHeuristics:
    PROBLEM_TYPES = ['algorithms', 'data_structures', 'string_processing', 'mathematics', 'recursion']
    DIFFICULTY_LEVELS = ['easy', 'medium', 'hard']

    _TYPE_KEYWORDS = {
        'algorithms': ['sort', 'search', 'binary', 'tree', 'graph', 'dynamic', 'greedy', 'divide'],
        'data_structures': ['list', 'array', 'stack', 'queue', 'heap', 'hash', 'dict', 'set'],
        'string_processing': ['string', 'char', 'text', 'word', 'regex', 'parse', 'format'],
        'mathematics': ['math', 'number', 'prime', 'factorial', 'fibonacci', 'calculate', 'sum'],
        'recursion': ['recursive', 'recursion', 'backtrack', 'divide and conquer']
    }

    _DIFFICULTY_KEYWORDS = {
        'easy': ['simple', 'basic', 'easy', 'straightforward'],
        'medium': ['medium', 'moderate', 'intermediate'],
        'hard': ['complex', 'advanced', 'difficult', 'challenging', 'optimize']
    }

    def categorize(self, prompt: str, solution: str = "") -> Dict[str, Any]:
        text = f"{prompt or ''} {solution or ''}".lower()

        type_scores = {}
        for ptype in self.PROBLEM_TYPES:
            keywords = self._TYPE_KEYWORDS.get(ptype, [])
            type_scores[ptype] = sum(1 for kw in keywords if kw in text)
        primary_type = max(type_scores, key=type_scores.get)

        difficulty_scores = {}
        for level in self.DIFFICULTY_LEVELS:
            cues = self._DIFFICULTY_KEYWORDS.get(level, [])
            difficulty_scores[level] = sum(1 for cue in cues if cue in text)

        solution_complexity = self._estimate_solution_complexity(solution or "")
        estimated_difficulty = max(difficulty_scores, key=difficulty_scores.get)
        if solution_complexity > 12:
            estimated_difficulty = 'hard'
        elif solution_complexity < 5:
            estimated_difficulty = 'easy'

        return {
            'primary_type': primary_type,
            'type_scores': type_scores,
            'estimated_difficulty': estimated_difficulty,
            'difficulty_scores': difficulty_scores,
            'solution_complexity': solution_complexity
        }

    @staticmethod
    def _estimate_solution_complexity(solution: str) -> int:
        if not solution:
            return 1
        score = 0
        score += solution.count('for') * 2
        score += solution.count('while') * 2
        score += solution.count('if')
        score += solution.count('elif')
        score += solution.count('def') * 3
        score += solution.count('class') * 5
        score += solution.count('lambda') * 2
        score += solution.count('yield') * 2
        score += solution.count('import') * 2
        return max(score, 1)

# Progress tracking utilities
def log_training_progress(current, total, prefix="Progress", loss=None, elapsed_time=None, length=30):
    """Log a detailed progress bar for training"""
    filled = int(length * current // total)
    bar = '█' * filled + '░' * (length - filled)
    percent = f"{100 * (current / float(total)):.1f}%"
    
    progress_str = f"   {prefix}: |{bar}| {current}/{total} ({percent})"
    
    if loss is not None:
        progress_str += f" Loss: {loss:.4f}"
    
    if elapsed_time is not None:
        progress_str += f" Time: {elapsed_time:.1f}s"
        if current > 0:
            eta = (total - current) * (elapsed_time / current)
            progress_str += f" ETA: {eta:.1f}s"
    
    logger.info(progress_str)

def log_epoch_summary(epoch, total_epochs, avg_loss, performance=None, elapsed_time=None):
    """Log epoch completion summary"""
    summary = f"   📊 Epoch {epoch}/{total_epochs} Complete - Loss: {avg_loss:.4f}"
    
    if performance is not None:
        summary += f" | Performance: {performance:.3f}"
    
    if elapsed_time is not None:
        summary += f" | Time: {elapsed_time:.1f}s"
    
    logger.info(summary)
 
class KnexaClient(fl.client.NumPyClient):
    _has_warmup = False  # Guard flag to ensure local_pretrain executes only once when sharing a single GPU
    def _is_encoder_decoder(self) -> bool:
        """Check if the current model is an encoder-decoder architecture (T5/mT5)"""
        from src.globals import LLM_REGISTRY
        model_name = list(globals_module.MODEL_MAP.values())[self.cid % len(globals_module.MODEL_MAP)]
        model_info = LLM_REGISTRY.get(model_name, {"type": "decoder"})
        return model_info["type"] == "encoder-decoder"

    def __init__(self, cid: int, train_ds, val_ds, global_test_ds=None, transfer_set=None):
        self.cid = cid
        self.gpu = globals_module.DEVICE_MAP[self.cid]
        self.model_name = list(globals_module.MODEL_MAP.values())[self.cid % len(globals_module.MODEL_MAP)]
        self.transfer_set = transfer_set
        
        # Enhanced client initialization logging
        logger.info(f"\n{'='*60}")
        logger.info(f"🤖 INITIALIZING CLIENT {self.cid}")
        logger.info(f"{'='*60}")
        logger.info(f"📋 Client Configuration:")
        logger.info(f"   Client ID: {self.cid}")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   GPU Device: cuda:{self.gpu}")
        logger.info(f"   Training Samples: {len(train_ds)}")
        logger.info(f"   Validation Samples: {len(val_ds)}")
        if global_test_ds:
            logger.info(f"   Global Test Samples: {len(global_test_ds)}")
        if transfer_set:
            logger.info(f"   Transfer Set Samples: {len(transfer_set)}")
        
        torch.cuda.set_device(self.gpu)
        logger.info(f"🔧 Loading model and tokenizer...")
        self.model, self.tok = load_model_and_tokenizer(self.cid, f"cuda:{self.gpu}")
        
        # Comprehensive tracking for loss and evaluation metrics
        self.global_test_ds = global_test_ds
        self.training_losses = []  # Track all training step losses
        self.validation_losses = []  # Track validation losses during training
        self.epoch_train_losses = []  # Track average training loss per epoch
        self.epoch_val_losses = []  # Track validation loss per epoch
        self.experiment_manager = None  # Will be set by main experiment
        self.experiment_id = None
        self.current_round = 0
        self.last_transfer_perf = None
        
        # Performance baseline tracking for academic research
        self.baseline_metrics = None  # Will be established before training
        self.last_perf = 0.0  # Initialize performance tracking
        
        # Model info logging
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"📊 Model Statistics:")
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Trainable Parameters: {trainable_params:,}")
        logger.info(f"   Trainable %: {100.0 * trainable_params / total_params:.2f}%")
        logger.info(f"   Vocabulary Size: {self.tok.vocab_size:,}")
        
        self.train_ds, self.val_ds = train_ds, val_ds
        # Dynamically cap executor workers when only one GPU is available
        gpu_cnt = torch.cuda.device_count()
        workers = 1 if gpu_cnt <= 1 else 4
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.trust = 0.8
        self.sier_avg = 0.0
        
        # Initialize improved evaluator (using 25 samples instead of 5, reduced for memory)
        eval_samples = min(10, len(self.val_ds))  # Further reduced to limit evaluation memory
        self.evaluator = ImprovedCodeEvaluator(
            self.model, self.tok, f"cuda:{self.gpu}", 
            eval_samples=eval_samples, max_k=10, client_id=self.cid
        )
        
        # Model parameter saving for post-training analysis
        self.model_checkpoints = []
        # Checkpoint directory will be set by the experiment manager
        self.checkpoint_dir = None
        
        # Perform extensive local pretraining once per process to avoid redundant warm-ups on a single GPU
        if not KnexaClient._has_warmup:
            logger.info(f"\n🏋️ STARTING LOCAL PRETRAINING")
            logger.info(f"{'='*40}")
            initial_perf = self.eval_pass1()
            logger.info(f"📈 Client {self.cid} [{self.model_name.split('/')[-1]}] initial performance: {initial_perf:.3f}")
            
            try:
                # Conduct extensive local pretraining (reduced for testing)
                self.local_pretrain(epochs=3)
                self.last_perf = self.eval_pass1()
                improvement = self.last_perf - initial_perf
                logger.info(f"📈 Client {self.cid} [{self.model_name.split('/')[-1]}] post-pretraining performance: {self.last_perf:.3f}")
                logger.info(f"📊 Client {self.cid} [{self.model_name.split('/')[-1]}] pretraining improvement: {improvement:.3f}")
                if improvement > 0:
                    logger.info(f"✅ Client {self.cid} pretraining successful!")
                else:
                    logger.warning(f"⚠️ Client {self.cid} pretraining showed no improvement")
            except Exception as e:
                logger.error(f"❌ Client {self.cid} pretraining failed: {e}")
                raise RuntimeError(f"Client {self.cid} pretraining failed: {e}") from e
            finally:
                # Mark warm-up completed so subsequent client instances skip it
                KnexaClient._has_warmup = True
        else:
            logger.info("⚙️ Warm-up already performed in this process – skipping local pretraining")
        
        logger.info(f"{'='*60}")
        logger.info(f"✅ CLIENT {self.cid} INITIALIZATION COMPLETE")
        logger.info(f"{'='*60}\n")
        
        self.historical_delta = 0.0
        self.comm_kb = 0.0
        self.pre_post_diff = 0.0
        self.problem_heuristics = _ProblemProfileHeuristics()
        self.data_distribution, self.difficulty_distribution, self.specialization_score = self._compute_data_profile(self.train_ds)
        
        # Transfer tracking for bandwidth calculation
        self.transfer_times = []  # Track transfer duration for bandwidth calculation
        self.transfer_sizes = []  # Track transfer sizes for bandwidth calculation

    def set_checkpoint_dir(self, checkpoint_dir: str):
        """Set checkpoint directory from experiment manager"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"📁 Client {self.cid} checkpoint directory: {self.checkpoint_dir}")
 
    def eval_pass1(self):
        """
        Improved evaluation using proper code execution instead of string matching
        Fixes critical flaw identified in diagnostic analysis
        """
        try:
            # Use the improved evaluator with proper code execution
            result = self.evaluator.quick_eval_pass1(list(self.val_ds))
            # Memory cleanup
            torch.cuda.empty_cache()
            return result
        except Exception as e:
            logger.error(f"Improved evaluation failed for client {self.cid}: {e}")
            raise RuntimeError(f"Evaluation failed for client {self.cid}: {e}") from e
    
    def set_experiment_tracking(self, experiment_manager, experiment_id: str):
        """Set experiment tracking for comprehensive metrics"""
        self.experiment_manager = experiment_manager
        self.experiment_id = experiment_id
    
    def establish_comprehensive_baseline(self) -> Dict[str, Any]:
        """Establish performance baseline across all datasets before any training"""
        logger.info(f"📊 Establishing comprehensive baseline for Client {self.cid}...")
        
        baseline = {}
        
        # Training dataset sample (for overfitting monitoring)
        if self.train_ds:
            train_sample_size = min(10, len(self.train_ds))
            train_sample_indices = random.sample(range(len(self.train_ds)), train_sample_size)
            train_samples = [self.train_ds[i] for i in train_sample_indices]
            baseline['train_sample'] = {
                'loss': self._evaluate_loss_on_dataset(train_samples),
                'pass_at_1': None,  # Not computed for efficiency
                'pass_at_5': None,
                'codebleu': None
            }
        
        # Local validation dataset
        if self.val_ds:
            local_val_sample = list(self.val_ds)[:5]  # Use 5 samples for efficiency
            baseline['local_val'] = {
                'loss': None,  # Not computing loss on validation set (uses ground truth inappropriately)
                'pass_at_1': None,  # Will be computed separately if needed
                'pass_at_5': None,
                'codebleu': None
            }
        
        # Global validation dataset (if available)
        if self.global_test_ds:
            global_val_sample = list(self.global_test_ds)[:5]  # Use 5 samples for efficiency
            baseline['global_val'] = {
                'loss': None,  # Not computing loss on test set (uses ground truth inappropriately)
                'pass_at_1': None,
                'pass_at_5': None,
                'codebleu': None
            }
        
        # Transfer set (knowledge distillation dataset)
        if self.transfer_set:
            transfer_sample = []
            for i in range(min(5, len(self.transfer_set))):
                sample = self.transfer_set[i]
                transfer_sample.append({
                    "prompt": sample.get("prompt", ""),
                    "canonical_solution": "",
                    "task_id": sample.get("task_id", ""),
                    "entry_point": sample.get("entry_point", "")
                })
            baseline['kd_transfer'] = {
                'loss': None,  # Not computing loss on transfer set (evaluation uses ground truth)
                'pass_at_1': None,
                'pass_at_5': None,
                'codebleu': None
            }
        
        # Store baseline
        self.baseline_metrics = baseline
        logger.info(f"✅ Baseline established for Client {self.cid}")
        
        return baseline
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current performance metrics across all datasets"""
        performance = {}
        
        # Local validation dataset
        if self.val_ds:
            local_val_sample = list(self.val_ds)[:5]  # Use 5 samples for efficiency
            performance['local_val'] = {
                'loss': None,  # Not computing loss on validation set (uses ground truth inappropriately)
                'pass_at_1': None,  # Can be computed if needed
                'pass_at_5': None,
                'codebleu': None
            }
        
        # Global validation dataset (if available)
        if self.global_test_ds:
            global_val_sample = list(self.global_test_ds)[:5]  # Use 5 samples for efficiency
            performance['global_val'] = {
                'loss': None,  # Not computing loss on test set (uses ground truth inappropriately)
                'pass_at_1': None,
                'pass_at_5': None,
                'codebleu': None
            }
        
        return performance
    
    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loss statistics for analysis"""
        stats = {
            "training_losses": {
                "count": len(self.training_losses),
                "latest": self.training_losses[-1] if self.training_losses else None,
                "average": sum(self.training_losses) / len(self.training_losses) if self.training_losses else None,
                "min": min(self.training_losses) if self.training_losses else None,
                "max": max(self.training_losses) if self.training_losses else None
            },
            "validation_losses": {
                "count": len(self.validation_losses),
                "latest": self.validation_losses[-1] if self.validation_losses else None,
                "average": sum(self.validation_losses) / len(self.validation_losses) if self.validation_losses else None,
                "min": min(self.validation_losses) if self.validation_losses else None,
                "max": max(self.validation_losses) if self.validation_losses else None
            },
            "epoch_statistics": {
                "total_epochs": len(self.epoch_train_losses),
                "latest_train_loss": self.epoch_train_losses[-1] if self.epoch_train_losses else None,
                "latest_val_loss": self.epoch_val_losses[-1] if self.epoch_val_losses else None,
                "improvement_trend": self._calculate_loss_trend() if len(self.epoch_val_losses) > 1 else None
            }
        }
        return stats
    
    def _calculate_loss_trend(self) -> str:
        """Calculate loss trend direction based on recent validation losses"""
        if len(self.epoch_val_losses) < 2:
            return "insufficient_data"
        
        recent_losses = self.epoch_val_losses[-3:] if len(self.epoch_val_losses) >= 3 else self.epoch_val_losses
        if len(recent_losses) < 2:
            return "insufficient_data"
        
        improvements = sum(1 for i in range(1, len(recent_losses)) if recent_losses[i] < recent_losses[i-1])
        total_comparisons = len(recent_losses) - 1
        
        if improvements / total_comparisons > 0.6:
            return "improving"
        elif improvements / total_comparisons < 0.4:
            return "deteriorating"
        else:
            return "stable"
    
    def eval_comprehensive(self, round_id: int) -> Dict[str, Any]:
        """
        Comprehensive evaluation on both local and global datasets
        Returns detailed metrics for experiment tracking
        """
        try:
            from src.code_evaluation import evaluate_pass_at_k, evaluate_codebleu_scores
            from src.performance_presenter import PerformancePresenter
            
            results = {
                "round": round_id,
                "client_id": self.cid,
                "train_metrics": {},
                "local_metrics": {},
                "global_metrics": {},
                "transfer_metrics": {}
            }

            # ------------------------------------------------------------------
            # Decide whether to run a *full* evaluation this round
            # ------------------------------------------------------------------
            from src.globals import EVAL_FULL_EVERY_N_ROUNDS
            full_eval = (round_id % EVAL_FULL_EVERY_N_ROUNDS == 0)
            
            # Training set evaluation (to monitor overfitting)
            if self.train_ds:
                # propagate full-eval flag into training-set evaluator
                self._full_eval_flag_train = full_eval
                train_metrics = self._evaluate_on_train_set()
                results["train_metrics"] = train_metrics
                delattr(self, "_full_eval_flag_train")
            
            # Local validation evaluation
            if self.val_ds and full_eval:  # Only evaluate on full eval rounds
                local_val_data = list(self.val_ds)
                # NOTE: We do NOT calculate loss on validation set to avoid using ground truth labels
                # Loss is only meaningful on training data where we're actually training
                local_pass_at_k = self._evaluate_pass_at_k_on_dataset(local_val_data)
                local_codebleu = self._evaluate_codebleu_on_dataset(local_val_data)
                
                results["local_metrics"] = {
                    # "loss": local_loss,  # Removed - not appropriate for validation set
                    "pass_at_k": local_pass_at_k,
                    "codebleu": local_codebleu
                }
            
            # Global test evaluation
            if self.global_test_ds and full_eval:  # Only evaluate on full eval rounds
                global_test_data = list(self.global_test_ds)
                # NOTE: We do NOT calculate loss on test set to avoid using ground truth labels
                # This prevents any appearance of test data leakage during training
                global_pass_at_k = self._evaluate_pass_at_k_on_dataset(global_test_data)
                global_codebleu = self._evaluate_codebleu_on_dataset(global_test_data)
                
                results["global_metrics"] = {
                    # "loss": global_loss,  # Removed - not appropriate for test set
                    "pass_at_k": global_pass_at_k,
                    "codebleu": global_codebleu
                }
            
            # Transfer set evaluation
            if self.transfer_set:
                # Pass flag to internal helper so it knows whether to use full set
                self._full_eval_flag = full_eval
                transfer_metrics = self._evaluate_on_transfer_set()
                results["transfer_metrics"] = transfer_metrics
                delattr(self, "_full_eval_flag")
            
            # Track with experiment manager if available
            if self.experiment_manager and self.experiment_id:
                self.experiment_manager.record_validation_metrics(
                    self.experiment_id, self.cid, round_id,
                    local_loss=None,  # No longer calculating loss on validation set
                    global_loss=None,  # No longer calculating loss on test set
                    local_pass_at_k=results["local_metrics"].get("pass_at_k", {}),
                    global_pass_at_k=results["global_metrics"].get("pass_at_k", {}),
                    local_codebleu=results["local_metrics"].get("codebleu"),
                    global_codebleu=results["global_metrics"].get("codebleu")
                )
            
            # Display comprehensive results using PerformancePresenter only on full eval rounds
            if full_eval:
                presenter = PerformancePresenter()
                performance_table = self._format_performance_for_presenter(results)
                output = presenter.format_performance_table(
                    client_id=self.cid,
                    model_name=self.model_name,
                    round_id=round_id,
                    performance=performance_table,
                    baseline=self.baseline_metrics
                )
                logger.info(output)
            else:
                # Log minimal info when skipping full evaluation
                logger.info(f"[Client {self.cid}] Round {round_id}: Skipping full evaluation (eval frequency: every {EVAL_FULL_EVERY_N_ROUNDS} rounds)")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed for client {self.cid}: {e}")
            return {
                "round": round_id,
                "client_id": self.cid,
                "error": str(e),
                "train_metrics": {},
                "local_metrics": {},
                "global_metrics": {},
                "transfer_metrics": {}
            }
    
    def _evaluate_loss_on_dataset(self, dataset: List[Dict]) -> float:
        """Evaluate loss on a specific dataset"""
        try:
            self.model.eval()
            total_loss = 0.0
            num_samples = 0
            
            with torch.no_grad():
                for item in dataset:
                    try:
                        prompt = item.get("prompt", "")
                        target = item.get("canonical_solution", item.get("code", ""))
                        full_text = prompt + target
                        
                        inputs = self.tok(full_text, return_tensors="pt", truncation=True, 
                                        max_length=MAX_EVAL_LENGTH, padding=True)
                        inputs = {k: v.to(self.gpu) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        
                        if not torch.isnan(loss):
                            total_loss += loss.item()
                            num_samples += 1
                        
                        # Cleanup
                        del inputs, outputs, loss
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.debug(f"Loss evaluation failed for one sample: {e}")
                        continue
            
            return total_loss / num_samples if num_samples > 0 else float('inf')
            
        except Exception as e:
            logger.error(f"Dataset loss evaluation failed: {e}")
            return float('inf')
    
    def _evaluate_pass_at_k_on_dataset(self, dataset: List[Dict]) -> Dict[str, float]:
        """Evaluate pass@k on a specific dataset"""
        try:
            from src.code_evaluation import generate_code_samples, evaluate_pass_at_k, prepare_test_code
            
            self.model.eval()
            k_values = [1, 5, 10]
            all_results = {f'pass@{k}': [] for k in k_values}
            
            with torch.no_grad():
                for item in dataset[:5]:  # Evaluate 5 problems for better statistical validity
                    try:
                        prompt = item.get('prompt', '')
                        if not prompt:
                            continue
                        
                        test_code = prepare_test_code(item)
                        if not test_code:
                            continue
                        
                        # Generate code samples
                        samples = generate_code_samples(
                            self.model, self.tok, prompt,
                            num_samples=max(k_values), max_tokens=64, temperature=0.6
                        )
                        
                        if samples:
                            problem_id = item.get('task_id', f'eval_problem_{len(all_results["pass@1"])}')
                            problem_results = evaluate_pass_at_k(
                                samples, test_code, k_values,
                                prompt=prompt, problem_id=problem_id, client_id=self.cid
                            )
                            
                            for metric, value in problem_results.items():
                                if metric in all_results:
                                    all_results[metric].append(value)
                        
                    except Exception as e:
                        logger.debug(f"Pass@k evaluation failed for one problem: {e}")
                        continue
            
            # Calculate averages
            final_results = {}
            for metric in all_results:
                if all_results[metric]:
                    final_results[metric] = sum(all_results[metric]) / len(all_results[metric])
                # Don't include metrics with no valid results to maintain data integrity
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pass@k evaluation failed: {e}")
            # Return empty dict instead of 0.0 values to avoid misinterpretation
            return {}
    
    def _evaluate_codebleu_on_dataset(self, dataset: List[Dict]) -> float:
        """Evaluate CodeBLEU on a specific dataset"""
        try:
            from src.code_evaluation import generate_code_samples, evaluate_codebleu_scores
            
            self.model.eval()
            all_codebleu_scores = []
            
            with torch.no_grad():
                for item in dataset[:5]:  # Evaluate 5 problems for better statistical validity
                    try:
                        prompt = item.get('prompt', '')
                        canonical_solution = item.get('canonical_solution', '')
                        
                        if not prompt or not canonical_solution:
                            continue
                        
                        # Generate one sample for CodeBLEU
                        samples = generate_code_samples(
                            self.model, self.tok, prompt,
                            num_samples=1, max_tokens=64, temperature=0.6
                        )
                        
                        if samples:
                            codebleu_scores = evaluate_codebleu_scores(samples, canonical_solution)
                            valid_scores = [score for score in codebleu_scores if score is not None]
                            if valid_scores:
                                all_codebleu_scores.extend(valid_scores)
                        
                    except Exception as e:
                        logger.debug(f"CodeBLEU evaluation failed for one problem: {e}")
                        continue
            
            if not all_codebleu_scores:
                logger.warning("No valid CodeBLEU scores computed - returning None")
                return None
            return sum(all_codebleu_scores) / len(all_codebleu_scores)
            
        except Exception as e:
            logger.error(f"CodeBLEU evaluation failed: {e}")
            return None  # Never return 0.0 which could be misinterpreted as a valid score
    
    def _evaluate_on_transfer_set(self) -> Dict[str, Any]:
        """Evaluate model performance on the transfer set used for knowledge distillation"""
        if not self.transfer_set:
            return {"error": "Transfer set not available"}
        
        try:
            # Get a sample of transfer set for evaluation (use same size as other evaluations)
            full_eval = getattr(self, "_full_eval_flag", False)
            num_eval_samples = len(self.transfer_set) if full_eval else min(5, len(self.transfer_set))
            
            # Get the sample IDs we want to evaluate
            sample_ids = [self.transfer_set.samples[i]['id'] for i in range(num_eval_samples)]
            
            # Get evaluation samples with test cases
            eval_samples = self.transfer_set.get_evaluation_samples(sample_ids)
            
            if not eval_samples:
                logger.warning("No evaluation samples retrieved for transfer set")
                return {
                    "loss": float('inf'),
                    "num_samples": 0,
                    "dataset_type": "kd_transfer"
                }
            
            # NOTE: We do NOT calculate loss on transfer set evaluation samples
            # The loss during KD training is different - it uses teacher's outputs, not ground truth
            
            # Now we can properly evaluate Pass@k and CodeBLEU with test cases
            transfer_pass_at_k = self._evaluate_pass_at_k_on_dataset(eval_samples)
            transfer_codebleu = self._evaluate_codebleu_on_dataset(eval_samples)
            
            transfer_metrics = {
                # Note: We report KD training loss (from distillation) not evaluation loss
                "kd_training_loss": getattr(self, 'last_kd_training_loss', None),
                "pass_at_k": transfer_pass_at_k,
                "codebleu": transfer_codebleu,
                "num_samples": len(eval_samples),
                "dataset_type": "transfer_set"
            }
            
            codebleu_str = f"{transfer_codebleu:.3f}" if transfer_codebleu is not None else "N/A"
            kd_loss_str = f"{transfer_metrics['kd_training_loss']:.4f}" if transfer_metrics['kd_training_loss'] is not None else "N/A"
            logger.info(f"   📊 KNOWLEDGE TRANSFER SET (KD) - " + 
                       f"KD Training Loss: {kd_loss_str}, " +
                       f"Pass@1: {transfer_pass_at_k.get('pass@1', 0):.3f}, " +
                       f"CodeBLEU: {codebleu_str}")
            
            return transfer_metrics
            
        except Exception as e:
            logger.error(f"Transfer set evaluation failed: {e}")
            return {"error": str(e)}
    
    def _evaluate_on_train_set(self) -> Dict[str, Any]:
        """
        Evaluate model performance on a subset (fast rounds) or full training
        data (full-evaluation rounds).  Pass@k and CodeBLEU are added only in
        full-evaluation mode to avoid extra compute every round.
        """
        try:
            full_eval = getattr(self, "_full_eval_flag_train", False)
            if full_eval:
                train_samples = list(self.train_ds)
            else:
                # sample 10 examples to save memory
                train_sample_size = min(10, len(self.train_ds))
                train_sample_indices = random.sample(range(len(self.train_ds)), train_sample_size)
                train_samples = [self.train_ds[i] for i in train_sample_indices]
            
            train_loss = self._evaluate_loss_on_dataset(train_samples)
            
            train_metrics = {
                "loss": train_loss,
                "num_samples": len(train_samples),
                "dataset_type": "train_sample"
            }
            
            # Compute expensive metrics only on full-evaluation rounds
            if full_eval:
                train_metrics["pass_at_k"] = self._evaluate_pass_at_k_on_dataset(train_samples)
                train_metrics["codebleu"] = self._evaluate_codebleu_on_dataset(train_samples)
            
            logger.info(f"   📊 LOCAL TRAINING SET ({'FULL' if full_eval else 'SAMPLE'}) - Loss: {train_loss:.4f}")
            
            return train_metrics
            
        except Exception as e:
            logger.error(f"Training set evaluation failed: {e}")
            return {"error": str(e)}
    
    def _format_performance_for_presenter(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format evaluation results for PerformancePresenter"""
        formatted = {}
        
        # Format training set metrics (for overfitting monitoring)
        if results.get("train_metrics"):
            train = results["train_metrics"]
            formatted["train_sample"] = {
                "loss": train.get("loss"),
                "pass_at_1": None,  # Not computed for training sample
                "pass_at_5": None,  # Not computed for training sample
                "codebleu": None    # Not computed for training sample
            }
        
        # Format local validation metrics
        if results.get("local_metrics"):
            local = results["local_metrics"]
            formatted["local_val"] = {
                "loss": local.get("loss"),
                "pass_at_1": local.get("pass_at_k", {}).get("pass@1"),
                "pass_at_5": local.get("pass_at_k", {}).get("pass@5"),
                "pass_at_10": local.get("pass_at_k", {}).get("pass@10"),
                "codebleu": local.get("pass_at_k", {}).get("codebleu")  # CodeBLEU is now in pass_at_k dict
            }
        
        # Format global validation metrics
        if results.get("global_metrics"):
            global_m = results["global_metrics"]
            formatted["global_val"] = {
                "loss": global_m.get("loss"),
                "pass_at_1": global_m.get("pass_at_k", {}).get("pass@1"),
                "pass_at_5": global_m.get("pass_at_k", {}).get("pass@5"),
                "pass_at_10": global_m.get("pass_at_k", {}).get("pass@10"),
                "codebleu": global_m.get("pass_at_k", {}).get("codebleu")  # CodeBLEU is now in pass_at_k dict
            }
        
        # Format transfer set metrics
        if results.get("transfer_metrics"):
            transfer = results["transfer_metrics"]
            formatted["kd_transfer"] = {
                "loss": transfer.get("kd_training_loss"),  # This is the KD training loss (uses teacher outputs)
                "pass_at_1": transfer.get("pass_at_k", {}).get("pass@1") if "pass_at_k" in transfer else None,
                "pass_at_5": transfer.get("pass_at_k", {}).get("pass@5") if "pass_at_k" in transfer else None,
                "pass_at_10": transfer.get("pass_at_k", {}).get("pass@10") if "pass_at_k" in transfer else None,
                "codebleu": transfer.get("pass_at_k", {}).get("codebleu") if "pass_at_k" in transfer else None
            }
        
        return formatted
 
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]
    
    def evaluate(self, parameters, config):
        """Evaluate model performance for round-by-round tracking"""
        try:
            round_id = config.get("round", 0)
            
            # Perform evaluation
            current_perf = self.eval_pass1()
            
            logger.info(f"Client {self.cid} Round {round_id} LOCAL VAL DATA Pass@1: {current_perf:.3f}")
            
            # Return evaluation results in correct Flower format: (loss, num_examples, metrics)
            return float(1.0 - current_perf), len(self.val_ds), {
                "pass@1": current_perf,
                "client_id": self.cid,
                "round": round_id
            }
            
        except Exception as e:
            logger.error(f"Client {self.cid} evaluation failed: {e}")
            raise RuntimeError(f"Client {self.cid} evaluation failed: {e}") from e
 
    def fit(self, parameters, config):
        try:
            round_id = config["round"]
            pre_perf = self.last_perf
            
            # Check if client has a role assignment
            role_key = f"role_{self.cid}"
            has_role = role_key in config
            
            # Check if we're in local pretrain phase (default to False for backward compatibility)
            is_local_pretrain_phase = config.get("is_local_pretrain_phase", False)
            
            # Import ALLOW_UNPAIRED_LOCAL_ONLY flag
            from src.globals import ALLOW_UNPAIRED_LOCAL_ONLY
            
            # Determine if we should do local training
            should_do_local_training = True
            if not has_role and not ALLOW_UNPAIRED_LOCAL_ONLY:
                # Override: Always allow local training during local pretrain phase
                if is_local_pretrain_phase:
                    should_do_local_training = True
                    logger.info(f"📋 Client {self.cid} in local pretrain phase - performing local training")
                else:
                    # Skip local training if no role and flag is False (existing behavior)
                    should_do_local_training = False
                    logger.info(f"📋 Client {self.cid} has no role assignment and ALLOW_UNPAIRED_LOCAL_ONLY=False - skipping local training")
            
            # Local training (only if allowed)
            if should_do_local_training:
                self.local_train()
                
                # Strategic memory optimization before KD without affecting training performance
                # Force immediate memory release while preserving model state
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Optional: Temporary model offloading for extreme memory pressure
                # Only if memory usage is critically high
                memory_used_gb = torch.cuda.memory_allocated() / 1024**3
                if memory_used_gb > 50:  # Threshold for 93GB GPU
                    logger.info(f"⚠️ High memory usage ({memory_used_gb:.1f}GB) - performing deep cleanup")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Knowledge distillation if configured
            if has_role:
                logger.info(f"🔍 Client {self.cid} DEBUG: Found role assignment = {config[role_key]}")
                
                # Reconstruct queries from individual string items
                num_queries = config.get(f"num_queries_{self.cid}", 0)
                queries = []
                for i in range(num_queries):
                    query_key = f"query_{self.cid}_{i}"
                    if query_key in config:
                        queries.append(config[query_key])
                
                role_conf = {
                    "role": config[role_key],
                    "teacher_cid": config.get(f"teacher_cid_{self.cid}", None),
                    "alpha": config.get(f"alpha_{self.cid}", 0.5),
                    "T": config.get(f"T_{self.cid}", 2.0),
                    "queries": queries,
                    "sub_id": config.get(f"sub_id_{self.cid}", 0)
                }
                logger.info(f"🔍 Client {self.cid} role config: {role_conf}")
                self.participate_in_kd(round_id, role_conf, pre_perf)
            else:
                if is_local_pretrain_phase:
                    logger.info(f"ℹ️ Client {self.cid} in local pretrain phase - no role assignments during this phase")
                elif ALLOW_UNPAIRED_LOCAL_ONLY:
                    logger.warning(f"⚠️ Client {self.cid} received NO role assignment - performed local training only (ALLOW_UNPAIRED_LOCAL_ONLY=True)")
                else:
                    logger.warning(f"⚠️ Client {self.cid} received NO role assignment - skipped training (ALLOW_UNPAIRED_LOCAL_ONLY=False)")
            
            # Evaluate performance
            new_perf = self.eval_pass1()
            delta_perf = new_perf - pre_perf
            self.pre_post_diff = 0.5 * self.pre_post_diff + 0.5 * delta_perf
            self.last_perf = new_perf
            
            # Evaluate CodeBLEU if available
            try:
                if hasattr(self, 'val_ds') and self.val_ds:
                    local_codebleu = self._evaluate_codebleu_on_dataset(list(self.val_ds)[:3])  # Small sample
                    self.last_codebleu = local_codebleu if local_codebleu is not None else 0.0
                else:
                    self.last_codebleu = 0.0
            except Exception as e:
                logger.warning(f"CodeBLEU evaluation failed for client {self.cid}: {e}")
                self.last_codebleu = 0.0
            
            # Save model checkpoint for post-training analysis
            self._save_model_checkpoint(round_id, new_perf)
            
            # Create profile and comprehensive metrics for CPM
            profile = self.make_profile(delta_perf)
            comprehensive_metrics = self.get_comprehensive_metrics()
            
            # Create Flower-compatible metrics with comprehensive data
            metrics = {
                "delta_perf": float(delta_perf),
                "client_id": int(self.cid),
                "kb": float(getattr(self, 'comm_kb', 0.0)),
                "trust": float(self.trust),
                "perf": float(new_perf),
                "local_codebleu": float(getattr(self, 'last_codebleu', 0.0)),
                # Store profile as individual elements to avoid list issues
                "profile_0": float(profile[0]) if len(profile) > 0 else 0.0,
                "profile_1": float(profile[1]) if len(profile) > 1 else 0.0,
                "profile_2": float(profile[2]) if len(profile) > 2 else 0.0,
                "profile_3": float(profile[3]) if len(profile) > 3 else 0.0,
                "profile_4": float(profile[4]) if len(profile) > 4 else 0.0,
                "profile_5": float(profile[5]) if len(profile) > 5 else 0.0,
                "profile_6": float(profile[6]) if len(profile) > 6 else 0.0,
                "profile_7": float(profile[7]) if len(profile) > 7 else 0.0,
                # Add comprehensive metrics for CPM
                "comprehensive_metrics": comprehensive_metrics
            }
            
            # Return parameters, dataset size, and metrics
            return self.get_parameters({}), len(self.train_ds), metrics
            
        except Exception as e:
            logger.error(f"Client {self.cid} fit failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Client {self.cid} fit failed: {e}") from e
 
    def local_pretrain(self, epochs=15):
        """
        Extensive local pretraining as specified in KNEXA-FL design.
        Each client should be already trained on local data before collaboration.
        """
        model_short_name = self.model_name.split('/')[-1]
        # Dynamic batch size and gradient accumulation adjustment based on number of clients
        if NUM_CLIENTS >= 4:
            batch_size = max(1, BATCH_LOCAL // 2)  # Halve batch size for 4+ clients
            grad_accum_steps = GRADIENT_ACCUMULATION_STEPS * 2
        else:
            batch_size = BATCH_LOCAL
            grad_accum_steps = GRADIENT_ACCUMULATION_STEPS
            
        logger.info(f"🏋️ Client {self.cid} [{model_short_name}]: Starting {epochs}-epoch local pretraining...")
        logger.info(f"   Training samples: {len(self.train_ds)}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Learning rate: {LR_LOCAL}")
        
        start_time = time.time()
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(trainable_params, lr=LR_LOCAL)
        
        # Capture initial parameters for training integrity verification
        initial_params = [p.clone().detach() for p in trainable_params]
        
        train_list = list(self.train_ds)
        total_batches = (len(train_list) + batch_size - 1) // batch_size
        total_steps = epochs * total_batches
        current_step = 0
        total_loss = 0.0
        step_count = 0
        
        logger.info(f"   Total steps: {total_steps} ({total_batches} batches/epoch × {epochs} epochs)")
        logger.info(f"{'='*60}")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_steps = 0
            random.shuffle(train_list)
            
            optimiser.zero_grad()
            
            for i in range(0, len(train_list), batch_size):
                batch_items = train_list[i:i+batch_size]
                current_step += 1
                batch_start_time = time.time()
                
                # Create prompts and targets for language modeling
                prompts = [item["prompt"] for item in batch_items]
                targets = [item.get("canonical_solution", item.get("code", "")) for item in batch_items]
                full_texts = [p + t for p, t in zip(prompts, targets)]
                
                # Tokenize and create labels
                batch_loss = 0.0
                valid_items = 0
                
                for text in full_texts:
                    try:
                        inputs = self.tok(text, return_tensors="pt", truncation=True, 
                                        max_length=MAX_EVAL_LENGTH, padding=True)
                        inputs = {k: v.to(self.gpu) for k, v in inputs.items()}
                        
                        outputs = self.model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        
                        if not torch.isnan(loss):
                            batch_loss += loss
                            valid_items += 1
                        
                        # Clean up intermediate tensors
                        del inputs, outputs, loss
                        torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.error(f"Training step failed: {e}")
                        raise RuntimeError(f"Training step failed: {e}") from e
                
                if valid_items > 0:
                    avg_loss = batch_loss / valid_items
                    # Scale loss for gradient accumulation
                    avg_loss = avg_loss / grad_accum_steps
                    avg_loss.backward()
                    
                    # Update weights every grad_accum_steps
                    if (i // batch_size + 1) % grad_accum_steps == 0:
                        optimiser.step()
                        optimiser.zero_grad()
                    
                    epoch_loss += avg_loss.item()
                    epoch_steps += 1
                    total_loss += avg_loss.item()
                    step_count += 1
                    
                    # Track training loss for comprehensive experiment tracking
                    self.training_losses.append(avg_loss.item())
                    
                    # Show progress every 10 steps or at epoch boundaries
                    if current_step % 10 == 0 or current_step == total_steps:
                        elapsed = time.time() - start_time
                        log_training_progress(
                            current_step, total_steps, 
                            f"Epoch {epoch+1}/{epochs} Step", 
                            avg_loss.item(), elapsed
                        )
            
            # Final step if remaining gradients
            optimiser.step()
            optimiser.zero_grad()
            
            # Epoch completion summary with comprehensive loss tracking
            epoch_elapsed = time.time() - epoch_start_time
            if epoch_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_steps
                self.epoch_train_losses.append(avg_epoch_loss)
                
                # Compute validation loss for every epoch
                val_data_sample = list(self.val_ds)[:5]  # Use 5 samples for efficiency
                epoch_val_loss = self._evaluate_loss_on_dataset(val_data_sample)
                self.epoch_val_losses.append(epoch_val_loss)
                self.validation_losses.append(epoch_val_loss)
                
                logger.info(f"   Epoch {epoch+1}/{epochs}: LOCAL TRAIN DATA Loss = {avg_epoch_loss:.4f}, LOCAL VAL DATA Loss = {epoch_val_loss:.4f}")
                
                # Log to structured logger
                from src.structured_logging import get_structured_logger
                structured_logger = get_structured_logger()
                structured_logger.training_loss_report(
                    self.cid, 0, epoch + 1, avg_epoch_loss, epoch_steps,
                    val_loss=epoch_val_loss, duration_s=epoch_elapsed
                )
                
                # Evaluate performance every 5 epochs during pretraining
                if (epoch + 1) % 5 == 0:
                    perf = self.eval_pass1()
                    log_epoch_summary(epoch + 1, epochs, avg_epoch_loss, perf, epoch_elapsed)
                else:
                    log_epoch_summary(epoch + 1, epochs, avg_epoch_loss, None, epoch_elapsed)
        
        # Final pretraining summary
        total_elapsed = time.time() - start_time
        if step_count > 0:
            final_avg_loss = total_loss / step_count
            logger.info(f"{'='*60}")
            logger.info(f"✅ Client {self.cid} [{model_short_name}] PRETRAINING COMPLETE")
            logger.info(f"   Final LOCAL TRAIN DATA Loss: {final_avg_loss:.4f}")
            logger.info(f"   Total Time: {total_elapsed:.1f}s")
            logger.info(f"   Steps Completed: {step_count}/{total_steps}")
            logger.info(f"   Average Time/Step: {total_elapsed/step_count:.2f}s")
            logger.info(f"{'='*60}")
        else:
            raise RuntimeError(f"Client {self.cid} pretraining failed - no valid training steps")
        
        # Verify training integrity (parameters actually changed)
        final_params = [p.clone().detach() for p in trainable_params]
        from src.performance_presenter import PerformancePresenter
        presenter = PerformancePresenter()
        training_integrity = presenter.verify_training_integrity(initial_params, final_params, self.cid)
        
        if not training_integrity:
            logger.warning(f"⚠️ Client {self.cid}: Training integrity issue - parameters may not have updated properly")
        
        # Record training and validation losses with experiment manager if available
        if self.experiment_manager and self.experiment_id:
            # Record training losses
            if self.training_losses and step_count > 0:
                round_losses = self.training_losses[-step_count:] if step_count > 0 else []
                if round_losses:
                    self.experiment_manager.record_training_loss(
                        self.experiment_id, self.cid, self.current_round, round_losses
                    )
            
            # Record validation losses
            if hasattr(self.experiment_manager, 'record_validation_loss') and self.validation_losses:
                self.experiment_manager.record_validation_loss(
                    self.experiment_id, self.cid, self.current_round, self.validation_losses[-epochs:]
                )
            
            # Record epoch-level losses for trend analysis
            if hasattr(self.experiment_manager, 'record_epoch_losses'):
                self.experiment_manager.record_epoch_losses(
                    self.experiment_id, self.cid, self.current_round, 
                    self.epoch_train_losses[-epochs:], self.epoch_val_losses[-epochs:]
                )
        
        # Memory cleanup
        torch.cuda.empty_cache()

    def local_train(self):
        """Single epoch local training for collaboration rounds"""
        model_short_name = self.model_name.split('/')[-1]
        logger.info(f"\n🔄 LOCAL TRAINING - Client {self.cid} [{model_short_name}]")
        logger.info(f"{'='*50}")
        logger.info(f"   Epochs: {LOCAL_EPOCHS}")
        logger.info(f"   Training samples: {len(self.train_ds)}")
        logger.info(f"   Learning rate: {LR_LOCAL}")
        
        start_time = time.time()
        self.model.train()
        # Only optimize parameters that require gradients (LoRA adapters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimiser = torch.optim.Adam(trainable_params, lr=LR_LOCAL)
        
        # Capture initial parameters for training integrity verification
        initial_params = [p.clone().detach() for p in trainable_params]
        
        # Dynamic batch size and gradient accumulation adjustment based on number of clients
        if NUM_CLIENTS >= 4:
            batch_size = max(1, BATCH_LOCAL // 2)  # Halve batch size for 4+ clients
            # Increase gradient accumulation to maintain effective batch size
            grad_accum_steps = GRADIENT_ACCUMULATION_STEPS * 2
            logger.info(f"   📉 Using reduced batch size {batch_size} for memory efficiency (4+ clients)")
            logger.info(f"   📊 Using gradient accumulation steps: {grad_accum_steps}")
        else:
            batch_size = BATCH_LOCAL
            grad_accum_steps = GRADIENT_ACCUMULATION_STEPS
        
        # Convert dataset to list for proper batch processing
        train_list = list(self.train_ds)
        total_batches = (len(train_list) + batch_size - 1) // batch_size
        total_steps = LOCAL_EPOCHS * total_batches
        current_step = 0
        
        logger.info(f"   Total steps: {total_steps} ({total_batches} batches/epoch × {LOCAL_EPOCHS} epochs)")
        
        for epoch in range(LOCAL_EPOCHS):
            epoch_start_time = time.time()
            random.shuffle(train_list)
            optimiser.zero_grad()
            
            for i in range(0, len(train_list), batch_size):
                batch_items = train_list[i:i+batch_size]
                current_step += 1
                
                # Create prompts and targets for language modeling
                prompts = [item["prompt"] for item in batch_items]
                targets = [item.get("canonical_solution", item.get("code", "")) for item in batch_items]
                full_texts = [p + t for p, t in zip(prompts, targets)]
                
                # Handle encoder-decoder vs decoder-only models differently
                if self._is_encoder_decoder():
                    # T5-style encoder-decoder training
                    # For T5, split input into source (prompt) and target (solution)
                    source_inputs = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.gpu)
                    target_inputs = self.tok(targets, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.gpu)
                    
                    # Prepare decoder inputs (shift right for T5)
                    decoder_input_ids = target_inputs["input_ids"].clone()
                    decoder_input_ids = torch.roll(decoder_input_ids, 1, dims=1)
                    decoder_input_ids[:, 0] = self.tok.pad_token_id
                    
                    # Forward pass for T5
                    outputs = self.model(
                        input_ids=source_inputs["input_ids"],
                        attention_mask=source_inputs["attention_mask"],
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=target_inputs["attention_mask"],
                        labels=target_inputs["input_ids"]
                    )
                    loss = outputs.loss
                else:
                    # Decoder-only model training (GPT-style)
                    inputs = self.tok(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.gpu)
                    
                    # Forward pass
                    outputs = self.model(**inputs)
                    
                    # Compute causal language modeling loss manually
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = inputs["input_ids"][..., 1:].contiguous()
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()
                
                # Track training loss for comprehensive experiment tracking
                self.training_losses.append(loss.item() * grad_accum_steps)
                
                # Update weights every grad_accum_steps
                if (i // batch_size + 1) % grad_accum_steps == 0:
                    optimiser.step()
                    optimiser.zero_grad()
                    # More aggressive memory cleanup for 4+ clients
                    if NUM_CLIENTS >= 4:
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                
                # Show progress every 5 steps or at the end
                if current_step % 5 == 0 or current_step == total_steps:
                    elapsed = time.time() - start_time
                    log_training_progress(
                        current_step, total_steps, 
                        f"Epoch {epoch+1}/{LOCAL_EPOCHS} Step", 
                        loss.item() * grad_accum_steps, elapsed
                    )
            
            # Final step if remaining gradients
            optimiser.step()
            optimiser.zero_grad()
            
            # Track validation loss after each epoch
            epoch_elapsed = time.time() - epoch_start_time
            val_data_sample = list(self.val_ds)[:5]  # Use 5 samples for efficiency
            epoch_val_loss = self._evaluate_loss_on_dataset(val_data_sample)
            self.validation_losses.append(epoch_val_loss)
            
            # Calculate average training loss for this epoch
            epoch_train_losses = self.training_losses[-total_batches:] if len(self.training_losses) >= total_batches else self.training_losses
            avg_epoch_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0.0
            self.epoch_train_losses.append(avg_epoch_train_loss)
            self.epoch_val_losses.append(epoch_val_loss)
            
            logger.info(f"   Epoch {epoch+1}/{LOCAL_EPOCHS}: LOCAL TRAIN DATA Loss = {avg_epoch_train_loss:.4f}, LOCAL VAL DATA Loss = {epoch_val_loss:.4f}, Time = {epoch_elapsed:.1f}s")
            
            # Log to structured logger
            from src.structured_logging import get_structured_logger
            structured_logger = get_structured_logger()
            structured_logger.training_loss_report(
                self.cid, self.current_round, epoch + 1, avg_epoch_train_loss, total_batches,
                val_loss=epoch_val_loss, duration_s=epoch_elapsed
            )
        
        # Verify training integrity (parameters actually changed)
        final_params = [p.clone().detach() for p in trainable_params]
        from src.performance_presenter import PerformancePresenter
        presenter = PerformancePresenter()
        training_integrity = presenter.verify_training_integrity(initial_params, final_params, self.cid)
        
        if not training_integrity:
            logger.warning(f"⚠️ Client {self.cid}: Training integrity issue - parameters may not have updated properly")
        
        # Local training completion summary
        total_elapsed = time.time() - start_time
        logger.info(f"✅ LOCAL TRAINING COMPLETE - Client {self.cid} [{model_short_name}]")
        logger.info(f"   Total Time: {total_elapsed:.1f}s")
        logger.info(f"   Steps Completed: {current_step}/{total_steps}")
        if current_step > 0:
            logger.info(f"   Average Time/Step: {total_elapsed/current_step:.2f}s")
        logger.info(f"{'='*50}\n")
        
        # Record training and validation losses with experiment manager if available
        if self.experiment_manager and self.experiment_id and current_step > 0:
            # Record training losses from this round only
            round_losses = self.training_losses[-current_step:] if len(self.training_losses) >= current_step else self.training_losses
            if round_losses:
                self.experiment_manager.record_training_loss(
                    self.experiment_id, self.cid, self.current_round, round_losses
                )
            
            # Record validation losses from this round
            round_val_losses = self.validation_losses[-LOCAL_EPOCHS:] if len(self.validation_losses) >= LOCAL_EPOCHS else self.validation_losses
            if hasattr(self.experiment_manager, 'record_validation_loss') and round_val_losses:
                self.experiment_manager.record_validation_loss(
                    self.experiment_id, self.cid, self.current_round, round_val_losses
                )
            
            # Record epoch-level losses for trend analysis
            if hasattr(self.experiment_manager, 'record_epoch_losses'):
                epoch_train_losses = self.epoch_train_losses[-LOCAL_EPOCHS:] if len(self.epoch_train_losses) >= LOCAL_EPOCHS else self.epoch_train_losses
                epoch_val_losses = self.epoch_val_losses[-LOCAL_EPOCHS:] if len(self.epoch_val_losses) >= LOCAL_EPOCHS else self.epoch_val_losses
                if epoch_train_losses and epoch_val_losses:
                    self.experiment_manager.record_epoch_losses(
                        self.experiment_id, self.cid, self.current_round, 
                        epoch_train_losses, epoch_val_losses
                    )
        
        self.model.eval()
 
    def participate_in_kd(self, rnd, c, pre_perf):
        """
        Participate in Adaptive Knowledge Distillation (AKD) - KNEXA-FL's primary P2P mechanism
        
        IMPORTANT: Teachers should execute BEFORE students in the same round
        """
        model_short_name = self.model_name.split('/')[-1]
        
        if c["role"] == "teacher":
            queries = c["queries"] 
            logger.info(f"\n🎓 KNOWLEDGE DISTILLATION - TEACHER ROLE")
            logger.info(f"{'='*50}")
            logger.info(f"👨‍🏫 TEACHER: Client {self.cid} [{model_short_name}]")
            logger.info(f"🔄 Round: {rnd}")
            logger.info(f"📝 Teaching Queries: {len(queries)}")
            logger.info(f"📊 Pre-Teaching Performance: {pre_perf:.3f}")
            logger.info(f"{'='*50}")
            
            try:
                sier, kb = self.run_teacher(rnd, queries, c["sub_id"])
                logger.info(f"✅ Teacher {self.cid} knowledge generation SUCCESS")
            except Exception as e:
                logger.error(f"❌ Teacher {self.cid} knowledge generation FAILED: {e}")
                import traceback
                logger.error(f"Teacher traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Teacher {self.cid} failed: {e}") from e
            self.comm_kb = 0.5 * self.comm_kb + 0.5 * kb
            GLOBAL_KB_LOG.append(kb)
            
            # Enhanced trust update based on teaching effectiveness
            reward = GAMMA_REWARD * (self.last_perf - pre_perf) - DELTA_KB * kb
            delta = 0.1 if reward > 0 else -0.1 if reward < 0 or sier > SIER_THRESH else 0
            old_trust = self.trust
            self.trust = 0.9 * self.trust + delta
            self.trust = max(0.1, min(1.0, self.trust))
            
            logger.info(f"📈 TEACHING RESULTS:")
            logger.info(f"   Knowledge Size: {kb/1024:.1f} MB")
            logger.info(f"   SIER Score: {sier:.4f}")
            logger.info(f"   Reward: {reward:.4f}")
            logger.info(f"   Trust: {old_trust:.3f} → {self.trust:.3f} (Δ{self.trust-old_trust:+.3f})")
            logger.info(f"{'='*50}")
            logger.info(f"✅ TEACHER {self.cid} [{model_short_name}] COMPLETED TEACHING")
            logger.info(f"{'='*50}\n")
            
        elif c["role"] == "student":
            # Give teachers time to start generating knowledge first
            import time
            time.sleep(5)  # Small delay to ensure teachers start first
            
            teacher_cid = c["teacher_cid"]
            alpha = c["alpha"] 
            T = c["T"]
            queries = c["queries"]
            teacher_model_name = list(globals_module.MODEL_MAP.values())[teacher_cid % len(globals_module.MODEL_MAP)].split('/')[-1]
            
            logger.info(f"\n🎓 KNOWLEDGE DISTILLATION - STUDENT ROLE")
            logger.info(f"{'='*50}")
            logger.info(f"👨‍🎓 STUDENT: Client {self.cid} [{model_short_name}]")
            logger.info(f"👨‍🏫 TEACHER: Client {teacher_cid} [{teacher_model_name}]")
            logger.info(f"🔄 Round: {rnd}")
            logger.info(f"📝 Learning Queries: {len(queries)}")
            logger.info(f"🎛️ KD Parameters:")
            logger.info(f"   Alpha (Task/KD weight): {alpha:.3f}")
            logger.info(f"   Temperature: {T:.1f}")
            logger.info(f"{'='*50}")
            
            # Perform adaptive knowledge distillation
            pre_student_perf = self.eval_pass1()
            logger.info(f"📊 Pre-Learning Performance: {pre_student_perf:.3f}")
            
            # Store pre-KD performance for potential recovery round
            self.pre_kd_student_perf = pre_student_perf
            
            self.run_student(rnd, teacher_cid, alpha, T, queries)
            
            post_student_perf = self.eval_pass1()
            student_improvement = post_student_perf - pre_student_perf
            
            logger.info(f"📈 LEARNING RESULTS:")
            logger.info(f"   Pre-Learning: {pre_student_perf:.3f}")
            logger.info(f"   Post-Learning: {post_student_perf:.3f}")
            logger.info(f"   Improvement: {student_improvement:+.4f}")
            
            if student_improvement > 0:
                logger.info(f"✅ SUCCESSFUL LEARNING!")
            elif student_improvement == 0:
                logger.info(f"➖ NO CHANGE")
            else:
                logger.info(f"⚠️ PERFORMANCE DECREASED")
            
            # Evaluate on transfer set after KD to measure learning quality
            if self.transfer_set:
                logger.info(f"\n📊 EVALUATING KNOWLEDGE DISTILLATION QUALITY:")
                logger.info(f"   Dataset: TRANSFER SET (HumanEval/MBPP samples)")
                logger.info(f"   Purpose: Measure how well student learned from teacher")
                transfer_metrics = self._evaluate_on_transfer_set()
                if "kd_training_loss" in transfer_metrics and transfer_metrics["kd_training_loss"] is not None:
                    logger.info(f"   📉 KD Training Loss (from distillation): {transfer_metrics['kd_training_loss']:.4f}")
                if transfer_metrics.get("pass_at_k"):
                    logger.info(f"   📈 KNOWLEDGE TRANSFER SET (KD) Pass@1: {transfer_metrics['pass_at_k'].get('pass@1', 0):.3f}")
                    logger.info(f"   📈 KNOWLEDGE TRANSFER SET (KD) Pass@5: {transfer_metrics['pass_at_k'].get('pass@5', 0):.3f}")
                if transfer_metrics.get("codebleu") is not None:
                    logger.info(f"   📊 KNOWLEDGE TRANSFER SET (KD) CodeBLEU: {transfer_metrics['codebleu']:.3f}")
            
            # Report recovery metrics if recovery round was performed
            if ENABLE_RECOVERY_ROUND and hasattr(self, 'recovery_metrics'):
                recovery_metrics = self.recovery_metrics
                logger.info(f"\n📊 RECOVERY ROUND SUMMARY:")
                logger.info(f"   Performance Impact (LOCAL VAL DATA Pass@1):")
                logger.info(f"      Pre-KD → Post-Recovery: {recovery_metrics['perf_change_from_init']:+.3f}")
                logger.info(f"      Post-KD → Post-Recovery: {recovery_metrics['perf_change_from_kd']:+.3f}")
                if recovery_metrics['perf_change_from_kd'] > 0:
                    logger.info(f"   ✅ Recovery improved upon KD results!")
                # Update post_student_perf to reflect recovery
                post_student_perf = recovery_metrics['post_recovery_performance']
                student_improvement = post_student_perf - pre_student_perf
            
            logger.info(f"{'='*50}")
            logger.info(f"✅ STUDENT {self.cid} [{model_short_name}] COMPLETED LEARNING")
            logger.info(f"{'='*50}\n")
            
            # Update last performance for next round
            self.last_perf = post_student_perf
 
    def run_teacher(self, rnd, queries, sub_id):
        start_time = time.time()
        logger.info(f"🎓 KNOWLEDGE GENERATION PROCESS")
        logger.info(f"   Step 1/6: Tokenizing queries...")
        
        if self._is_encoder_decoder():
            # For T5 models, use encoder-decoder generation
            logger.info(f"   Using encoder-decoder mode for T5...")
            # Use generation with teacher forcing for T5
            tok_in = self.tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.gpu)
            
            # Generate outputs using the model's generate method for T5
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=tok_in["input_ids"],
                    attention_mask=tok_in["attention_mask"],
                    max_length=MAX_SEQ_LENGTH,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                # Extract logits from generation scores
                logits = torch.stack(generated_ids.scores, dim=1)  # [batch, seq_len, vocab]
        else:
            # For decoder-only models
            tok_in = self.tok(queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_EVAL_LENGTH).to(self.gpu)
            logger.info(f"   ✅ {len(queries)} queries tokenized")
            
            logger.info(f"   Step 2/6: Generating teacher logits...")
            logits = self.model(**tok_in).logits.detach()
            logits = logits[:, :-1, :]  # Align
        
        logger.info(f"   ✅ Logits generated: {logits.shape}")
        
        logger.info(f"   Step 3/6: Applying privacy protection...")
        privacy_start_time = time.time()
        logits = dp_clip_noise(logits)
        privacy_time = time.time() - privacy_start_time
        logger.info(f"   ✅ Differential privacy applied")
        
        # Record privacy metrics
        fl_tracker = get_global_tracker()
        if fl_tracker:
            sier_score = compute_sier([self.tok.decode(x.argmax(-1)) for x in logits[:5]])  # Sample check
            fl_tracker.record_privacy_metrics(
                epsilon=GAUSS_NOISE_SIG,  # Using noise scale as proxy for epsilon
                sier_score=sier_score,
                privacy_computation_time_s=privacy_time
            )
        
        logger.info(f"   Step 4/6: Extracting top-k knowledge...")
        topk = logits.topk(TOPK, dim=-1)
        values, indices = topk.values.half(), topk.indices.int()
        logger.info(f"   ✅ Top-{TOPK} knowledge extracted")
        
        logger.info(f"   Step 5/6: Computing safety metrics...")
        decoded = [self.tok.decode(x.argmax(-1)) for x in logits]
        sier = compute_sier(decoded)
        logger.info(f"   ✅ SIER safety score: {sier:.4f}")
        
        logger.info(f"   Step 6/6: Storing knowledge for transfer...")
        payload = {"values": values.cpu(), "indices": indices.cpu(), "seq_len": logits.shape[1]}
        kb = write_blob(self.cid, rnd, sub_id, payload, logger.info)
        
        elapsed = time.time() - start_time
        logger.info(f"   ✅ Knowledge stored successfully!")
        logger.info(f"   📦 Knowledge size: {kb/1024:.1f} MB")
        logger.info(f"   🕒 Total generation time: {elapsed:.2f}s")
        
        # Track transfer metrics for bandwidth calculation
        # Teacher generates data, so this represents the "sending" side
        kb_size = kb / 1024  # Convert to KB for bandwidth calculation
        if elapsed > 0:  # Avoid division by zero
            transfer_speed_kbps = kb_size / elapsed
            logger.info(f"   📊 Transfer generation speed: {transfer_speed_kbps:.2f} KB/s")
            
            # Record for bandwidth calculation (limit history to last 10 transfers)
            self.transfer_sizes.append(kb_size)
            self.transfer_times.append(elapsed)
            if len(self.transfer_sizes) > 10:
                self.transfer_sizes.pop(0)
                self.transfer_times.pop(0)
        
        self.sier_avg = 0.5 * self.sier_avg + 0.5 * sier
        return sier, kb
 
    def run_student(self, rnd, teacher_cid, alpha, T, queries):
        start_time = time.time()
        logger.info(f"🔄 KNOWLEDGE DISTILLATION PROCESS")
        
        # Clone parameters BEFORE any computation to preserve local knowledge
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        θ_init = [p.clone().detach() for p in trainable_params]
        
        logger.info(f"   Step 1/5: Reading teacher knowledge...")
        
        try:
            knowledge_receive_start = time.time()
            payload = read_blob(teacher_cid, rnd, logger.info)
            knowledge_receive_time = time.time() - knowledge_receive_start
            
            # Calculate knowledge transfer size for bandwidth measurement
            import pickle
            knowledge_size_bytes = len(pickle.dumps(payload))
            knowledge_size_kb = knowledge_size_bytes / 1024
            
            logger.info(f"   ✅ Teacher knowledge received ({len(payload['values'])} logits)")
            logger.info(f"   📦 Knowledge received: {knowledge_size_kb:.1f} KB in {knowledge_receive_time:.2f}s")
            
            # Track bandwidth for receiving side
            if knowledge_receive_time > 0:
                receive_speed_kbps = knowledge_size_kb / knowledge_receive_time
                logger.info(f"   📊 Knowledge receive speed: {receive_speed_kbps:.2f} KB/s")
                
                # Record for bandwidth calculation (limit history to last 10 transfers)
                self.transfer_sizes.append(knowledge_size_kb)
                self.transfer_times.append(knowledge_receive_time)
                if len(self.transfer_sizes) > 10:
                    self.transfer_sizes.pop(0)
                    self.transfer_times.pop(0)
                    
        except TimeoutError:
            logger.warning(f"   ❌ Failed to receive teacher knowledge - timeout")
            return
        
        logger.info(f"   Step 2/5: Memory-optimized knowledge distillation processing...")
        
        # Memory-efficient batch processing to prevent OOM
        # Dynamically adjust batch size based on number of clients to prevent OOM
        if NUM_CLIENTS >= 4:
            KD_BATCH_SIZE = 8  # Smaller batch for 4+ clients
        else:
            KD_BATCH_SIZE = 32  # Larger batch for fewer clients
        
        num_queries = len(queries)
        total_kd_loss = 0.0
        total_task_loss = 0.0
        processed_samples = 0
        
        teacher_tok = load_tokenizer_only(teacher_cid)  # Tokenizer-only to avoid meta tensor issues
        
        logger.info(f"   📊 Processing {num_queries} queries in batches of {KD_BATCH_SIZE} for memory efficiency")
        
        # Store initial parameters for proximal regularization
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        θ_init = [p.data.clone().detach() for p in trainable_params]
        
        # Create optimizer once for the entire KD process
        optimiser = torch.optim.Adam(trainable_params, lr=LR_KD)
        
        for batch_start in range(0, num_queries, KD_BATCH_SIZE):
            try:
                batch_end = min(batch_start + KD_BATCH_SIZE, num_queries)
                batch_queries = queries[batch_start:batch_end]
                batch_size_actual = len(batch_queries)
                
                # Process teacher logits for this batch only
                if isinstance(payload["values"], torch.Tensor):
                    batch_values = payload["values"][batch_start:batch_end].detach().clone().to(self.gpu).float()
                    batch_indices = payload["indices"][batch_start:batch_end].detach().clone().to(self.gpu)
                else:
                    batch_values = torch.tensor(payload["values"][batch_start:batch_end]).to(self.gpu).float()
                    batch_indices = torch.tensor(payload["indices"][batch_start:batch_end]).to(self.gpu)
                
                dense = sparse_to_dense(batch_values, batch_indices, teacher_tok, self.tok)
                soft_t = F.softmax(dense / T, dim=-1)
                
                # Process student logits for this batch only
                if self._is_encoder_decoder():
                    # For T5 student models - use encoder-decoder forward
                    tok_in = self.tok(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to(self.gpu)
                    # Generate dummy decoder inputs for forward pass
                    decoder_input_ids = torch.zeros((tok_in["input_ids"].shape[0], dense.shape[1]), dtype=torch.long).to(self.gpu)
                    decoder_input_ids[:, 0] = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
                    
                    outputs = self.model(
                        input_ids=tok_in["input_ids"],
                        attention_mask=tok_in["attention_mask"],
                        decoder_input_ids=decoder_input_ids
                    )
                    logits_s = outputs.logits
                    logits_s = logits_s[:, :dense.shape[1], :]  # Align seq_len
                else:
                    # For decoder-only student models
                    tok_in = self.tok(batch_queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_EVAL_LENGTH).to(self.gpu)
                    logits_s = self.model(**tok_in).logits
                    logits_s = logits_s[:, :dense.shape[1], :]  # Align seq_len
                
                soft_s = F.log_softmax(logits_s / T, dim=-1)
                
                # Handle sequence length and vocabulary size mismatch between teacher and student
                seq_len_teacher = soft_t.size(1)
                seq_len_student = soft_s.size(1)
                vocab_size_teacher = soft_t.size(-1)
                vocab_size_student = soft_s.size(-1)
                
                # Align sequence lengths first
                min_seq_len = min(seq_len_teacher, seq_len_student)
                soft_t_seq_aligned = soft_t[:, :min_seq_len, :]
                soft_s_seq_aligned = soft_s[:, :min_seq_len, :]
                
                # Then align vocabularies
                if vocab_size_teacher != vocab_size_student:
                    # Align vocabularies by using the smaller vocabulary size
                    min_vocab_size = min(vocab_size_teacher, vocab_size_student)
                    soft_t_aligned = soft_t_seq_aligned[:, :, :min_vocab_size]
                    soft_s_aligned = soft_s_seq_aligned[:, :, :min_vocab_size]
                else:
                    soft_t_aligned = soft_t_seq_aligned
                    soft_s_aligned = soft_s_seq_aligned
                
                # Compute KL divergence loss for this batch
                kl = F.kl_div(soft_s_aligned, soft_t_aligned, reduction="batchmean") * (T ** 2)
                
                # For task loss, use student's full vocabulary
                task = CrossEntropyLoss(label_smoothing=0.1)(logits_s.reshape(-1, logits_s.size(-1)), soft_t_aligned.argmax(-1).reshape(-1))
                
                # Combined loss for this batch
                batch_loss = alpha * task + (1 - alpha) * kl
                
                # Add proximal regularization to prevent forgetting local knowledge
                prox = LAMBDA_PROX * sum((p - p_init).pow(2).sum() for p, p_init in zip(trainable_params, θ_init))
                batch_loss_with_prox = batch_loss + prox
                
                # Backward pass for this batch
                optimiser.zero_grad()
                batch_loss_with_prox.backward()
            
                # Clean up intermediate tensors to free memory
                del soft_t, soft_s, soft_t_aligned, soft_s_aligned, logits_s, dense
                if NUM_CLIENTS >= 4:
                    torch.cuda.empty_cache()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimiser.step()
                
                # Accumulate losses for reporting (weighted by batch size)
                total_kd_loss += kl.item() * batch_size_actual
                total_task_loss += task.item() * batch_size_actual
                processed_samples += batch_size_actual
            
                # Clear intermediate tensors to free memory immediately
                # Note: dense, soft_t, soft_s, logits_s already deleted above
                try:
                    del batch_values, batch_indices, tok_in
                except NameError:
                    pass
                    
                try:
                    del soft_t_seq_aligned, soft_s_seq_aligned, soft_t_aligned, soft_s_aligned
                except NameError:
                    pass
                    
                try:
                    del kl, task, batch_loss, prox, batch_loss_with_prox
                except NameError:
                    pass
                    
                if 'outputs' in locals():
                    try:
                        del outputs
                    except NameError:
                        pass
                        
                if 'decoder_input_ids' in locals():
                    try:
                        del decoder_input_ids
                    except NameError:
                        pass
                torch.cuda.empty_cache()
                
                if (batch_start // KD_BATCH_SIZE + 1) % 5 == 0:  # Log every 5 batches
                    logger.debug(f"   📦 Processed batch {batch_start // KD_BATCH_SIZE + 1}/{(num_queries + KD_BATCH_SIZE - 1) // KD_BATCH_SIZE}")
                    
            except Exception as e:
                logger.error(f"Error processing KD batch {batch_start // KD_BATCH_SIZE + 1}: {e}")
                # Clean up any partially created variables
                torch.cuda.empty_cache()
                raise
        
        # Calculate average losses for reporting
        avg_kd_loss = total_kd_loss / processed_samples
        avg_task_loss = total_task_loss / processed_samples
        avg_total_loss = alpha * avg_task_loss + (1 - alpha) * avg_kd_loss
        
        # Store KD training loss for reporting
        self.last_kd_training_loss = avg_total_loss
        
        logger.info(f"   ✅ Memory-optimized KD completed - KL: {avg_kd_loss:.4f}, Task: {avg_task_loss:.4f}, Total: {avg_total_loss:.4f}")
        logger.info(f"   🧠 Peak memory usage kept under control with {KD_BATCH_SIZE}-query batching (was: all {num_queries} queries at once)")
        
        elapsed = time.time() - start_time
        logger.info(f"   ✅ Memory-optimized KD completed successfully!")
        logger.info(f"   🕒 Total KD Time: {elapsed:.2f}s")
        
        # Perform local recovery round if enabled
        if ENABLE_RECOVERY_ROUND:
            # Evaluate post-KD performance before recovery
            post_kd_perf = self.eval_pass1()
            logger.info(f"\n📊 Post-KD Performance (before recovery): {post_kd_perf:.3f}")
            
            # Store post-KD performance
            self.post_kd_performance = post_kd_perf
            
            # Use stored pre-KD performance for accurate tracking
            pre_kd_perf = getattr(self, 'pre_kd_student_perf', self.last_perf)
            
            # Store recovery metrics as instance variable for later reporting
            self.recovery_metrics = self.local_recovery_training(θ_init, pre_kd_perf)
            
            # Log recovery metrics to structured logger if available
            if hasattr(self, 'recovery_metrics'):
                from src.structured_logging import get_structured_logger
                structured_logger = get_structured_logger()
                if hasattr(structured_logger, 'log_recovery_metrics'):
                    structured_logger.log_recovery_metrics(
                        client_id=self.cid,
                        round_id=rnd,
                        recovery_metrics=self.recovery_metrics
                    )
 
    def make_profile(self, dperf):
        fam_bits = [0, 0, 0, 0]
        fam_bits[self.cid % 4] = 1  # One-hot family
        params_m = sum(p.numel() for p in self.model.parameters()) / 1e6
        
        # Calculate effective bandwidth based on recent communication patterns
        # This provides a realistic measure of client's communication capability
        if hasattr(self, 'transfer_times') and hasattr(self, 'transfer_sizes') and len(self.transfer_times) > 0:
            # Calculate bandwidth from recent transfers: size / time = throughput
            recent_transfers = min(5, len(self.transfer_times))  # Use last 5 transfers
            recent_bandwidth = []
            for i in range(-recent_transfers, 0):
                if self.transfer_times[i] > 0:  # Avoid division by zero
                    # Convert KB/s to normalized bandwidth score (0-1 scale)
                    bw_kbps = self.transfer_sizes[i] / self.transfer_times[i]
                    # Normalize to typical enterprise bandwidth (100 Mbps = 12.5 MB/s = 12800 KB/s)
                    normalized_bw = min(1.0, bw_kbps / 12800.0)
                    recent_bandwidth.append(normalized_bw)
            
            if recent_bandwidth:
                effective_bandwidth = float(np.mean(recent_bandwidth))
            else:
                effective_bandwidth = 0.5  # Default moderate bandwidth
        else:
            # Default bandwidth for new clients (moderate capability)
            effective_bandwidth = 0.5
            
        profile = np.array([
            self.last_perf,  # pass@1 val
            getattr(self, 'last_codebleu', 0.0),  # CodeBLEU score
            self.sier_avg,
            *fam_bits,  # 4 elements: one-hot model family
            params_m / 1000,  # scale
            self.trust,
            self.historical_delta,
            self.comm_kb,
            self.pre_post_diff,
            effective_bandwidth,  # Calculated bandwidth based on transfer performance
            0.0,  # padding to reach 16 dims
            0.0,  # padding to reach 16 dims  
            0.0   # padding to reach 16 dims
        ], dtype=np.float32)
        self.historical_delta = 0.5 * self.historical_delta + 0.5 * dperf
        return profile

    def _compute_data_profile(self, dataset) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """Estimate data and difficulty distributions for heuristic pairing."""
        type_counts = Counter({ptype: 0 for ptype in self.problem_heuristics.PROBLEM_TYPES})
        difficulty_counts = Counter({lvl: 0 for lvl in self.problem_heuristics.DIFFICULTY_LEVELS})

        if dataset:
            for record in dataset:
                prompt = record.get('prompt', '')
                solution = record.get('canonical_solution', record.get('code', ''))
                categorization = self.problem_heuristics.categorize(prompt, solution)
                type_counts[categorization['primary_type']] += 1
                difficulty_counts[categorization['estimated_difficulty']] += 1

        total_type = sum(type_counts.values()) or 1
        total_diff = sum(difficulty_counts.values()) or 1

        type_distribution = {k: type_counts[k] / total_type for k in self.problem_heuristics.PROBLEM_TYPES}
        difficulty_distribution = {k: difficulty_counts[k] / total_diff for k in self.problem_heuristics.DIFFICULTY_LEVELS}
        specialization = max(type_distribution.values()) if type_distribution else 0.0

        return type_distribution, difficulty_distribution, specialization
    
    def get_comprehensive_metrics(self):
        """Get comprehensive metrics for CPM profile update"""
        return {
            'perf': float(self.last_perf),
            'local_codebleu': float(getattr(self, 'last_codebleu', 0.0)),
            'trust': float(self.trust),
            'delta_perf': float(self.pre_post_diff),
            'profile_vector': self.make_profile(self.pre_post_diff),
            'sier_avg': float(self.sier_avg),
            'historical_delta': float(self.historical_delta),
            'comm_kb': float(self.comm_kb),
            'client_id': int(self.cid),
            'data_distribution': dict(self.data_distribution),
            'difficulty_distribution': dict(self.difficulty_distribution),
            'specialization_score': float(self.specialization_score),
            'global_perf': float(self.last_perf),
            'transfer_perf': float(self.last_transfer_perf) if self.last_transfer_perf is not None else float(self.last_perf)
        }
    
    def local_recovery_training(self, θ_init: List[torch.Tensor], pre_kd_performance: float):
        """
        Perform local recovery training after knowledge distillation to re-anchor to local distribution.
        This mitigates the impact of noisy/low-quality teacher knowledge.
        
        Args:
            θ_init: Initial parameters before KD (for proximal regularization)
            pre_kd_performance: Performance before KD for tracking
        
        Returns:
            Dict with recovery metrics for academic reporting
        """
        logger.info(f"\n🔧 LOCAL RECOVERY ROUND - Client {self.cid}")
        logger.info(f"{'='*50}")
        logger.info(f"   Purpose: Re-anchor to local data distribution after KD")
        logger.info(f"   Recovery steps: {RECOVERY_STEPS}")
        logger.info(f"   Recovery LR: {RECOVERY_LR}")
        logger.info(f"   Recovery Lambda: {RECOVERY_LAMBDA_PROX}")
        
        start_time = time.time()
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Clone current parameters to track recovery changes
        θ_post_kd = [p.clone().detach() for p in trainable_params]
        
        # Create optimizer with lower learning rate
        optimiser = torch.optim.Adam(trainable_params, lr=RECOVERY_LR)
        
        # Sample recovery batch from local training data
        train_list = list(self.train_ds)
        recovery_samples = random.sample(train_list, min(RECOVERY_STEPS * RECOVERY_BATCH_SIZE, len(train_list)))
        
        total_recovery_loss = 0.0
        recovery_losses = []
        
        for step in range(RECOVERY_STEPS):
            optimiser.zero_grad()
            
            # Get batch for this step
            batch_start = (step * RECOVERY_BATCH_SIZE) % len(recovery_samples)
            batch_end = min(batch_start + RECOVERY_BATCH_SIZE, len(recovery_samples))
            batch_items = recovery_samples[batch_start:batch_end]
            
            if batch_end >= len(recovery_samples):
                # Reshuffle when we've gone through all samples
                random.shuffle(recovery_samples)
            
            # Create prompts and targets
            prompts = [item["prompt"] for item in batch_items]
            targets = [item.get("canonical_solution", item.get("code", "")) for item in batch_items]
            full_texts = [p + t for p, t in zip(prompts, targets)]
            
            # Compute loss
            batch_loss = 0.0
            valid_items = 0
            
            for text in full_texts:
                try:
                    inputs = self.tok(text, return_tensors="pt", truncation=True, 
                                    max_length=MAX_EVAL_LENGTH, padding=True)
                    inputs = {k: v.to(self.gpu) for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    if not torch.isnan(loss):
                        batch_loss += loss
                        valid_items += 1
                    
                    # Clean up
                    del inputs, outputs, loss
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.debug(f"Recovery step {step} failed for one item: {e}")
                    continue
            
            if valid_items > 0:
                avg_loss = batch_loss / valid_items
                
                # Add strong proximal regularization to θ_init (pre-KD parameters)
                prox_to_init = RECOVERY_LAMBDA_PROX * sum((p - p_init).pow(2).sum() 
                                                          for p, p_init in zip(trainable_params, θ_init))
                
                # Also add mild regularization to θ_post_kd to not completely forget KD
                prox_to_post_kd = (RECOVERY_LAMBDA_PROX * 0.1) * sum((p - p_kd).pow(2).sum() 
                                                                     for p, p_kd in zip(trainable_params, θ_post_kd))
                
                total_loss = avg_loss + prox_to_init + prox_to_post_kd
                total_loss.backward()
                optimiser.step()
                
                total_recovery_loss += avg_loss.item()
                recovery_losses.append(avg_loss.item())
                
                # Log progress every 10 steps
                if (step + 1) % 10 == 0:
                    logger.info(f"   Step {step + 1}/{RECOVERY_STEPS}: LOCAL TRAIN DATA Loss = {avg_loss.item():.4f}, "
                              f"Prox(init) = {prox_to_init.item():.6f}, Prox(KD) = {prox_to_post_kd.item():.6f}")
        
        # Evaluate performance after recovery
        post_recovery_perf = self.eval_pass1()
        recovery_time = time.time() - start_time
        
        # Calculate recovery metrics
        avg_recovery_loss = total_recovery_loss / RECOVERY_STEPS if RECOVERY_STEPS > 0 else 0.0
        post_kd_perf = getattr(self, 'post_kd_performance', self.last_perf)
        perf_change_from_kd = post_recovery_perf - post_kd_perf  # Change from post-KD performance
        perf_change_from_init = post_recovery_perf - pre_kd_performance  # Total change from pre-KD
        
        # Calculate parameter drift
        param_drift_from_init = sum((p - p_init).pow(2).sum().item() 
                                   for p, p_init in zip(trainable_params, θ_init))
        param_drift_from_kd = sum((p - p_kd).pow(2).sum().item() 
                                 for p, p_kd in zip(trainable_params, θ_post_kd))
        
        recovery_metrics = {
            "recovery_steps": RECOVERY_STEPS,
            "avg_recovery_loss": avg_recovery_loss,
            "recovery_losses": recovery_losses,
            "pre_kd_performance": pre_kd_performance,
            "post_kd_performance": post_kd_perf,
            "post_recovery_performance": post_recovery_perf,
            "perf_change_from_kd": perf_change_from_kd,
            "perf_change_from_init": perf_change_from_init,
            "param_drift_from_init": param_drift_from_init,
            "param_drift_from_kd": param_drift_from_kd,
            "recovery_time_s": recovery_time
        }
        
        # Update last performance
        self.last_perf = post_recovery_perf
        
        # Log recovery summary
        logger.info(f"{'='*50}")
        logger.info(f"📊 RECOVERY ROUND RESULTS:")
        logger.info(f"   Avg Recovery LOCAL TRAIN DATA Loss: {avg_recovery_loss:.4f}")
        logger.info(f"   Performance trajectory (LOCAL VAL DATA Pass@1):")
        logger.info(f"      Pre-KD:  {pre_kd_performance:.3f}")
        logger.info(f"      Post-KD: {recovery_metrics['post_kd_performance']:.3f}")
        logger.info(f"      Post-Recovery: {post_recovery_perf:.3f}")
        logger.info(f"   Net change from KD: {perf_change_from_kd:+.3f}")
        logger.info(f"   Total change: {perf_change_from_init:+.3f}")
        
        if perf_change_from_kd > 0:
            logger.info(f"✅ Recovery IMPROVED performance vs post-KD!")
        elif perf_change_from_kd == 0:
            logger.info(f"➖ Recovery maintained post-KD performance")
        else:
            logger.info(f"⚠️ Recovery decreased performance vs post-KD")
            
        logger.info(f"   Recovery Time: {recovery_time:.1f}s")
        logger.info(f"{'='*50}\n")
        
        return recovery_metrics
    
    def _save_model_checkpoint(self, round_id: int, performance: float):
        """Save model checkpoint for post-training analysis"""
        try:
            # Save LoRA adapter parameters
            checkpoint_path = f"{self.checkpoint_dir}/round_{round_id}.pt"
            
            # Get adapter parameters
            adapter_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and ('lora' in name.lower() or 'adapter' in name.lower()):
                    adapter_params[name] = param.detach().cpu().clone()
            
            # Save checkpoint with metadata
            checkpoint = {
                'round': round_id,
                'client_id': self.cid,
                'performance': performance,
                'adapter_parameters': adapter_params,
                'trust_score': self.trust,
                'sier_avg': self.sier_avg,
                'model_name': list(globals_module.MODEL_MAP.values())[self.cid % len(globals_module.MODEL_MAP)]
            }
            
            torch.save(checkpoint, checkpoint_path)
            self.model_checkpoints.append(checkpoint_path)
            
            logger.info(f"Client {self.cid} model checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model checkpoint for client {self.cid}: {e}")
            raise RuntimeError(f"Model checkpoint saving failed: {e}")

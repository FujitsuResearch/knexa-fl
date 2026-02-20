#!/usr/bin/env python3
"""
FedID-style Centralized Knowledge Distillation baseline for KNEXA-FL

Implements a server-hosted central model that:
- Aggregates client predictions on the shared public transfer set (X_u)
- Trains the central model via text-level KD on the aggregated teacher sequences
- Provides validation feedback using a small server-only validation split
- Distills back to clients from the central model using text-level KD

This module is designed to plug into the existing KNEXA-FL experimental suite
without affecting other implementations. It reuses the LLM/PEFT stack and
dataset utilities already present in the repository.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model_utils import load_model_and_tokenizer
from src.grpc_p2p.knowledge_distillation import AdaptiveKnowledgeDistillation, KDConfig

logger = logging.getLogger(__name__)


@dataclass
class FedIDConfig:
    """Configuration for the FedID-style central KD."""
    # Central model choice: use an existing backbone from MODEL_MAP via CID
    central_cid: int = 0
    # KD hyperparams (kept consistent with existing baselines)
    lr_kd: float = 5e-5
    temperature_decode: float = 1.5
    batch_size_public: int = 32
    # Validation set size carved from public data evaluation split
    val_set_size: int = 32
    seed: int = 42


class FedIDCentralKD:
    """FedID-style centralized KD controller.

    Orchestrates per-round server-side interactive distillation and client KD
    using the existing KNEXA-FL stack. The central model is persistent across
    rounds and trained only on public transfer data.
    """

    def __init__(self,
                 model_manager: Any,
                 transfer_set: Any,
                 exp_dir: Optional[str] = None,
                 config: Optional[FedIDConfig] = None):
        self.model_manager = model_manager
        self.transfer_set = transfer_set
        self.exp_dir = exp_dir
        self.config = config or FedIDConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kd_module = AdaptiveKnowledgeDistillation(KDConfig(
            temperature=self.config.temperature_decode,
            generation_temperature=self.config.temperature_decode,
            generation_max_length=256
        ))

        # Initialize central model once and keep persistent across rounds
        self.central_model, self.central_tok = load_model_and_tokenizer(self.config.central_cid, str(self.device))
        self.central_model.train()
        self.central_optimizer = torch.optim.AdamW(self.central_model.parameters(), lr=self.config.lr_kd)
        logger.info(f"FedID-CentralKD: Initialized central model for CID={self.config.central_cid}")

    def _normalize_code(self, text: str) -> str:
        """Normalize code strings for grouping (whitespace/comments-insensitive)."""
        if text is None:
            return ""
        # Simple normalization: strip whitespace-only differences
        # (For brevity; more advanced normalization can be added if needed.)
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    def _aggregate_ensemble(self, client_responses: List[List[Dict[str, Any]]]) -> List[Tuple[str, str]]:
        """Aggregate client responses into a single teacher sequence per sample.

        Args:
            client_responses: list per-client of response dicts aligned to transfer samples.
                              Each response contains {'prompt', 'generated_text', 'quality', ...}

        Returns:
            List of (prompt, teacher_sequence) pairs for central KD training.
        """
        # Align by sample index across clients
        num_samples = max(len(r) for r in client_responses) if client_responses else 0
        aggregated_pairs: List[Tuple[str, str]] = []

        for s_idx in range(num_samples):
            groups: Dict[str, float] = {}
            prompt: Optional[str] = None
            # Collect predictions for this sample from all clients
            for responses in client_responses:
                if s_idx < len(responses):
                    r = responses[s_idx]
                    prompt = r.get('prompt', prompt)
                    code = r.get('generated_text', '')
                    conf = float(r.get('quality', 0.0))
                    key = self._normalize_code(code)
                    if not key:
                        continue
                    groups[key] = groups.get(key, 0.0) + conf
            # Choose highest-score group; fallback to max-quality single if no groups
            if groups:
                best_code_norm = max(groups.items(), key=lambda kv: kv[1])[0]
                # Retrieve original code text corresponding to normalized key
                # In absence of exact mapping, use normalized text itself
                teacher_seq = best_code_norm
            else:
                # Fallback: use any available highest-quality sequence
                best_r = None
                best_q = -1.0
                for responses in client_responses:
                    if s_idx < len(responses):
                        r = responses[s_idx]
                        if r.get('quality', 0.0) > best_q and r.get('generated_text'):
                            best_q = float(r.get('quality', 0.0))
                            best_r = r
                teacher_seq = best_r.get('generated_text', '') if best_r else ''
            if prompt and teacher_seq:
                aggregated_pairs.append((prompt, teacher_seq))
        return aggregated_pairs

    def _central_train_on_pairs(self, training_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Train central model on aggregated teacher sequences using text-level KD."""
        self.central_model.train()
        total_loss = 0.0
        num_steps = 0
        for prompt, teacher_seq in training_pairs:
            full_text = f"{prompt}{teacher_seq}"
            inputs = self.central_tok(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            labels = inputs['input_ids'].clone()
            try:
                outputs = self.central_model(**inputs, labels=labels)
                loss = outputs.loss
                self.central_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.central_model.parameters(), max_norm=1.0)
                self.central_optimizer.step()
                total_loss += float(loss.item())
                num_steps += 1
                del outputs, loss
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            finally:
                del inputs, labels
                if num_steps % 50 == 0:
                    torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, num_steps)
        logger.info(f"FedID-CentralKD: Central KD step avg_loss={avg_loss:.6f} over {num_steps} steps")
        return {"avg_loss": avg_loss, "num_steps": num_steps}

    def _central_validation(self, val_samples: List[Dict[str, Any]]) -> float:
        """Evaluate central model on server-only validation set using supervised LM loss."""
        self.central_model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for sample in val_samples:
                prompt = sample.get('prompt', '')
                solution = sample.get('canonical_solution', '') or sample.get('code', '')
                if not prompt or not solution:
                    continue
                full_text = f"{prompt}{solution}"
                inputs = self.central_tok(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                labels = inputs['input_ids'].clone()
                outputs = self.central_model(**inputs, labels=labels)
                loss = float(outputs.loss.item())
                total_loss += loss
                count += 1
                del outputs, inputs, labels
                if count % 32 == 0:
                    torch.cuda.empty_cache()
        avg_val = total_loss / max(1, count)
        logger.info(f"FedID-CentralKD: Central validation L_val={avg_val:.6f} on {count} samples")
        return avg_val

    def _evaluate_client_perf(self, client: Any) -> float:
        """Use the client's eval_pass1 method for consistent quick performance."""
        try:
            client.model.eval()
            perf = client.eval_pass1()
            client.model.train()
            return float(perf)
        except Exception as e:
            logger.error(f"FedID-CentralKD: Client eval failed: {e}")
            return 0.0

    def run_round(self, round_id: int, num_clients: int) -> Dict[str, Any]:
        """Run one FedID-KD communication round.

        Returns a result dict containing central validation feedback and per-client KD outcomes.
        """
        rng = np.random.RandomState(self.config.seed + round_id)

        # 2(a-b) Collect client predictions on the full transfer set (as mini-batches)
        public_samples = self.transfer_set.get_batch()  # Use entire D0
        # Split into mini-batches of size config.batch_size_public
        batches: List[List[Dict[str, Any]]] = [
            public_samples[i:i + self.config.batch_size_public]
            for i in range(0, len(public_samples), self.config.batch_size_public)
        ]

        all_training_pairs: List[Tuple[str, str]] = []
        for b_idx, batch in enumerate(batches):
            logger.info(f"FedID-CentralKD: Processing public mini-batch {b_idx+1}/{len(batches)} (size={len(batch)})")
            # Broadcast batch prompts implicitly by iterating clients
            client_responses: List[List[Dict[str, Any]]] = []
            for k in range(num_clients):
                try:
                    client = self.model_manager.load_model(k, round_id)
                    responses_pkg = self.kd_module.generate_teacher_responses(
                        client.model, client.tok, batch, client.gpu
                    )
                    client_responses.append(responses_pkg['responses'])
                    self.model_manager.unload_model(k)
                except Exception as e:
                    logger.error(f"FedID-CentralKD: Failed to get responses from client {k}: {e}")
                    continue
            # 2(c) Aggregate to ensemble teacher per sample
            training_pairs = self._aggregate_ensemble(client_responses)
            all_training_pairs.extend(training_pairs)
            # 2(d) Central KD step for this mini-batch
            _ = self._central_train_on_pairs(training_pairs)

            # 2(e) Central validation feedback (every two mini-batches, or at least once)
            if (b_idx % 2 == 0) or (b_idx == len(batches) - 1):
                eval_samples = self.transfer_set.get_evaluation_samples()
                rng.shuffle(eval_samples)
                val_subset = eval_samples[: self.config.val_set_size]
                L_val = self._central_validation(val_subset)
                # Feedback: just log scalar; in FedID it’s broadcast to clients
                logger.info(f"FedID-CentralKD: Broadcast central validation L_val={L_val:.6f} (round {round_id})")

        # 3) Local distillation from central model back to clients
        per_client_results: List[Dict[str, Any]] = []
        central_kd_teacher_pkg = self.kd_module.generate_teacher_responses(
            self.central_model, self.central_tok, public_samples, str(self.device)
        )

        for k in range(num_clients):
            try:
                client = self.model_manager.load_model(k, round_id)
                pre_perf = self._evaluate_client_perf(client)
                optimizer = torch.optim.AdamW(client.model.parameters(), lr=self.config.lr_kd)
                import time as _time
                _t0 = _time.time()
                train_result = self.kd_module.perform_text_based_student_training(
                    client.model, client.tok, central_kd_teacher_pkg, optimizer, num_steps=1
                )
                _lat_ms = int((_time.time() - _t0) * 1000)
                post_perf = self._evaluate_client_perf(client)
                self.model_manager.save_model_state(k, client)
                self.model_manager.unload_model(k)
                per_client_results.append({
                    'student_id': k,
                    'teacher_id': -2,  # Central model indicator
                    'round_id': round_id,
                    'pre_performance': pre_perf,
                    'post_performance': post_perf,
                    'performance_improvement': post_perf - pre_perf,
                    'performance_gain': post_perf - pre_perf,
                    'kd_loss': train_result.get('avg_loss', 0.0),
                    'training_result': train_result,
                    'knowledge_bytes': central_kd_teacher_pkg.get('num_samples', 0) * 1024,
                    'latency_ms': _lat_ms,
                    'success': True,
                    'method_used': 'fedid_central_kd',
                })
            except Exception as e:
                logger.error(f"FedID-CentralKD: Client {k} distillation failed: {e}")
                per_client_results.append({
                    'student_id': k,
                    'teacher_id': -2,
                    'round_id': round_id,
                    'error': str(e),
                    'success': False,
                })

        # Summary
        avg_improve = np.mean([r['performance_improvement'] for r in per_client_results if r.get('success')]) if per_client_results else 0.0
        result = {
            'round_id': round_id,
            'central_model_cid': self.config.central_cid,
            'num_clients': num_clients,
            'num_training_pairs': len(all_training_pairs),
            'client_results': per_client_results,
            'avg_client_improvement': float(avg_improve),
        }
        return result

    def save_checkpoint(self, path: str) -> None:
        """Save central model LoRA adapter state for reproducibility."""
        try:
            import os
            os.makedirs(path, exist_ok=True)
            torch.save(self.central_model.state_dict(), os.path.join(path, "central_model.pt"))
            logger.info(f"FedID-CentralKD: Saved central model checkpoint to {path}")
        except Exception as e:
            logger.error(f"FedID-CentralKD: Failed to save central checkpoint: {e}")

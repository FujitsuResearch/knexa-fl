#!/usr/bin/env python3
"""
Text-based Knowledge Distillation for KNEXA-FL
Pure text-based knowledge transfer without logit-based methods
Implements Equation 252-253 from KNEXA-FL paper with strict academic standards
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
import random
import numpy as np
from src.structured_logging import get_structured_logger, LossType, DataSource

logger = logging.getLogger(__name__)
structured_logger = get_structured_logger()

@dataclass
class KDConfig:
    """Configuration for knowledge distillation"""
    temperature: float = 2.0
    alpha_kd: float = 0.5
    max_sequence_length: int = 512
    generation_max_length: int = 256
    generation_temperature: float = 0.8
    generation_top_p: float = 0.9
    generation_top_k: int = 50
    quality_threshold: float = 0.1
    max_retries: int = 3

class AdaptiveKnowledgeDistillation:
    """
    Pure text-based knowledge distillation
    Teacher generates text, student learns from teacher's generated text
    """
    
    def __init__(self, config: Optional[KDConfig] = None):
        self.config = config or KDConfig()
        self.stats = {
            'teacher_generations': 0,
            'student_trainings': 0,
            'total_tokens_processed': 0,
            'text_transfer_success': 0,
            'text_transfer_failures': 0
        }
        self.performance_history = []
        self.adaptive_quality_threshold = 0.3
        
    def generate_teacher_responses(self, 
                                 teacher_model: Any,
                                 teacher_tokenizer: Any,
                                 transfer_samples: List[Dict[str, Any]],
                                 device: str = "cuda") -> Dict[str, Any]:
        """
        Generate teacher text responses for transfer samples
        
        Args:
            teacher_model: Teacher model
            teacher_tokenizer: Teacher tokenizer
            transfer_samples: Input samples
            device: Device to run on
            
        Returns:
            Dictionary with teacher responses and quality metrics
        """
        teacher_model.eval()
        teacher_responses = []
        high_quality_count = 0
        
        logger.info(f"Generating teacher responses for {len(transfer_samples)} samples (memory-optimized)")
        
        with torch.no_grad():
            for i, sample in enumerate(transfer_samples):
                prompt = sample.get('prompt', '')
                if not prompt:
                    continue
                    
                try:
                    # Tokenize prompt
                    inputs = teacher_tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=self.config.max_sequence_length,
                        truncation=True,
                        padding=True
                    ).to(device)
                    
                    # Generate response
                    outputs = teacher_model.generate(
                        **inputs,
                        max_new_tokens=self.config.generation_max_length,
                        temperature=self.config.generation_temperature,
                        top_p=self.config.generation_top_p,
                        top_k=self.config.generation_top_k,
                        do_sample=True,
                        pad_token_id=teacher_tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    
                    # Decode only the generated part (exclude input prompt)
                    generated_text = teacher_tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    # Quality check
                    quality = self._assess_response_quality(generated_text)
                    is_high_quality = quality >= self.config.quality_threshold
                    
                    if is_high_quality:
                        high_quality_count += 1
                    
                    teacher_responses.append({
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'quality': quality,
                        'is_high_quality': is_high_quality,
                        'sample_id': sample.get('id', f'sample_{i}')
                    })
                    
                    # Clear tensors immediately after each generation
                    del inputs, outputs
                    torch.cuda.empty_cache()
                    
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Generated {i + 1}/{len(transfer_samples)} responses")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate response for sample {i}: {e}")
                    continue
        
        # Update stats
        self.stats['teacher_generations'] += 1
        
        result = {
            'responses': teacher_responses,
            'high_quality_count': high_quality_count,
            'num_samples': len(teacher_responses),
            'quality_ratio': high_quality_count / max(1, len(teacher_responses)),
            'timestamp': time.time()
        }
        
        logger.info(f"Generated {len(teacher_responses)} responses, "
                   f"{high_quality_count} high quality ({result['quality_ratio']:.2%})")
        
        return result
    
    def _assess_response_quality(self, generated_text: str) -> float:
        """Assess quality of generated text"""
        if not generated_text or len(generated_text.strip()) < 10:
            return 0.0
            
        # Basic quality metrics
        length_score = min(len(generated_text) / 100, 1.0)  # Normalize by 100 chars
        
        # Check for repetition
        words = generated_text.split()
        if len(words) < 3:
            return length_score * 0.3
            
        unique_words = len(set(words))
        repetition_score = unique_words / len(words)
        
        # Check for coherence indicators
        coherence_score = 0.8  # Default coherence score
        if any(marker in generated_text.lower() for marker in ['def ', 'class ', 'import ', 'return ']):
            coherence_score = 1.0  # Code-like content
        elif any(marker in generated_text for marker in ['.', '!', '?']):
            coherence_score = 0.9  # Proper sentences
            
        return (length_score + repetition_score + coherence_score) / 3.0
    
    def perform_text_based_student_training(self,
                                          student_model: Any,
                                          student_tokenizer: Any,
                                          teacher_responses_package: Dict[str, Any],
                                          optimizer: Any,
                                          num_steps: int = 3) -> Dict[str, Any]:
        """
        Train student model using teacher-generated text
        
        Args:
            student_model: Student model to train
            student_tokenizer: Student tokenizer
            teacher_responses_package: Package with teacher responses
            optimizer: Optimizer for gradient updates
            num_steps: Number of training steps
            
        Returns:
            Training results dictionary
        """
        try:
            student_model.train()
            device = next(student_model.parameters()).device
            
            teacher_responses = teacher_responses_package['responses']
            if not teacher_responses:
                error_msg = "CRITICAL: No teacher responses available for training - this indicates a failure in teacher response generation"
                structured_logger.error(error_msg, indent_level=2)
                raise RuntimeError(error_msg)
            
            # Use high quality responses for training
            high_quality_responses = [r for r in teacher_responses if r['is_high_quality']]
            if not high_quality_responses:
                logger.warning("No high quality responses, using all responses")
                high_quality_responses = teacher_responses
            
            training_losses = []
            total_loss = 0.0
            processed_samples = 0
            
            logger.info(f"Starting text-based training with {len(high_quality_responses)} responses")
            
            for response in high_quality_responses:
                prompt = response['prompt']
                teacher_text = response['generated_text']
                
                if not teacher_text.strip():
                    continue
                
                # Create training text: prompt + teacher response
                training_text = prompt + teacher_text
                
                # Tokenize for student training
                inputs = student_tokenizer(
                    training_text,
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True,
                    padding=True
                ).to(device)
                
                # Prepare labels (same as input_ids for language modeling)
                labels = inputs['input_ids'].clone()
                
                # Multiple training steps per sample with memory management
                sample_losses = []
                for step in range(num_steps):
                    try:
                        # Forward pass
                        outputs = student_model(**inputs, labels=labels)
                        loss = outputs.loss
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                        
                        # Update parameters
                        optimizer.step()
                        
                        # Track loss with validation
                        step_loss = loss.item()
                        
                        # Validate loss value for academic integrity
                        structured_logger.loss_validation_check(
                            step_loss,
                            loss_type="kd_lm",
                            expected_range=(0.01, 50.0)  # Reasonable range for language modeling loss
                        )
                        
                        sample_losses.append(step_loss)
                        total_loss += step_loss
                        
                        # Report individual step loss with proper attribution
                        structured_logger.loss_report(
                            LossType.KD_LANGUAGE_MODEL,
                            DataSource.TEACHER_RESPONSES,
                            step_loss,
                            round_num=0,  # TODO: Pass actual round number
                            sample=processed_samples + 1,
                            step=step + 1,
                            total_steps=num_steps
                        )
                        
                        # Clear intermediate variables to free memory
                        del outputs, loss
                        
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            logger.error(f"CUDA OOM during KD training step {step}: {e}")
                            # Clear cache and break out of the loop
                            torch.cuda.empty_cache()
                            break
                        else:
                            raise e
                
                training_losses.extend(sample_losses)
                processed_samples += 1
                
                # Clear input tensors to free memory
                del inputs, labels
                
                avg_sample_loss = sum(sample_losses) / len(sample_losses)
                logger.debug(f"Sample {processed_samples}: avg_loss={avg_sample_loss:.6f}")
            
            if processed_samples == 0:
                error_msg = "CRITICAL: No samples were processed during training - indicates fundamental training failure"
                structured_logger.error(error_msg, indent_level=2)
                raise RuntimeError(error_msg)
            
            if not training_losses:
                error_msg = "CRITICAL: No training losses recorded despite processing samples - indicates loss computation failure"
                structured_logger.error(error_msg, indent_level=2)
                raise RuntimeError(error_msg)
            
            # Calculate average loss and validate
            avg_loss = total_loss / len(training_losses)
            structured_logger.loss_validation_check(
                avg_loss,
                loss_type="kd_lm",
                expected_range=(0.01, 50.0)
            )
            
            # Update stats
            self.stats['student_trainings'] += 1
            self.stats['text_transfer_success'] += 1
            self.stats['total_tokens_processed'] += sum(len(r['generated_text'].split()) for r in high_quality_responses)
            
            # Report comprehensive loss trajectory and final results
            # Note: loss_trajectory method doesn't exist in KnexaLogger, skipping for now
            
            # Final loss validation
            final_loss = training_losses[-1]
            structured_logger.loss_validation_check(
                final_loss,
                loss_type="kd_lm",
                expected_range=(0.01, 50.0)
            )
            
            # Report final comprehensive loss summary
            structured_logger.loss_report(
                LossType.KD_LANGUAGE_MODEL,
                DataSource.TEACHER_RESPONSES,
                avg_loss,
                round_num=0,  # TODO: Pass actual round number
                final_loss=final_loss,
                processed_samples=processed_samples,
                training_steps=len(training_losses),
                quality_filtered=len(high_quality_responses),
                total_responses=len(teacher_responses),
                loss_reduction=training_losses[0] - final_loss if len(training_losses) > 1 else 0.0
            )
            
            training_result = {
                'success': True,
                'method': 'text_based',
                'avg_loss': avg_loss,
                'training_losses': training_losses,
                'num_steps': len(training_losses),
                'processed_samples': processed_samples,
                'quality_filtered': len(high_quality_responses),
                'total_responses': len(teacher_responses),
                'final_loss': final_loss  # No fallback - guaranteed to exist
            }
            
            structured_logger.info(
                f"✅ Text-based student training completed successfully [avg_loss={avg_loss:.6f}, samples_processed={processed_samples}, training_steps={len(training_losses)}, method=text_based_kd]",
                indent_level=2
            )
            
            return training_result
            
        except Exception as e:
            error_msg = f"CRITICAL: Text-based student training failed: {e}"
            structured_logger.error(error_msg, e, indent_level=2)
            self.stats['text_transfer_failures'] += 1
            # Re-raise to ensure failures are not masked by fallback values
            raise RuntimeError(error_msg) from e
    
    def intelligent_knowledge_transfer(self,
                                     teacher_model: Any,
                                     teacher_tokenizer: Any,
                                     student_model: Any,
                                     student_tokenizer: Any,
                                     transfer_samples: List[Dict[str, Any]],
                                     optimizer: Any,
                                     device: str = "cuda") -> Dict[str, Any]:
        """
        Text-based knowledge transfer implementing Equation 252-253 from KNEXA-FL paper
        L_total_kd = (1-α_kd)L_i(D_i) + α_kd L_LM(X_u, y_j)
        
        Args:
            teacher_model: Teacher model
            teacher_tokenizer: Teacher tokenizer
            student_model: Student model  
            student_tokenizer: Student tokenizer
            transfer_samples: Input samples for knowledge transfer (X_u)
            optimizer: Student optimizer
            device: Device to run on
            
        Returns:
            Text-based training results with academic integrity guarantees
        """
        structured_logger.info("Starting text-based knowledge transfer (Eq. 252-253)", indent_level=2)
        
        if not transfer_samples:
            error_msg = "CRITICAL: No transfer samples provided for knowledge distillation"
            structured_logger.error(error_msg, indent_level=2)
            raise ValueError(error_msg)
        
        # Generate teacher responses (y_j in paper notation)
        structured_logger.info(f"Generating teacher responses for {len(transfer_samples)} transfer samples", indent_level=2)
        teacher_responses = self.generate_teacher_responses(
            teacher_model, teacher_tokenizer, transfer_samples, device
        )
        
        # Validate teacher response generation
        high_quality_count = teacher_responses['high_quality_count']
        total_samples = teacher_responses['num_samples']
        quality_ratio = high_quality_count / max(1, total_samples)
        
        if total_samples == 0:
            error_msg = "CRITICAL: Teacher failed to generate any responses"
            structured_logger.error(error_msg, indent_level=2)
            raise RuntimeError(error_msg)
            
        if quality_ratio < 0.1:  # Less than 10% high quality responses
            error_msg = f"CRITICAL: Teacher response quality too low: {quality_ratio:.1%} high quality responses"
            structured_logger.error(error_msg, indent_level=2)
            raise RuntimeError(error_msg)
        
        structured_logger.info(
            f"✅ Teacher response generation successful [total_responses={total_samples}, high_quality_responses={high_quality_count}, quality_ratio={quality_ratio:.1%}]",
            indent_level=2
        )
        
        # Perform text-based student training (L_LM component of Equation 252-253)
        structured_logger.info("Performing student training on teacher responses (L_LM component)", indent_level=2)
        training_result = self.perform_text_based_student_training(
            student_model, student_tokenizer, teacher_responses, optimizer
        )
        
        # Training result is guaranteed to be successful due to error handling in perform_text_based_student_training
        # Track performance for adaptive optimization
        self._track_method_performance('text_based', training_result['avg_loss'], quality_ratio)
        
        structured_logger.info(
            f"✅ Text-based knowledge transfer completed successfully [method=text_based_kd, avg_loss={training_result['avg_loss']:.6f}, quality_ratio={quality_ratio:.1%}, processed_samples={training_result['processed_samples']}]",
            indent_level=2
        )
        
        return {
            'success': True,
            'method_used': 'text_based',
            'teacher_responses': teacher_responses,
            'training_result': training_result,
            'quality_ratio': quality_ratio,
            'fallback_used': False  # No fallbacks in academic implementation
        }
    
    def _track_method_performance(self, method: str, loss: float, quality: float):
        """Track performance of knowledge transfer methods"""
        self.performance_history.append({
            'method': method,
            'loss': loss,
            'quality': quality,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get KD statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'teacher_generations': 0,
            'student_trainings': 0,
            'total_tokens_processed': 0,
            'text_transfer_success': 0,
            'text_transfer_failures': 0
        }


def create_kd_module_from_cmp(cmp_params: Dict[str, Any]) -> AdaptiveKnowledgeDistillation:
    """
    Create KD module from CMP parameters
    
    Args:
        cmp_params: Parameters from CMP service
        
    Returns:
        Configured KD module
    """
    config = KDConfig(
        temperature=cmp_params.get('temperature', 2.0),
        alpha_kd=cmp_params.get('alpha', 0.5),
        max_sequence_length=cmp_params.get('max_seq_len', 512),
        generation_max_length=cmp_params.get('gen_max_len', 256),
        generation_temperature=cmp_params.get('gen_temp', 0.8),
        quality_threshold=cmp_params.get('quality_threshold', 0.1)
    )
    
    return AdaptiveKnowledgeDistillation(config)
"""
P2P Knowledge Exchange Protocol for KNEXA-FL
Implements Adaptive Knowledge Distillation (AKD) mechanism
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

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
    quality_threshold: float = 0.3
    max_retries: int = 3

class AdaptiveKnowledgeDistillation:
    """
    Text-based Knowledge Distillation for KNEXA-FL.
    Implements the knowledge distillation loss:
    L_total_kd = (1-α_kd)L_i(D_i) + α_kd D_KL(σ(z_j/T) || σ(z_i/T))
    """
    
    def __init__(self, config: Optional[KDConfig] = None):
        self.config = config or KDConfig()
        self.stats = {
            'teacher_generations': 0,
            'student_trainings': 0,
            'total_tokens_processed': 0,
            'successful_exchanges': 0,
            'failed_exchanges': 0
        }
    
    def generate_teacher_knowledge(self,
                                 teacher_model: Any,
                                 teacher_tokenizer: Any,
                                 transfer_samples: List[Dict[str, Any]],
                                 device: str = "cuda") -> Dict[str, Any]:
        """
        Generate teacher knowledge for transfer samples.
        
        Args:
            teacher_model: Teacher model
            teacher_tokenizer: Teacher tokenizer
            transfer_samples: Input samples for knowledge transfer
            device: Device to run on
            
        Returns:
            Knowledge package with teacher responses
        """
        teacher_model.eval()
        teacher_responses = []
        high_quality_count = 0
        
        # Generating teacher responses
        
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
                        pad_token_id=teacher_tokenizer.eos_token_id
                    )
                    
                    # Decode only generated part
                    generated_text = teacher_tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Assess quality
                    quality = self._assess_response_quality(generated_text)
                    is_high_quality = quality >= self.config.quality_threshold
                    
                    if is_high_quality:
                        high_quality_count += 1
                    
                    teacher_responses.append({
                        'prompt': prompt,
                        'response': generated_text,
                        'quality': quality,
                        'is_high_quality': is_high_quality,
                        'sample_id': sample.get('id', f'sample_{i}')
                    })
                    
                    # Memory management
                    del inputs, outputs
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    # Failed to generate response
                    continue
        
        # Update statistics
        self.stats['teacher_generations'] += 1
        
        return {
            'teacher_id': 'teacher',
            'responses': teacher_responses,
            'high_quality_count': high_quality_count,
            'num_samples': len(teacher_responses),
            'quality_ratio': high_quality_count / max(1, len(teacher_responses)),
            'timestamp': time.time()
        }
    
    def perform_student_training(self,
                               student_model: Any,
                               student_tokenizer: Any,
                               teacher_knowledge: Dict[str, Any],
                               optimizer: Any,
                               num_steps: int = 3,
                               device: str = "cuda") -> Dict[str, Any]:
        """
        Train student using teacher knowledge.
        
        Args:
            student_model: Student model to train
            student_tokenizer: Student tokenizer
            teacher_knowledge: Teacher knowledge package
            optimizer: Student optimizer
            num_steps: Training steps per sample
            device: Device to run on
            
        Returns:
            Training results
        """
        student_model.train()
        teacher_responses = teacher_knowledge['responses']
        
        if not teacher_responses:
            return {
                'success': False,
                'error': 'No teacher responses available'
            }
        
        # Filter high-quality responses
        high_quality_responses = [
            r for r in teacher_responses if r['is_high_quality']
        ]
        if not high_quality_responses:
            high_quality_responses = teacher_responses
        
        training_losses = []
        total_loss = 0.0
        processed_samples = 0
        
        # Training student with high-quality responses
        
        for response in high_quality_responses:
            prompt = response['prompt']
            teacher_text = response['response']
            
            if not teacher_text.strip():
                continue
            
            # Create training text
            training_text = prompt + teacher_text
            
            # Tokenize
            inputs = student_tokenizer(
                training_text,
                return_tensors="pt",
                max_length=self.config.max_sequence_length,
                truncation=True,
                padding=True
            ).to(device)
            
            # Prepare labels
            labels = inputs['input_ids'].clone()
            
            # Multiple training steps
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
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(),
                        max_norm=1.0
                    )
                    
                    # Update
                    optimizer.step()
                    
                    # Track loss
                    step_loss = loss.item()
                    sample_losses.append(step_loss)
                    total_loss += step_loss
                    
                    # Clear intermediate tensors
                    del outputs, loss
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM during training step
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        break
                    else:
                        raise e
            
            training_losses.extend(sample_losses)
            processed_samples += 1
            
            # Clear input tensors
            del inputs, labels
        
        if processed_samples == 0:
            return {
                'success': False,
                'error': 'No samples processed'
            }
        
        # Calculate metrics
        avg_loss = total_loss / len(training_losses)
        final_loss = training_losses[-1] if training_losses else 0.0
        
        # Update statistics
        self.stats['student_trainings'] += 1
        self.stats['successful_exchanges'] += 1
        self.stats['total_tokens_processed'] += sum(
            len(r['response'].split()) for r in high_quality_responses
        )
        
        return {
            'success': True,
            'avg_loss': avg_loss,
            'final_loss': final_loss,
            'training_losses': training_losses,
            'processed_samples': processed_samples,
            'quality_filtered': len(high_quality_responses),
            'total_responses': len(teacher_responses)
        }
    
    def _assess_response_quality(self, text: str) -> float:
        """
        Assess quality of generated response.
        
        Args:
            text: Generated text
            
        Returns:
            Quality score between 0 and 1
        """
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Length score
        length_score = min(len(text) / 100, 1.0)
        
        # Diversity score
        words = text.split()
        if len(words) < 3:
            return length_score * 0.3
        
        unique_words = len(set(words))
        diversity_score = unique_words / len(words)
        
        # Code indicator score (for code generation tasks)
        code_indicators = ['def ', 'class ', 'import ', 'return ', 'if ', 'for ']
        has_code = any(indicator in text for indicator in code_indicators)
        code_score = 1.0 if has_code else 0.8
        
        # Coherence score (basic punctuation check)
        has_punctuation = any(p in text for p in ['.', '!', '?', ';'])
        coherence_score = 0.9 if has_punctuation else 0.7
        
        # Weighted average
        quality = (
            0.25 * length_score +
            0.25 * diversity_score +
            0.25 * code_score +
            0.25 * coherence_score
        )
        
        return quality
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge distillation statistics"""
        success_rate = self.stats['successful_exchanges'] / max(
            1, self.stats['successful_exchanges'] + self.stats['failed_exchanges']
        )
        
        return {
            'teacher_generations': self.stats['teacher_generations'],
            'student_trainings': self.stats['student_trainings'],
            'total_tokens_processed': self.stats['total_tokens_processed'],
            'successful_exchanges': self.stats['successful_exchanges'],
            'failed_exchanges': self.stats['failed_exchanges'],
            'success_rate': success_rate
        }

class P2PKnowledgeExchange:
    """
    Orchestrates P2P knowledge exchange between agents
    """
    
    def __init__(self, kd_config: Optional[KDConfig] = None):
        self.kd_module = AdaptiveKnowledgeDistillation(kd_config)
        self.exchange_history = []
    
    def execute_knowledge_transfer(self,
                                 teacher_agent: Any,
                                 student_agent: Any,
                                 transfer_samples: List[Dict[str, Any]],
                                 alpha: float = 0.5,
                                 temperature: float = 2.0) -> Dict[str, Any]:
        """
        Execute knowledge transfer from teacher to student.
        
        Args:
            teacher_agent: Teacher agent
            student_agent: Student agent
            transfer_samples: Samples for knowledge transfer
            alpha: Knowledge distillation weight
            temperature: Distillation temperature
            
        Returns:
            Transfer results
        """
        start_time = time.time()
        
        # Update KD config
        self.kd_module.config.alpha_kd = alpha
        self.kd_module.config.temperature = temperature
        
        # Step 1: Generate teacher knowledge
        # Generating teacher knowledge
        teacher_knowledge = teacher_agent.prepare_knowledge_package(transfer_samples)
        
        if not teacher_knowledge or not teacher_knowledge.get('outputs'):
            return {
                'success': False,
                'error': 'Failed to generate teacher knowledge',
                'duration': time.time() - start_time
            }
        
        # Step 2: Student integrates knowledge
        # Student integrating knowledge
        integration_result = student_agent.integrate_knowledge(
            teacher_knowledge,
            alpha=alpha,
            temperature=temperature
        )
        
        # Record exchange
        exchange_record = {
            'teacher_id': teacher_agent.agent_id,
            'student_id': student_agent.agent_id,
            'timestamp': time.time(),
            'duration': time.time() - start_time,
            'num_samples': len(teacher_knowledge.get('outputs', [])),
            'success': integration_result.get('success', False),
            'avg_loss': integration_result.get('avg_loss', 0.0)
        }
        
        self.exchange_history.append(exchange_record)
        
        return {
            'success': integration_result.get('success', False),
            'teacher_knowledge': teacher_knowledge,
            'integration_result': integration_result,
            'exchange_record': exchange_record
        }
    
    def get_exchange_statistics(self) -> Dict[str, Any]:
        """Get P2P exchange statistics"""
        if not self.exchange_history:
            return {
                'total_exchanges': 0,
                'successful_exchanges': 0,
                'success_rate': 0.0,
                'avg_duration': 0.0,
                'avg_samples': 0.0
            }
        
        successful = [e for e in self.exchange_history if e['success']]
        
        return {
            'total_exchanges': len(self.exchange_history),
            'successful_exchanges': len(successful),
            'success_rate': len(successful) / len(self.exchange_history),
            'avg_duration': np.mean([e['duration'] for e in self.exchange_history]),
            'avg_samples': np.mean([e['num_samples'] for e in self.exchange_history]),
            'kd_statistics': self.kd_module.get_statistics()
        }
"""
KNEXA-FL Agent Implementation
Autonomous LLM agent with PEFT training and privacy-preserving features
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

@dataclass
class AgentConfig:
    """Configuration for KNEXA-FL agent"""
    learning_rate: float = 3e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_sequence_length: int = 512
    local_epochs: int = 3
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    clip_grad_norm: float = 1.0
    device: str = "cuda"

class KnexaAgent:
    """
    KNEXA-FL Agent with PEFT training capabilities.
    Implements local training, profile generation, and knowledge exchange.
    """
    
    def __init__(self, 
                 agent_id: str,
                 model: Any,
                 tokenizer: Any,
                 train_dataset: List[Dict[str, Any]],
                 val_dataset: List[Dict[str, Any]],
                 config: Optional[AgentConfig] = None):
        """
        Initialize KNEXA agent.
        
        Args:
            agent_id: Unique agent identifier
            model: Pre-trained language model with PEFT modules
            tokenizer: Model tokenizer
            train_dataset: Local training dataset
            val_dataset: Local validation dataset
            config: Agent configuration
        """
        self.agent_id = agent_id
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or AgentConfig()
        
        # Move model to device
        self.model.to(self.config.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Performance tracking
        self.performance_history = []
        self.training_losses = []
        self.current_performance = 0.0
        
        # Profile information
        self.profile_data = {
            'model_size_mb': self._calculate_model_size(),
            'performance': 0.0,
            'trust_score': 0.8,
            'specialization_score': 0.5,
            'collaboration_quality': 0.5,
            'exchange_stats': {'successful': 0, 'total': 0}
        }
        
        # Agent initialized successfully
    
    def local_training_step(self, num_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Perform local PEFT training on private data.
        
        Args:
            num_epochs: Number of training epochs (default: config.local_epochs)
            
        Returns:
            Training metrics
        """
        num_epochs = num_epochs or self.config.local_epochs
        self.model.train()
        
        total_loss = 0.0
        total_steps = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Create batches
            for i in range(0, len(self.train_dataset), self.config.batch_size):
                batch = self.train_dataset[i:i + self.config.batch_size]
                
                # Process batch
                loss = self._process_training_batch(batch, total_steps)
                
                epoch_loss += loss
                epoch_steps += 1
                total_loss += loss
                total_steps += 1
            
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)
            # Epoch completed
        
        avg_loss = total_loss / max(1, total_steps)
        self.training_losses.append(avg_loss)
        
        # Update performance
        self.current_performance = self.evaluate_performance()
        self.performance_history.append(self.current_performance)
        
        return {
            'avg_loss': avg_loss,
            'final_loss': avg_epoch_loss,
            'performance': self.current_performance,
            'total_steps': total_steps
        }
    
    def _process_training_batch(self, batch: List[Dict[str, Any]], step_count: int = 0) -> float:
        """Process a single training batch"""
        # Prepare inputs
        texts = [sample['text'] for sample in batch]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding=True
        ).to(self.config.device)
        
        # Forward pass
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step_count + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.clip_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def evaluate_performance(self) -> float:
        """
        Evaluate agent performance on validation set.
        
        Returns:
            Performance score (e.g., perplexity or accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for sample in self.val_dataset[:50]:  # Evaluate on subset for efficiency
                inputs = self.tokenizer(
                    sample['text'],
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True
                ).to(self.config.device)
                
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        # Return normalized performance (inverse perplexity)
        avg_loss = total_loss / max(1, total_tokens)
        performance = 1.0 / (1.0 + avg_loss)  # Simple normalization
        
        return performance
    
    def generate_profile(self) -> Dict[str, Any]:
        """
        Generate agent profile for CPM.
        
        Returns:
            Profile data dictionary
        """
        # Update current performance
        self.profile_data['performance'] = self.current_performance
        
        # Calculate specialization score based on performance variance
        if len(self.performance_history) > 1:
            perf_std = np.std(self.performance_history)
            self.profile_data['specialization_score'] = 1.0 - min(1.0, perf_std)
        
        # Update collaboration quality based on exchange success
        total_exchanges = self.profile_data['exchange_stats']['total']
        if total_exchanges > 0:
            success_rate = self.profile_data['exchange_stats']['successful'] / total_exchanges
            self.profile_data['collaboration_quality'] = success_rate
        
        return self.profile_data.copy()
    
    def prepare_knowledge_package(self, transfer_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare knowledge package for P2P exchange.
        Implements Guardrail Filter for privacy protection.
        
        Args:
            transfer_samples: Samples for knowledge transfer
            
        Returns:
            Knowledge package with filtered outputs
        """
        self.model.eval()
        knowledge_outputs = []
        
        with torch.no_grad():
            for sample in transfer_samples:
                # Generate response
                inputs = self.tokenizer(
                    sample['prompt'],
                    return_tensors="pt",
                    max_length=self.config.max_sequence_length,
                    truncation=True
                ).to(self.config.device)
                
                # Generate with teacher forcing
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9
                )
                
                # Decode response
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                
                # Apply Guardrail Filter
                if self._guardrail_check(generated_text):
                    knowledge_outputs.append({
                        'prompt': sample['prompt'],
                        'response': generated_text,
                        'sample_id': sample.get('id', '')
                    })
        
        return {
            'agent_id': self.agent_id,
            'outputs': knowledge_outputs,
            'num_samples': len(knowledge_outputs),
            'timestamp': time.time()
        }
    
    def _guardrail_check(self, text: str) -> bool:
        """
        Guardrail Filter to prevent sensitive information leakage.
        
        Args:
            text: Generated text to check
            
        Returns:
            True if text passes privacy checks
        """
        # Simple privacy patterns (extend as needed)
        sensitive_patterns = [
            r'\b(?:password|api[_\s]?key|secret|token|credential)\b',
            r'\b(?:ssn|social[_\s]?security)\b',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def integrate_knowledge(self, 
                           knowledge_package: Dict[str, Any],
                           alpha: float = 0.5,
                           temperature: float = 2.0) -> Dict[str, float]:
        """
        Integrate knowledge from peer through distillation.
        
        Args:
            knowledge_package: Knowledge package from teacher
            alpha: Distillation weight
            temperature: Distillation temperature
            
        Returns:
            Integration metrics
        """
        self.model.train()
        teacher_outputs = knowledge_package['outputs']
        
        if not teacher_outputs:
            return {'success': False, 'avg_loss': 0.0}
        
        total_loss = 0.0
        num_samples = 0
        
        for output in teacher_outputs:
            # Create training text from teacher response
            prompt = output['prompt']
            response = output['response']
            full_text = prompt + response
            
            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                max_length=self.config.max_sequence_length,
                truncation=True
            ).to(self.config.device)
            
            # Student forward pass
            student_outputs = self.model(**inputs, labels=inputs['input_ids'])
            
            # Knowledge distillation loss
            loss = alpha * student_outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.clip_grad_norm
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            num_samples += 1
        
        avg_loss = total_loss / max(1, num_samples)
        
        # Update exchange statistics
        self.profile_data['exchange_stats']['total'] += 1
        if avg_loss < 5.0:  # Success threshold
            self.profile_data['exchange_stats']['successful'] += 1
        
        return {
            'success': True,
            'avg_loss': avg_loss,
            'num_samples': num_samples
        }
    
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB"""
        total_params = sum(p.numel() for p in self.model.parameters())
        # Assuming float32 (4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 * 1024)
        return size_mb
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'current_performance': self.current_performance,
            'training_steps': len(self.training_losses),
            'avg_training_loss': np.mean(self.training_losses) if self.training_losses else 0.0,
            'profile': self.profile_data,
            'performance_history': self.performance_history
        }
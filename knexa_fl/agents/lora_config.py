"""
LoRA (Low-Rank Adaptation) Configuration for KNEXA-FL
Parameter-Efficient Fine-Tuning setup for heterogeneous LLMs
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    rank: int = 16                      # Low-rank dimension
    alpha: int = 32                     # LoRA scaling parameter
    dropout: float = 0.1                # Dropout probability
    target_modules: List[str] = None    # Modules to apply LoRA
    bias: str = "none"                  # Bias training strategy
    task_type: str = "CAUSAL_LM"        # Task type
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for common architectures
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

class AdaptiveLoRAOptimizer:
    """
    Adaptive LoRA configuration optimizer for heterogeneous models
    """
    
    def __init__(self):
        # Predefined configurations for different model families
        self.model_configs = {
            # Small models (< 500M parameters)
            "small": LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            ),
            
            # Medium models (500M - 1B parameters)
            "medium": LoRAConfig(
                rank=16,
                alpha=32,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            
            # Large models (> 1B parameters)
            "large": LoRAConfig(
                rank=32,
                alpha=64,
                dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                               "up_proj", "down_proj", "gate_proj"]
            )
        }
        
        # Architecture-specific configurations
        self.architecture_configs = {
            "gpt": {
                "target_modules": ["c_attn", "c_proj", "c_fc"]
            },
            "bloom": {
                "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            },
            "llama": {
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", 
                                 "up_proj", "down_proj", "gate_proj"]
            },
            "t5": {
                "target_modules": ["q", "v", "k", "o", "wi", "wo"]
            },
            "qwen": {
                "target_modules": ["c_attn", "c_proj", "mlp.w1", "mlp.w2"]
            }
        }
    
    def get_optimal_config(self, 
                          model_name: str,
                          model_size_mb: float,
                          available_memory_mb: float = 8192) -> LoRAConfig:
        """
        Get optimal LoRA configuration based on model characteristics.
        
        Args:
            model_name: Name of the model
            model_size_mb: Model size in MB
            available_memory_mb: Available GPU memory in MB
            
        Returns:
            Optimized LoRA configuration
        """
        # Determine size category
        if model_size_mb < 1000:  # < 500M params (assuming 2MB per million)
            size_category = "small"
        elif model_size_mb < 2000:  # 500M - 1B params
            size_category = "medium"
        else:
            size_category = "large"
        
        # Start with size-based config
        config = self.model_configs[size_category]
        
        # Adjust for architecture
        architecture = self._detect_architecture(model_name)
        if architecture in self.architecture_configs:
            config.target_modules = self.architecture_configs[architecture]["target_modules"]
        
        # Adjust rank based on available memory
        memory_factor = available_memory_mb / 8192  # Normalize to 8GB baseline
        if memory_factor < 0.5:
            config.rank = max(4, config.rank // 2)
            config.alpha = config.rank * 2
        elif memory_factor > 2.0:
            config.rank = min(64, config.rank * 2)
            config.alpha = config.rank * 2
        
        return config
    
    def _detect_architecture(self, model_name: str) -> str:
        """Detect model architecture from name"""
        model_name_lower = model_name.lower()
        
        if "gpt" in model_name_lower:
            return "gpt"
        elif "bloom" in model_name_lower:
            return "bloom"
        elif "llama" in model_name_lower:
            return "llama"
        elif "t5" in model_name_lower:
            return "t5"
        elif "qwen" in model_name_lower:
            return "qwen"
        else:
            return "unknown"
    
    def calculate_lora_parameters(self, config: LoRAConfig, base_params: int) -> Dict[str, int]:
        """
        Calculate number of trainable parameters with LoRA.
        
        Args:
            config: LoRA configuration
            base_params: Number of base model parameters
            
        Returns:
            Parameter statistics
        """
        # Estimate based on typical transformer architecture
        # Assuming attention modules are ~30% of total parameters
        attention_params = int(base_params * 0.3)
        
        # LoRA adds rank * (input_dim + output_dim) parameters per module
        # Rough estimate: each module has sqrt(attention_params/4) dimension
        module_dim = int((attention_params / 4) ** 0.5)
        
        lora_params_per_module = config.rank * (module_dim * 2)
        total_lora_params = lora_params_per_module * len(config.target_modules)
        
        return {
            "base_parameters": base_params,
            "lora_parameters": total_lora_params,
            "total_parameters": base_params + total_lora_params,
            "trainable_percentage": (total_lora_params / base_params) * 100,
            "rank": config.rank,
            "num_modules": len(config.target_modules)
        }
    
    def merge_lora_weights(self, 
                          base_state_dict: Dict[str, torch.Tensor],
                          lora_state_dict: Dict[str, torch.Tensor],
                          config: LoRAConfig) -> Dict[str, torch.Tensor]:
        """
        Merge LoRA weights back into base model.
        
        Args:
            base_state_dict: Base model state dict
            lora_state_dict: LoRA adapter state dict
            config: LoRA configuration
            
        Returns:
            Merged state dict
        """
        merged_state_dict = base_state_dict.copy()
        scaling = config.alpha / config.rank
        
        for name, param in lora_state_dict.items():
            if "lora_A" in name:
                # Find corresponding lora_B
                base_name = name.replace("lora_A", "")
                lora_b_name = name.replace("lora_A", "lora_B")
                
                if lora_b_name in lora_state_dict:
                    # Compute LoRA update: W + scaling * B @ A
                    lora_a = lora_state_dict[name]
                    lora_b = lora_state_dict[lora_b_name]
                    
                    # Find base weight
                    for base_key in base_state_dict:
                        if base_name in base_key and "weight" in base_key:
                            base_weight = base_state_dict[base_key]
                            lora_update = scaling * (lora_b @ lora_a)
                            merged_state_dict[base_key] = base_weight + lora_update
                            break
        
        return merged_state_dict
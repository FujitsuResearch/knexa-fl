#!/usr/bin/env python3
"""
Simplified LoRA Optimizer for KNEXA-FL
Uses actual empirical measurement to determine optimal LoRA parameters
"""

import torch
import logging
from typing import Dict, Tuple, List
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import gc

logger = logging.getLogger(__name__)

class LoRAOptimizer:
    """
    Research-based LoRA parameter optimizer using empirical measurement
    """
    
    def __init__(self):
        self.target_percentages = {
            "small": 4.0,    # <200M params
            "medium": 3.0,   # 200M-500M  
            "large": 2.5     # >500M params
        }
    
    def count_parameters(self, model):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    
    def test_lora_rank(self, model_name: str, target_modules: List[str], rank: int) -> Tuple[int, int, float]:
        """Test a specific LoRA rank and return parameter counts"""
        try:
            # Load model (use CPU to save GPU memory for testing)
            model_info = LLM_REGISTRY.get(model_name, {"type": "decoder"})
            
            if model_info["type"] == "encoder-decoder":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            
            # Get base parameter count
            total_params, _ = self.count_parameters(model)
            
            # Apply LoRA
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank * 2,  # Simple 2x ratio
                lora_dropout=0.05,
                target_modules=target_modules
            )
            
            peft_model = get_peft_model(model, lora_config)
            total_with_lora, trainable = self.count_parameters(peft_model)
            
            # Calculate percentage
            trainable_percentage = (trainable / total_params) * 100
            
            # Clean up
            del model, peft_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return total_params, trainable, trainable_percentage
            
        except Exception as e:
            logger.error(f"Error testing rank {rank} for {model_name}: {e}")
            return 0, 0, 0.0
    
    def find_optimal_rank(self, model_name: str, target_modules: List[str], 
                         target_percentage: float = None) -> Tuple[int, int]:
        """Find optimal rank using binary search"""
        
        # Get model info
        model_info = LLM_REGISTRY.get(model_name, {"params": "100M"})
        params_str = model_info.get("params", "100M")
        
        # Parse parameter count  
        if 'B' in params_str:
            param_count = float(params_str.replace('B', '')) * 1e9
        elif 'M' in params_str:
            param_count = float(params_str.replace('M', '')) * 1e6
        else:
            param_count = 100e6
        
        # Determine target percentage
        if target_percentage is None:
            if param_count < 200e6:
                target_percentage = self.target_percentages["small"]
            elif param_count < 500e6:
                target_percentage = self.target_percentages["medium"]
            else:
                target_percentage = self.target_percentages["large"]
        
        logger.info(f"🎯 Finding optimal LoRA rank for {model_name}")
        logger.info(f"   Target trainable percentage: {target_percentage:.1f}%")
        
        # Test a range of ranks to find optimal
        best_rank = 8
        best_alpha = 16
        best_diff = float('inf')
        
        ranks_to_test = [4, 8, 16, 24, 32, 48, 64]  # Reasonable range
        
        for rank in ranks_to_test:
            total_params, trainable, actual_percentage = self.test_lora_rank(
                model_name, target_modules, rank
            )
            
            if actual_percentage > 0:
                diff = abs(actual_percentage - target_percentage)
                
                logger.info(f"   Rank {rank:2d}: {trainable:,} params ({actual_percentage:.2f}%) - "
                           f"diff from target: {diff:.2f}%")
                
                if diff < best_diff:
                    best_diff = diff
                    best_rank = rank
                    best_alpha = rank * 2  # Simple 2x ratio for reliability
        
        logger.info(f"✅ Optimal configuration: rank={best_rank}, alpha={best_alpha}")
        logger.info(f"   Best difference from target: {best_diff:.2f}%")
        
        return best_rank, best_alpha

def get_optimal_lora_params(model_name: str, target_modules: List[str]) -> Tuple[int, int]:
    """Get optimal LoRA parameters for a model"""
    optimizer = LoRAOptimizer()
    return optimizer.find_optimal_rank(model_name, target_modules)

# For backward compatibility with globals import
from src.globals import LLM_REGISTRY
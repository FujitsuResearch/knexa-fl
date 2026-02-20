import torch, logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
from src.globals import *
from src.lora_optimizer import get_optimal_lora_params
 
logger = logging.getLogger(__name__)
 
def load_model_and_tokenizer(cid: int, device: str):
    base_name = MODEL_MAP[cid % len(MODEL_MAP)]
    model_info = LLM_REGISTRY.get(base_name, {"arch": "unknown", "type": "decoder", "params": "unknown"})
    
    logger.info(f"[Load] CID{cid} → {base_name} ({model_info['params']}, {model_info['arch']}) on {device}")
    
    # Load appropriate model type based on architecture
    if model_info["type"] == "encoder-decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(base_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_name).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
        else:
            # For T5 models, use the default pad token
            tokenizer.pad_token = tokenizer.special_tokens_map.get('pad_token', '</s>')
    
    # Determine target modules based on model architecture from registry
    arch = model_info["arch"]
    
    if arch == "pythia":
        # Pythia models (GPT-NeoX architecture)
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif arch == "opt":
        # OPT models (Facebook)
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif arch == "t5" or arch == "mt5":
        # T5/mT5 models (encoder-decoder)
        target_modules = ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    elif arch == "codegen":
        # CodeGen models (Salesforce)
        target_modules = ["qkv_proj", "out_proj", "fc"]
    elif arch == "bloom":
        # BLOOM models
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif arch == "cerebras":
        # Cerebras-GPT models
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif arch == "qwen":
        # Qwen models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif arch == "gpt2":
        # GPT-2 and DistilGPT-2
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif arch == "gpt-neo":
        # GPT-Neo models
        target_modules = ["q_proj", "k_proj", "v_proj", "c_fc", "c_proj"]
    elif arch == "dialogpt":
        # DialoGPT models (GPT-2 based)
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif arch == "codegpt":
        # CodeGPT models (GPT-2 based)
        target_modules = ["c_attn", "c_proj", "c_fc"]
    elif arch == "codellama":
        # CodeLlama models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        # Default fallback (GPT-2 style)
        target_modules = ["c_attn", "c_proj", "c_fc"]
    
    # Get optimal LoRA configuration using empirical measurement
    try:
        optimal_rank, optimal_alpha = get_optimal_lora_params(base_name, target_modules)
        
        logger.info(f"📊 OPTIMAL LoRA Configuration for {base_name}:")
        logger.info(f"   Empirically-Optimized Rank: {optimal_rank} (vs. legacy {LORA_RANK})")
        logger.info(f"   Calculated Alpha: {optimal_alpha} (vs. legacy {LORA_ALPHA})")
        
        # Use optimal configuration
        lora = LoraConfig(
            r=optimal_rank, 
            lora_alpha=optimal_alpha, 
            lora_dropout=LORA_DROPOUT,
            target_modules=target_modules
        )
        
    except Exception as e:
        logger.warning(f"⚠️ LoRA optimization failed: {e}")
        logger.info(f"🔧 Fallback to legacy configuration: rank={LORA_RANK}, alpha={LORA_ALPHA}")
        
        # Fallback to legacy configuration
        lora = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=target_modules
        )
    
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    
    # Log final configuration details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100
    
    logger.info(f"✅ Final LoRA Statistics:")
    logger.info(f"   Total Parameters: {total_params:,}")
    logger.info(f"   Trainable Parameters: {trainable_params:,}")
    logger.info(f"   Trainable Percentage: {trainable_percentage:.2f}%")
    
    # Calculate improvement over legacy (theoretical)
    legacy_percentage = 0.72  # Observed from logs for Pythia-160M
    if trainable_percentage > legacy_percentage:
        improvement_factor = trainable_percentage / legacy_percentage
        logger.info(f"📈 Estimated improvement over legacy: {improvement_factor:.1f}x trainable parameters")
    # Disable gradient checkpointing to avoid gradient issues
    # if hasattr(model, "gradient_checkpointing_enable"):
    #     model.gradient_checkpointing_enable()
    return model, tokenizer

def load_tokenizer_only(cid: int):
    """Load only tokenizer to avoid meta tensor issues in parallel execution"""
    base_name = MODEL_MAP[cid % len(MODEL_MAP)]
    logger.info(f"[TokenizerOnly] CID{cid} → {base_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
        else:
            tokenizer.pad_token = tokenizer.special_tokens_map.get('pad_token', '</s>')
    
    return tokenizer
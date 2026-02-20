"""
globals.py – Hyper-parameters and constants shared by all modules
"""

import math
import random
import numpy as np
import torch

# Master seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# General settings
NUM_CLIENTS = 4
NUM_ROUNDS = 30
LOCAL_EPOCHS = 1    # Reduced from 3 to prevent overfitting in federated non-IID setting
BATCH_LOCAL = 8     # Reduced from 16 for better memory efficiency and gradient quality
BATCH_KD = 32       # Full batch size for optimal KD throughput
GRADIENT_ACCUMULATION_STEPS = 8   # Maintain effective batch size of 64 for training stability
LR_LOCAL = 3e-5     # Reduced from 1e-4 for better stability with pre-trained LLMs
LR_KD = 5e-5        # Increased from 1e-5 to make KD updates more meaningful
WARMUP_STEPS = 100  # Added warmup to reduce early training loss spikes
DEVICE_MAP = {0: 0, 1: 0, 2: 0, 3: 0}  # CID → GPU index (single H100, optimized for 90GB)
ROUND_TIMEOUT_S = 120  # Increased timeout for larger batches

# Comprehensive LLM Registry for Heterogeneous Federated Learning
# All models from the specified list - organized by size and architecture type
LLM_REGISTRY = {
    # Ultra-Small Models (70-85M parameters)
    "EleutherAI/pythia-70m": {"params": "70M", "arch": "pythia", "type": "decoder", "license": "Apache 2.0"},
    "google/t5-v1_1-small": {"params": "77M", "arch": "t5", "type": "encoder-decoder", "license": "Apache 2.0"},
    "distilgpt2": {"params": "82M", "arch": "gpt2", "type": "decoder", "license": "Apache 2.0"},
    
    # Small Models (100-160M parameters)  
    "cerebras/Cerebras-GPT-111M": {"params": "111M", "arch": "cerebras", "type": "decoder", "license": "Apache 2.0"},
    "microsoft/DialoGPT-small": {"params": "117M", "arch": "gpt2", "type": "decoder", "license": "MIT"},
    "gpt2": {"params": "124M", "arch": "gpt2", "type": "decoder", "license": "MIT"},
    "EleutherAI/gpt-neo-125M": {"params": "125M", "arch": "gpt-neo", "type": "decoder", "license": "MIT"},
    "facebook/opt-125m": {"params": "125M", "arch": "opt", "type": "decoder", "license": "Non-commercial"},
    "EleutherAI/pythia-160m": {"params": "160M", "arch": "pythia", "type": "decoder", "license": "Apache 2.0"},
    
    # Medium Models (250-410M parameters)
    "cerebras/Cerebras-GPT-256M": {"params": "256M", "arch": "cerebras", "type": "decoder", "license": "Apache 2.0"},
    "google/mt5-small": {"params": "300M", "arch": "mt5", "type": "encoder-decoder", "license": "Apache 2.0"},
    "facebook/opt-350m": {"params": "350M", "arch": "opt", "type": "decoder", "license": "Non-commercial"},
    "Salesforce/codegen-350M-mono": {"params": "350M", "arch": "codegen", "type": "decoder", "license": "BSD-3-Clause"},
    "EleutherAI/pythia-410m": {"params": "410M", "arch": "pythia", "type": "decoder", "license": "Apache 2.0"},
    
    # Large Models (560-620M parameters)
    "bigscience/bloom-560m": {"params": "560M", "arch": "bloom", "type": "decoder", "license": "BigScience RAIL"},
    "cerebras/Cerebras-GPT-590M": {"params": "590M", "arch": "cerebras", "type": "decoder", "license": "Apache 2.0"},
    "Qwen/Qwen1.5-0.5B": {"params": "620M", "arch": "qwen", "type": "decoder", "license": "Apache 2.0"},
}

# Current 4-client heterogeneous configuration (from specified list)
MODEL_MAP = {
    0: "EleutherAI/pythia-160m",          # 160M Pythia (GPT-NeoX style)
    1: "facebook/opt-125m",               # 125M OPT (ReLU activations)
    2: "distilgpt2",                      # 82M DistilGPT-2 (decoder-only, compatible)
    3: "Salesforce/codegen-350M-mono",    # 350M CodeGen (code-focused)
    4: "gpt2",                            # 124M GPT-2
    5: "cerebras/Cerebras-GPT-111M",      # 111M Cerebras-GPT
    6: "EleutherAI/gpt-neo-125M",         # 125M GPT-Neo
    7: "google/t5-v1_1-small",            # 77M T5 (moved to unused position)
}

# Predefined heterogeneous configurations from specified model list
HETEROGENEOUS_CONFIGS = {
    "small_diverse": {
        0: "EleutherAI/pythia-70m",       # 70M (ultra-small)
        1: "distilgpt2",                  # 82M (distilled)
        2: "cerebras/Cerebras-GPT-111M",  # 111M (hardware-optimized)
        3: "gpt2"                         # 124M (classic)
    },
    "medium_diverse": {
        0: "EleutherAI/pythia-160m",          # 160M Pythia
        1: "facebook/opt-125m",               # 125M OPT
        2: "distilgpt2",                      # 82M DistilGPT-2 (decoder-only, T5 replacement)
        3: "Salesforce/codegen-350M-mono"     # 350M CodeGen
    },
    "large_diverse": {
        0: "EleutherAI/pythia-410m",          # 410M Pythia
        1: "facebook/opt-350m",               # 350M OPT
        2: "bigscience/bloom-560m",           # 560M BLOOM
        3: "Qwen/Qwen1.5-0.5B"               # 620M Qwen
    },
    "architecture_diverse": {
        0: "EleutherAI/pythia-160m",          # GPT-NeoX style
        1: "facebook/opt-125m",               # OPT (ReLU)
        2: "gpt2",                            # Classic GPT-2 (decoder-only)
        3: "cerebras/Cerebras-GPT-256M"       # Hardware-optimized
    },
    "xlarge_diverse": {
        0: "EleutherAI/pythia-1b",            # 1B Pythia
        1: "facebook/opt-1.3b",               # 1.3B OPT
        2: "EleutherAI/pythia-1.4b",          # 1.4B Pythia
        3: "bigscience/bloom-1b7"             # 1.7B BLOOM
    },
    "ultra_large_diverse": {
        0: "EleutherAI/pythia-2.8b",          # 2.8B Pythia
        1: "facebook/opt-2.7b",               # 2.7B OPT
        2: "EleutherAI/pythia-6.9b",          # 6.9B Pythia
        3: "facebook/opt-6.7b"                # 6.7B OPT
    },
    "max_vram_cached": {
        0: "Qwen/Qwen1.5-0.5B",              # 620M Qwen (largest cached)
        1: "cerebras/Cerebras-GPT-590M",      # 590M Cerebras (2nd largest cached)
        2: "bigscience/bloom-560m",           # 560M BLOOM (3rd largest cached)
        3: "EleutherAI/pythia-410m"           # 410M Pythia (4th largest cached)
    },
    "stable_vram_cached": {
        0: "EleutherAI/pythia-410m",          # 410M Pythia (efficient)
        1: "facebook/opt-350m",               # 350M OPT (efficient)
        2: "EleutherAI/pythia-160m",          # 160M Pythia (smaller vocab)
        3: "facebook/opt-125m"                # 125M OPT (conservative)
    },
    "custom_experiment": {
        0: "microsoft/DialoGPT-small",        # 117M DialoGPT (conversational)
        1: "EleutherAI/pythia-410m",         # 410M Pythia (general)
        2: "facebook/opt-350m",              # 350M OPT (efficient)
        3: "bigscience/bloom-560m",          # 560M BLOOM (multilingual)
        4: "Qwen/Qwen1.5-0.5B",            # 620M Qwen (Chinese-English)
        5: "Salesforce/codegen-350M-mono",   # 350M CodeGen (code-focused)
        6: "gpt2",                           # 124M GPT-2 (classic)
        7: "distilgpt2"                      # 82M DistilGPT-2 (lightweight)
    },
    # New 8-client configurations
    "small_diverse_8": {
        0: "EleutherAI/pythia-70m",           # 70M Pythia (ultra-small)
        1: "distilgpt2",                      # 82M DistilGPT-2 (efficient)
        2: "cerebras/Cerebras-GPT-111M",      # 111M Cerebras (hardware-optimized)
        3: "gpt2",                            # 124M GPT-2 (classic)
        4: "EleutherAI/gpt-neo-125M",         # 125M GPT-Neo (alternative arch)
        5: "facebook/opt-125m",               # 125M OPT (ReLU activations)
        6: "EleutherAI/pythia-160m",          # 160M Pythia (larger variant)
        7: "google/t5-v1_1-small"             # 77M T5 (encoder-decoder)
    },
    "medium_diverse_8": {
        0: "EleutherAI/pythia-160m",          # 160M Pythia (GPT-NeoX)
        1: "facebook/opt-125m",               # 125M OPT (efficient)
        2: "distilgpt2",                      # 82M DistilGPT-2 (lightweight)
        3: "Salesforce/codegen-350M-mono",    # 350M CodeGen (code-specialized)
        4: "cerebras/Cerebras-GPT-256M",      # 256M Cerebras (hardware-optimized)
        5: "facebook/opt-350m",               # 350M OPT (larger variant)
        6: "EleutherAI/pythia-410m",          # 410M Pythia (robust)
        7: "gpt2"                             # 124M GPT-2 (baseline)
    },
    "large_diverse_8": {
        0: "EleutherAI/pythia-410m",          # 410M Pythia (efficient)
        1: "facebook/opt-350m",               # 350M OPT (ReLU)
        2: "bigscience/bloom-560m",           # 560M BLOOM (multilingual)
        3: "Qwen/Qwen1.5-0.5B",              # 620M Qwen (Chinese-English)
        4: "cerebras/Cerebras-GPT-590M",      # 590M Cerebras (large)
        5: "Salesforce/codegen-350M-mono",    # 350M CodeGen (code)
        6: "cerebras/Cerebras-GPT-256M",      # 256M Cerebras (medium)
        7: "EleutherAI/pythia-160m"           # 160M Pythia (small)
    },
    "architecture_diverse_8": {
        0: "EleutherAI/pythia-160m",          # GPT-NeoX architecture
        1: "facebook/opt-125m",               # OPT architecture (ReLU)
        2: "gpt2",                            # Classic GPT-2 
        3: "cerebras/Cerebras-GPT-256M",      # Cerebras optimized
        4: "bigscience/bloom-560m",           # BLOOM architecture
        5: "distilgpt2",                      # Distilled GPT-2
        6: "Salesforce/codegen-350M-mono",    # CodeGen architecture
        7: "google/t5-v1_1-small"             # T5 encoder-decoder
    },
    "balanced_cached_8": {
        0: "EleutherAI/pythia-410m",          # 410M (large, efficient)
        1: "facebook/opt-350m",               # 350M (medium-large)
        2: "cerebras/Cerebras-GPT-256M",      # 256M (medium)
        3: "EleutherAI/pythia-160m",          # 160M (small-medium)
        4: "facebook/opt-125m",               # 125M (small)
        5: "gpt2",                            # 124M (small, classic)
        6: "cerebras/Cerebras-GPT-111M",      # 111M (ultra-small)
        7: "distilgpt2"                       # 82M (ultra-small, efficient)
    },
    "code_focused_8": {
        0: "Salesforce/codegen-350M-mono",    # 350M CodeGen (primary code model)
        1: "EleutherAI/pythia-410m",          # 410M Pythia (general purpose)
        2: "facebook/opt-350m",               # 350M OPT (alternative)
        3: "gpt2",                            # 124M GPT-2 (baseline)
        4: "EleutherAI/pythia-160m",          # 160M Pythia (smaller)
        5: "cerebras/Cerebras-GPT-256M",      # 256M Cerebras (efficient)
        6: "facebook/opt-125m",               # 125M OPT (lightweight)
        7: "distilgpt2"                       # 82M DistilGPT-2 (minimal)
    },
    "max_performance_8": {
        0: "Qwen/Qwen1.5-0.5B",              # 620M (largest cached)
        1: "cerebras/Cerebras-GPT-590M",      # 590M (second largest)
        2: "bigscience/bloom-560m",           # 560M (multilingual)
        3: "EleutherAI/pythia-410m",          # 410M (efficient large)
        4: "facebook/opt-350m",               # 350M (medium-large)
        5: "Salesforce/codegen-350M-mono",    # 350M (code-focused)
        6: "cerebras/Cerebras-GPT-256M",      # 256M (medium)
        7: "EleutherAI/pythia-160m"           # 160M (small-medium)
    },
    "lightweight_8": {
        0: "EleutherAI/pythia-70m",           # 70M (ultra-small)
        1: "google/t5-v1_1-small",            # 77M (encoder-decoder)
        2: "distilgpt2",                      # 82M (distilled)
        3: "cerebras/Cerebras-GPT-111M",      # 111M (hardware-optimized)
        4: "gpt2",                            # 124M (classic)
        5: "EleutherAI/gpt-neo-125M",         # 125M (GPT-Neo)
        6: "facebook/opt-125m",               # 125M (OPT)
        7: "EleutherAI/pythia-160m"           # 160M (slightly larger)
    }
}

# Function to easily switch configurations
def set_model_configuration(config_name="medium_diverse"):
    """Set MODEL_MAP to one of the predefined configurations"""
    global MODEL_MAP
    if config_name in HETEROGENEOUS_CONFIGS:
        MODEL_MAP = HETEROGENEOUS_CONFIGS[config_name].copy()
        return True
    return False

# Set default configuration based on NUM_CLIENTS
# If NUM_CLIENTS > 4, use the full 8-client MODEL_MAP definition
# Otherwise, use optimized 4-client configuration
if NUM_CLIENTS <= 4:
    set_model_configuration("max_vram_cached")  # Using largest cached models (410M-620M)
elif NUM_CLIENTS <= 8:
    set_model_configuration("custom_experiment")  # Use custom config for 5-8 clients
# If NUM_CLIENTS > 4, MODEL_MAP already has 8 entries defined above

# LoRA hyperparameters
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Knowledge Distillation parameters (defaults, may be tuned by CPM at runtime)
TEMP_DEFAULT = 1.5  # Reduced from 2.0 for sharper soft targets in code generation
KD_ALPHA_GRID = [0.2, 0.3, 0.4]  # Reduced from [0.3, 0.5, 0.7] to keep teacher influence ≤40%
LAMBDA_PROX = 1e-2  # Increased from 1e-3 for stronger regularization to prevent catastrophic forgetting

# Local Recovery Round parameters (post-KD adaptation)
ENABLE_RECOVERY_ROUND = True  # Enable local recovery after KD
RECOVERY_STEPS = 50  # Number of local training steps for recovery
RECOVERY_LR = 5e-5  # Learning rate for recovery (0.1 * LR_LOCAL)
RECOVERY_LAMBDA_PROX = 2e-2  # Stronger proximal constraint during recovery (2 * LAMBDA_PROX)
RECOVERY_BATCH_SIZE = 4  # Batch size for recovery training

# Differential Privacy settings
CLIP_NORM = 1.0
GAUSS_NOISE_SIG = 0.1
DELTA_DP = 1e-5

# Communication & compression
TOPK = 100
KB_TARGET = 100
DELTA_KB = 1e-3  # Auto-tuned after round 2
SIER_THRESH = 0.01

# LinUCB (bandit) hyperparameters
LINUCB_LAMBDA = 0.01
LINUCB_BETA0 = 1.0

# Reward scaling
GAMMA_REWARD = 1.0

# Sequence length settings for maximum VRAM utilization
MAX_SEQ_LENGTH = 512  # Increased from 128 for higher VRAM usage
MAX_EVAL_LENGTH = 512  # Increased for evaluation accuracy

# ---------------------------------------------------------------------
# Evaluation frequency control
# ---------------------------------------------------------------------
# To balance computational cost with reporting fidelity we perform
# *full* evaluation (i.e.\ on the entire local-validation, global-test,
# and transfer sets) only every N communication rounds.  Intermediate
# rounds continue to use the small 5-sample probe previously in place.
# Set N=1 to evaluate every round.
# This can be overridden by command-line argument --eval-frequency
EVAL_FULL_EVERY_N_ROUNDS = 1

# ---------------------------------------------------------------------
# Unpaired client behavior control
# ---------------------------------------------------------------------
# This flag controls how clients without P2P role assignments behave during knowledge exchange rounds:
#
# When True (original behavior):
#   - Unpaired clients perform standard local training on their private data
#   - Only paired clients participate in knowledge distillation
#   - Some clients may not benefit from collaborative learning in each round
#
# When False (new behavior for complete P2P participation):
#   - All clients MUST be paired for knowledge distillation
#   - Unpaired clients skip local training to avoid conflicting gradients
#   - The pairing algorithm ensures all clients get role assignments
#   - Fallback pairings are created if the bandit doesn't cover all clients
#
# Use Cases:
#   - Set to True for experiments comparing P2P vs local-only training
#   - Set to False to ensure all clients benefit from knowledge transfer
#   - Set to False when local pretraining is complete and only KD is desired
#
ALLOW_UNPAIRED_LOCAL_ONLY = False  # Set to False to ensure all clients participate in KD

# Regex patterns for guardrails / SIER
PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email addresses
    r"[A-Fa-f0-9]{32}",  # generic API keys / hashes
    r"\b(?:fuck|shit|bitch)\b",  # mild profanity
]

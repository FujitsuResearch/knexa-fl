#!/bin/bash

# KNEXA-FL Experiment Script
# Comprehensive parameter exploration with full experimental control
# Advanced automation for systematic studies

set -e

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "KNEXA-FL experiment execution with comprehensive parameter control"
    echo ""
    echo "OPTIONS:"
    echo "  -r, --rounds NUM          Number of federated rounds (default: 25)"
    echo "  -s, --seed NUM            Random seed for reproducibility (default: 42)"
    echo "  -n, --name NAME           Experiment name (default: auto-generated)"
    echo "  -m, --method METHOD       Method variant (default: KNEXA-FL)"
    echo "  -M, --model-config CONFIG Model configuration preset (default: max_vram_cached)"
    echo "  -P, --pairing-mode MODE   Pairing strategy: bandit|heuristic|random (default: bandit)"
    echo "                            Available configurations:"
    echo "                            4-client configurations:"
    echo "                            - small_diverse: 70M-124M models (pythia-70m, distilgpt2, Cerebras-111M, gpt2)"
    echo "                            - medium_diverse: 82M-350M models (pythia-160m, opt-125m, distilgpt2, codegen-350M)"
    echo "                            - large_diverse: 350M-620M models (pythia-410m, opt-350m, bloom-560m, Qwen-0.5B)"
    echo "                            - architecture_diverse: 124M-256M models (pythia-160m, opt-125m, gpt2, Cerebras-256M)"
    echo "                            - stable_vram_cached: 125M-410M models (pythia-410m, opt-350m, pythia-160m, opt-125m)"
    echo "                            - max_vram_cached: 410M-620M models (Qwen-0.5B, Cerebras-590M, bloom-560m, pythia-410m)"
    echo "                            8-client configurations:"
    echo "                            - small_diverse_8: 70M-160M diverse models for lightweight experiments"
    echo "                            - medium_diverse_8: 82M-410M balanced mix of architectures"
    echo "                            - large_diverse_8: 160M-620M high-performance models"
    echo "                            - architecture_diverse_8: Maximum architectural diversity (8 different architectures)"
    echo "                            - balanced_cached_8: 82M-410M balanced size distribution"
    echo "                            - code_focused_8: Code-generation focused with CodeGen-350M lead"
    echo "                            - max_performance_8: 160M-620M largest available models"
    echo "                            - lightweight_8: 70M-160M ultra-lightweight for fast experiments"
    echo "  -c, --clients NUM         Number of clients (default: 4)"
    echo "  -p, --pass-at-k MODE      Pass@k evaluation mode: always|strategic|never (default: strategic)"
    echo "  -a, --alpha FLOAT         Dirichlet alpha for non-IID data (default: 0.1)"
    echo "  -l, --lr-local FLOAT      Local learning rate (default: 5e-5)"
    echo "  -k, --lr-kd FLOAT         Knowledge distillation learning rate (default: 1e-4)"
    echo "  -t, --temperature FLOAT   KD temperature (default: 2.0)"
    echo "  -d, --save-dir PATH       Save directory (default: auto-generated)"
    echo "  -e, --endpoint URL        CPM endpoint for P2P mode (default: localhost:8000)"
    echo "  -b, --batch-local NUM     Local batch size (default: 16)"
    echo "  -B, --batch-kd NUM        KD batch size (default: 8)"
    echo "  -L, --local-pretrain-rounds NUM  Initial local-only training rounds (default: 0)"
    echo "                                   No P2P collaboration during these rounds"
    echo "                                   Set equal to -r for local-only baseline"
    echo "  -g, --gpu NUM             GPU device to use (0 or 1, default: system default)"
    echo "  -q, --quick               Quick mode: reduced rounds for testing"
    echo "  -v, --verbose             Verbose logging"
    echo "  -f, --eval-frequency NUM  Evaluation frequency: full pass@k every N rounds (default: 1)"
    echo "                            Set to higher values to speed up experiments"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Run with all defaults"
    echo "  $0 -r 50 -s 123 -n \"My_Experiment\"  # Custom rounds, seed, and name"
    echo "  $0 -q                                 # Quick test run"
    echo "  $0 -p always -v                      # Full pass@k evaluation with verbose logging"
    echo "  $0 -r 100 -f 5                       # 100 rounds, evaluate every 5th round"
    echo ""
    exit 0
}

# Default values
ROUNDS=25
SEED=42
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME=""
METHOD="KNEXA-FL"
NUM_CLIENTS=4
PASS_AT_K="strategic"
PAIRING_MODE="bandit"
ALPHA_DIRICHLET=0.1
LR_LOCAL=5e-5
LR_KD=1e-4
TEMPERATURE=2.0
SAVE_DIR=""
CPM_ENDPOINT="localhost:8000"
BATCH_LOCAL=16
BATCH_KD=8
QUICK_MODE=false
VERBOSE=false
REAL_MODE=true
LOCAL_PRETRAIN_ROUNDS=0
GPU_ID=""  # Empty means use system default/all available GPUs
MODEL_CONFIG=""  # Empty means use default configuration
EVAL_FREQUENCY=1  # Default: evaluate every round

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -n|--name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -M|--model-config)
            MODEL_CONFIG="$2"
            shift 2
            ;;
        -P|--pairing-mode)
            PAIRING_MODE="$2"
            shift 2
            ;;
        -c|--clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        -p|--pass-at-k)
            PASS_AT_K="$2"
            shift 2
            ;;
        -a|--alpha)
            ALPHA_DIRICHLET="$2"
            shift 2
            ;;
        -l|--lr-local)
            LR_LOCAL="$2"
            shift 2
            ;;
        -k|--lr-kd)
            LR_KD="$2"
            shift 2
            ;;
        -t|--temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        -d|--save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        -e|--endpoint)
            CPM_ENDPOINT="$2"
            shift 2
            ;;
        -b|--batch-local)
            BATCH_LOCAL="$2"
            shift 2
            ;;
        -B|--batch-kd)
            BATCH_KD="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -L|--local-pretrain-rounds)
            LOCAL_PRETRAIN_ROUNDS="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        -f|--eval-frequency)
            EVAL_FREQUENCY="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Adjust parameters for quick mode
if [ "$QUICK_MODE" = true ]; then
    ROUNDS=5
    PASS_AT_K="strategic"
    echo "🚀 Quick mode enabled: rounds=$ROUNDS, pass@k=$PASS_AT_K"
fi

# Set default experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="${METHOD}_${TIMESTAMP}"
fi

# Note: SAVE_DIR is deprecated - Unified Artifact Manager now auto-generates timestamped directories
# ExperimentManager creates: YYYY-MM-DD_HH-MM-SS_<experiment_name> with unified structure
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="auto_generated_by_unified_system"
fi

# Select python interpreter
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python)}

export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# Critical memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_LAUNCH_BLOCKING=0

# Create minimal directory structure (ExperimentManager handles the rest)
mkdir -p experimental_artifacts/knexa_fl/results

# Set verbose logging if requested
if [ "$VERBOSE" = true ]; then
    export KNEXA_FL_VERBOSE=1
fi

# Display configuration
echo "============================================================"
echo "🚀 KNEXA-FL Experiment"
echo "============================================================"
echo "📊 Configuration:"
echo "   Experiment Name: $EXPERIMENT_NAME"
echo "   Method: $METHOD"
echo "   Rounds: $ROUNDS"
echo "   Seed: $SEED"
echo "   Clients: $NUM_CLIENTS"
echo "   Pass@k Mode: $PASS_AT_K"
echo "   Alpha (Dirichlet): $ALPHA_DIRICHLET"
echo "   LR Local: $LR_LOCAL"
echo "   LR KD: $LR_KD"
echo "   Temperature: $TEMPERATURE"
echo "   Batch Size Local: $BATCH_LOCAL"
echo "   Batch Size KD: $BATCH_KD"
echo "   Local Pretrain Rounds: $LOCAL_PRETRAIN_ROUNDS"
echo "   Model Config: ${MODEL_CONFIG:-default (max_vram_cached)}"
echo "   GPU Device: ${GPU_ID:-system default}"
echo "   Eval Frequency: $EVAL_FREQUENCY (full pass@k every $EVAL_FREQUENCY rounds)"
echo "   Save Directory: $SAVE_DIR"
echo "   CPM Endpoint: $CPM_ENDPOINT"
echo "   Quick Mode: $QUICK_MODE"
echo "   Verbose: $VERBOSE"
echo "   Timestamp: $TIMESTAMP"
echo "============================================================"
echo "🔬 FEATURES:"
echo "   ✅ Knowledge Transfer"
echo "   ✅ Advanced Experiment Management"
echo "   ✅ Automated Result Archival"
echo "   ✅ Code Generation Logging"
echo "   ✅ Comprehensive Reports"
echo "   ✅ Cross-Experiment Comparison"
echo "   ✅ Professional Standards"
echo "============================================================"

# Build additional arguments
ADDITIONAL_ARGS=""
if [ "$VERBOSE" = true ]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --verbose"
fi
if [ ! -z "$MODEL_CONFIG" ]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --model-config $MODEL_CONFIG"
fi

# Log experiment start
echo "🎭 Starting KNEXA-FL Experiment..."
echo "⏰ Start Time: $(date)"
echo "📋 Experiment ID will be generated automatically"
echo "🔗 All results will be saved with comprehensive management"
echo ""

# Create save directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Create experiment configuration file
cat > "$SAVE_DIR/run_config.json" << EOF
{
    "experiment_name": "$EXPERIMENT_NAME",
    "method": "$METHOD",
    "rounds": $ROUNDS,
    "seed": $SEED,
    "num_clients": $NUM_CLIENTS,
    "pass_at_k_mode": "$PASS_AT_K",
    "pairing_mode": "$PAIRING_MODE",
    "alpha_dirichlet": $ALPHA_DIRICHLET,
    "lr_local": $LR_LOCAL,
    "lr_kd": $LR_KD,
    "temperature": $TEMPERATURE,
    "batch_size_local": $BATCH_LOCAL,
    "batch_size_kd": $BATCH_KD,
    "local_pretrain_rounds": $LOCAL_PRETRAIN_ROUNDS,
    "eval_frequency": $EVAL_FREQUENCY,
    "model_config": "${MODEL_CONFIG:-default}",
    "save_directory": "$SAVE_DIR",
    "cpm_endpoint": "$CPM_ENDPOINT",
    "quick_mode": $QUICK_MODE,
    "verbose": $VERBOSE,
    "timestamp": "$TIMESTAMP",
    "start_time": "$(date -Iseconds)",
    "command_line": "$0 $*"
}
EOF

echo "💾 Configuration saved to: $SAVE_DIR/run_config.json"

# Set GPU device if specified
if [ ! -z "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Run the experiment
START_TIME=$(date +%s)
"$PYTHON_BIN" src/main_p2p_real.py \
    --rounds $ROUNDS \
    --seed $SEED \
    --save-dir "$SAVE_DIR" \
    --clients $NUM_CLIENTS \
    --local-pretrain-rounds $LOCAL_PRETRAIN_ROUNDS \
    --eval-frequency $EVAL_FREQUENCY \
    --pairing-mode $PAIRING_MODE \
    $ADDITIONAL_ARGS \
    2>&1 | tee "experimental_artifacts/knexa_fl/knexa_fl_${TIMESTAMP}.log"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Update configuration with completion info
cat > "$SAVE_DIR/run_completion.json" << EOF
{
    "end_time": "$(date -Iseconds)",
    "duration_seconds": $DURATION,
    "status": "completed",
    "log_file": "experimental_artifacts/knexa_fl/knexa_fl_${TIMESTAMP}.log"
}
EOF

# echo ""
# echo "🏁 KNEXA-FL Experiment Completed!"
# echo "============================================================"
# echo "⏰ End Time: $(date)"
# echo "⏱️  Duration: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
# echo "📝 Experiment Log: experimental_artifacts/knexa_fl/knexa_fl_${TIMESTAMP}.log"
# echo "🔬 Experiment Results: experimental_artifacts/knexa_fl/results/runs/[EXPERIMENT_ID]/"
# echo "📊 Structured Data: experimental_artifacts/knexa_fl/results/runs/[EXPERIMENT_ID]/metrics/"
# echo "💾 Model Checkpoints: experimental_artifacts/knexa_fl/results/runs/[EXPERIMENT_ID]/checkpoints/"
# echo "📋 Code Generation: experimental_artifacts/knexa_fl/results/runs/[EXPERIMENT_ID]/code_generation/"
# echo "⚙️  Run Configuration: $SAVE_DIR/run_config.json (legacy compatibility)"
# echo "============================================================"
# echo "✅ Federated learning executed successfully!"
# echo "🔬 All performance improvements from knowledge transfer"
# echo "🎯 Model parameters updated via gradient descent"
# echo "📈 Results suitable for publication"
# echo "📊 Advanced experiment management active"
# echo ""
# echo "🔍 To analyze experiment results:"
# echo "   python experiment_tools.py list"
# echo "   python experiment_tools.py show [EXPERIMENT_ID]"
# echo "   python experiment_tools.py report [EXPERIMENT_ID]"
# echo "   python experiment_tools.py best --metric pass@1"
# echo "   python experiment_tools.py compare [EXP1] [EXP2]"
# echo "   python experiment_tools.py export --output-file results.csv"

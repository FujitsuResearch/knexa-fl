#!/bin/bash

# Random P2P Baseline using KNEXA-FL Codebase
# This script runs the KNEXA-FL implementation with random pairing strategy
# Everything else remains identical for fair comparison

set -e

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Random P2P baseline using KNEXA-FL implementation with random pairing"
    echo ""
    echo "OPTIONS:"
    echo "  -r, --rounds NUM          Number of federated rounds (default: 25)"
    echo "  -s, --seed NUM            Random seed for reproducibility (default: 42)"
    echo "  -n, --name NAME           Experiment name (default: auto-generated)"
    echo "  -M, --model-config CONFIG Model configuration preset (default: max_vram_cached)"
    echo "  -c, --clients NUM         Number of clients (default: 6)"
    echo "  -p, --pass-at-k MODE      Pass@k evaluation mode: always|strategic|never (default: strategic)"
    echo "  -a, --alpha FLOAT         Dirichlet alpha for non-IID data (default: 0.1)"
    echo "  -l, --lr-local FLOAT      Local learning rate (default: 3e-5)"
    echo "  -k, --lr-kd FLOAT         Knowledge distillation learning rate (default: 5e-5)"
    echo "  -t, --temperature FLOAT   KD temperature (default: 1.5)"
    echo "  -d, --save-dir PATH       Save directory (default: auto-generated)"
    echo "  -b, --batch-local NUM     Local batch size (default: 8)"
    echo "  -B, --batch-kd NUM        KD batch size (default: 8)"
    echo "  -L, --local-pretrain-rounds NUM  Initial local-only training rounds (default: 0)"
    echo "  -g, --gpu NUM             GPU device to use (0 or 1, default: system default)"
    echo "  -f, --eval-frequency NUM  Evaluation frequency: full pass@k every N rounds (default: 1)"
    echo "  -v, --verbose             Verbose logging"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Run with all defaults"
    echo "  $0 -r 50 -s 123 -n \"My_Experiment\"   # Custom rounds, seed, and name"
    echo "  $0 -r 100 -f 5                        # 100 rounds, evaluate every 5th round"
    echo ""
    exit 0
}

# Default values (matching KNEXA-FL exactly)
ROUNDS=25
SEED=42
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME=""
NUM_CLIENTS=6
PASS_AT_K="strategic"
ALPHA_DIRICHLET=0.1
LR_LOCAL=3e-5  # Match KNEXA-FL
LR_KD=5e-5     # Match KNEXA-FL
TEMPERATURE=1.5  # Match KNEXA-FL
SAVE_DIR=""
BATCH_LOCAL=8  # Match KNEXA-FL
BATCH_KD=8     # Match KNEXA-FL
VERBOSE=false
LOCAL_PRETRAIN_ROUNDS=0
GPU_ID=""
MODEL_CONFIG=""  # Default to max_vram_cached
EVAL_FREQUENCY=1

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
        -M|--model-config)
            MODEL_CONFIG="$2"
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
        -b|--batch-local)
            BATCH_LOCAL="$2"
            shift 2
            ;;
        -B|--batch-kd)
            BATCH_KD="$2"
            shift 2
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

# Set default experiment name if not provided
if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="Random_P2P_${TIMESTAMP}"
fi

# Set default model config based on number of clients
if [ -z "$MODEL_CONFIG" ]; then
    if [ "$NUM_CLIENTS" -le 4 ]; then
        MODEL_CONFIG="max_vram_cached"
    else
        MODEL_CONFIG="max_performance_8"
    fi
fi

# Setup save directory - Use baselines directory for random P2P
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="experimental_artifacts/baselines/checkpoints"
fi

# Setup environment
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false

# Critical memory optimization settings (matching KNEXA-FL)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_LAUNCH_BLOCKING=0

# Set GPU device if specified
if [ ! -z "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Configure globals.py parameters via environment variables
export KNEXA_NUM_CLIENTS=$NUM_CLIENTS
export KNEXA_BATCH_LOCAL=$BATCH_LOCAL
export KNEXA_BATCH_KD=$BATCH_KD
export KNEXA_LR_LOCAL=$LR_LOCAL
export KNEXA_LR_KD=$LR_KD
export KNEXA_TEMP_DEFAULT=$TEMPERATURE
export KNEXA_ALPHA_DIRICHLET=$ALPHA_DIRICHLET
export KNEXA_EVAL_FULL_EVERY_N_ROUNDS=$EVAL_FREQUENCY

# Pass@k evaluation mode is handled via globals module
# No need to modify source files at runtime

# Display configuration
echo "============================================================"
echo "🎲 Random P2P Baseline (KNEXA-FL Implementation)"
echo "============================================================"
echo "📊 Configuration:"
echo "   Experiment Name: $EXPERIMENT_NAME"
echo "   Method: Random P2P (using KNEXA-FL codebase)"
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
echo "   Model Config: $MODEL_CONFIG"
echo "   GPU Device: ${GPU_ID:-system default}"
echo "   Eval Frequency: $EVAL_FREQUENCY (full pass@k every $EVAL_FREQUENCY rounds)"
echo "   Save Directory: $SAVE_DIR"
echo "   Verbose: $VERBOSE"
echo "   Timestamp: $TIMESTAMP"
echo "============================================================"
echo "🔬 Using KNEXA-FL implementation with:"
echo "   ✅ Pass@K Code Generation Metrics"
echo "   ✅ CodeBLEU Score Evaluation"
echo "   ✅ Knowledge Distillation"
echo "   ✅ Non-IID Data Distribution"
echo "   ✅ Comprehensive Loss Tracking"
echo "   ✅ Model Checkpointing"
echo "   ✅ RANDOM Peer-to-Peer Pairing (not LinUCB)"
echo "============================================================"

echo ""
echo "🎭 Starting Random P2P Baseline Experiment..."
echo "⏰ Start Time: $(date)"
echo ""

# Build command arguments
ARGS=""
ARGS="$ARGS --rounds $ROUNDS"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --save-dir $SAVE_DIR"
ARGS="$ARGS --clients $NUM_CLIENTS"
ARGS="$ARGS --pairing-strategy random"  # KEY DIFFERENCE: Use random pairing
ARGS="$ARGS --eval-frequency $EVAL_FREQUENCY"

if [ "$LOCAL_PRETRAIN_ROUNDS" -gt 0 ]; then
    ARGS="$ARGS --local-pretrain-rounds $LOCAL_PRETRAIN_ROUNDS"
fi

if [ ! -z "$MODEL_CONFIG" ]; then
    ARGS="$ARGS --model-config $MODEL_CONFIG"
fi

# Run the experiment using the flexible main_p2p_flex.py
START_TIME=$(date +%s)
python src/main_p2p_flex.py $ARGS
EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "🏁 Random P2P Baseline Experiment Completed!"
echo "============================================================"
echo "⏰ End Time: $(date)"
echo "⏱️  Duration: ${DURATION}s ($(($DURATION / 60))m $(($DURATION % 60))s)"
echo "💾 Results saved to: $SAVE_DIR"
echo "============================================================"

# Display comparison instructions
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "📊 To compare with KNEXA-FL (LinUCB), run:"
    echo "   python scripts/compare_results.py --baseline random --method linucb"
    echo ""
    echo "📈 To view results:"
    echo "   Check the experiment artifacts in: experimental_artifacts/baselines/"
fi

echo ""
echo "✅ Random P2P baseline experiment complete!"
echo "🔬 Results ready for comparison with KNEXA-FL (LinUCB)"

exit $EXIT_CODE
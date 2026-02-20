#!/bin/bash

# FedID-style Centralized KD Baseline using KNEXA-FL Codebase
# Runs the FedID-style server interactive distillation baseline on the main 6-client setup.

set -e

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "FedID-style Central KD baseline using KNEXA-FL implementation"
    echo ""
    echo "OPTIONS:"
    echo "  -r, --rounds NUM          Number of federated rounds (default: 20)"
    echo "  -s, --seed NUM            Random seed (default: 42)"
    echo "  -n, --name NAME           Experiment name (default: auto-generated)"
    echo "  -M, --model-config CONFIG Model configuration preset (default: max_performance_8)"
    echo "  -c, --clients NUM         Number of clients (default: 6)"
    echo "  -p, --pass-at-k MODE      Pass@k evaluation mode: always|strategic|never (default: strategic)"
    echo "  -a, --alpha FLOAT         Dirichlet alpha for non-IID data (default: 0.1)"
    echo "  -l, --lr-local FLOAT      Local learning rate (default: 3e-5)"
    echo "  -k, --lr-kd FLOAT         Knowledge distillation learning rate (default: 5e-5)"
    echo "  -t, --temperature FLOAT   KD temperature (default: 1.5)"
    echo "  -d, --save-dir PATH       Save directory (default: experimental_artifacts/baselines/checkpoints)"
    echo "  -L, --local-pretrain-rounds NUM  Initial local-only training rounds (default: 0)"
    echo "  -g, --gpu NUM             GPU device to use (default: system default)"
    echo "  -f, --eval-frequency NUM  Evaluation frequency: full pass@k every N rounds (default: 1)"
    echo "  -h, --help                Show this help message"
    exit 0
}

# Defaults per rebuttal spec
ROUNDS=20
SEED=42
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME=""
NUM_CLIENTS=6
PASS_AT_K="strategic"
ALPHA_DIRICHLET=0.1
LR_LOCAL=3e-5
LR_KD=5e-5
TEMPERATURE=1.5
SAVE_DIR=""
LOCAL_PRETRAIN_ROUNDS=0
GPU_ID=""
MODEL_CONFIG="max_performance_8"
EVAL_FREQUENCY=1

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--rounds) ROUNDS="$2"; shift 2 ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        -n|--name) EXPERIMENT_NAME="$2"; shift 2 ;;
        -M|--model-config) MODEL_CONFIG="$2"; shift 2 ;;
        -c|--clients) NUM_CLIENTS="$2"; shift 2 ;;
        -p|--pass-at-k) PASS_AT_K="$2"; shift 2 ;;
        -a|--alpha) ALPHA_DIRICHLET="$2"; shift 2 ;;
        -l|--lr-local) LR_LOCAL="$2"; shift 2 ;;
        -k|--lr-kd) LR_KD="$2"; shift 2 ;;
        -t|--temperature) TEMPERATURE="$2"; shift 2 ;;
        -d|--save-dir) SAVE_DIR="$2"; shift 2 ;;
        -L|--local-pretrain-rounds) LOCAL_PRETRAIN_ROUNDS="$2"; shift 2 ;;
        -g|--gpu) GPU_ID="$2"; shift 2 ;;
        -f|--eval-frequency) EVAL_FREQUENCY="$2"; shift 2 ;;
        -h|--help) show_usage ;;
        *) echo "Unknown option: $1"; show_usage ;;
    esac
done

if [ -z "$EXPERIMENT_NAME" ]; then
    EXPERIMENT_NAME="FedID_CentralKD_${TIMESTAMP}"
fi

if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="experimental_artifacts/baselines/checkpoints"
fi

# Environment
export PYTHONPATH=$(pwd)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_LAUNCH_BLOCKING=0

if [ ! -z "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Globals overrides
export KNEXA_NUM_CLIENTS=$NUM_CLIENTS
export KNEXA_LR_LOCAL=$LR_LOCAL
export KNEXA_LR_KD=$LR_KD
export KNEXA_TEMP_DEFAULT=$TEMPERATURE
export KNEXA_ALPHA_DIRICHLET=$ALPHA_DIRICHLET
export KNEXA_EVAL_FULL_EVERY_N_ROUNDS=$EVAL_FREQUENCY

echo "============================================================"
echo "🧠 FedID-Style Central KD Baseline (KNEXA-FL)"
echo "============================================================"
echo "   Rounds: $ROUNDS"
echo "   Seed: $SEED"
echo "   Clients: $NUM_CLIENTS"
echo "   Model Config: $MODEL_CONFIG"
echo "   LR Local: $LR_LOCAL"
echo "   LR KD: $LR_KD"
echo "   Temperature: $TEMPERATURE"
echo "   Eval Frequency: $EVAL_FREQUENCY"
echo "   Save Directory: $SAVE_DIR"
echo "============================================================"

ARGS=""
ARGS="$ARGS --rounds $ROUNDS"
ARGS="$ARGS --seed $SEED"
ARGS="$ARGS --save-dir $SAVE_DIR"
ARGS="$ARGS --clients $NUM_CLIENTS"
ARGS="$ARGS --pairing-strategy fedid_kd"
ARGS="$ARGS --eval-frequency $EVAL_FREQUENCY"
ARGS="$ARGS --model-config $MODEL_CONFIG"

START_TIME=$(date +%s)
python src/main_p2p_flex.py $ARGS
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "🏁 FedID Central KD Baseline Completed"
echo "⏱️  Duration: ${DURATION}s"
echo "💾 Results saved to: $SAVE_DIR"

exit $EXIT_CODE


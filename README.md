# KNEXA-FL

**Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation**

Inderjeet Singh, Eleonore Vissol-Gaudin, Andikan Otung, Motoyoshi Sekiya

Fujitsu Research of Europe, Slough, United Kingdom

Published at **AAAI 2026** Main Track

Paper: [KNEXA_FL_Main.pdf](KNEXA_FL_Main.pdf) | Supplementary: [KNEXA_FL_Supplementary_Material.pdf](KNEXA_FL_Supplementary_Material.pdf)

---

## Overview

KNEXA-FL is a hybrid framework for orchestrated decentralization in federated LLM fine-tuning. It resolves the trade-off between (a) the security vulnerabilities of centralized aggregation and (b) the statistical inefficiency of random peer-to-peer pairing. The framework introduces a non-aggregating **Central Profiler/Matchmaker (CPM)** that formulates P2P collaboration as a contextual bandit problem, using a **LinUCB** algorithm on abstract agent profiles to learn an optimal matchmaking policy. Actual knowledge transfer occurs directly between matched peers via secure text-based distillation, without the CPM ever accessing the models.

Key results on a heterogeneous code generation task (6 agents, 410M-620M parameters):
- **+50% relative improvement** in Pass@1 over random P2P collaboration
- **Stable convergence**, in contrast to centralized distillation baselines that suffer catastrophic performance collapse
- **Sub-linear regret** of the LinUCB matchmaker relative to the oracle pairing

## Repository Structure

```
knexa-fl/
|
|-- knexa_fl/                        # Core KNEXA-FL framework (Paper Sec. 3, Algorithms 1-2)
|   |-- cpm/                         # Central Profiler/Matchmaker
|   |   |-- orchestrator.py          #   CPM orchestration logic (Sec. 3.3)
|   |   |-- linucb.py                #   LinUCB contextual bandit (Eq. 1, Sec. 3.3)
|   |   +-- privacy_profile.py       #   Privacy-preserving profile sanitization (Sec. 3.1)
|   |-- agents/                      # LLM Agent components
|   |   |-- agent.py                 #   Agent with local PEFT training (Sec. 3.2)
|   |   |-- privacy_guardrail.py     #   Guardrail filter and SIER (Sec. 3.2)
|   |   +-- lora_config.py           #   Adaptive LoRA configuration (Supp. Table 1)
|   |-- p2p/                         # Peer-to-peer knowledge exchange
|   |   +-- knowledge_exchange.py    #   Adaptive Knowledge Distillation (Eq. 2-3, Sec. 3.2)
|   +-- main_example.py              # End-to-end example orchestrating the full protocol
|
|-- src/                             # Experiment source code
|   |-- main_p2p_real.py             # Main experiment driver (KNEXA-FL with LinUCB)
|   |-- main_p2p_flex.py             # Flexible driver for baselines (random, central_kd, heuristic)
|   |-- client.py                    # Full FL client (local training, KD, evaluation)
|   |-- bandit.py                    # Production LinUCB implementation
|   |-- code_evaluation.py           # Pass@k (with code execution) and CodeBLEU evaluation
|   |-- model_utils.py               # Model loading with per-architecture LoRA (Table 3)
|   |-- data_utils.py                # Dataset loading and Dirichlet non-IID partitioning (Sec. 4.1)
|   |-- globals.py                   # All hyperparameters (Supp. Table 1)
|   |-- experiment_manager.py        # Experiment lifecycle, checkpointing, logging
|   |-- grpc_p2p/                    # gRPC-based P2P communication layer
|   |   |-- knowledge_distillation.py#   Text-based Adaptive KD (Eq. 2-3)
|   |   |-- cpm_service.py           #   CPM gRPC service
|   |   |-- direct_p2p.py            #   Direct P2P exchange
|   |   |-- privacy_profile.py       #   Profile construction and sanitization
|   |   |-- transfer_set.py          #   Public transfer set management
|   |   |-- knexa_p2p.proto          #   Protocol Buffer definitions
|   |   +-- ...
|   +-- ...                          # Additional utilities (logging, metrics, reporting)
|
|-- baselines/                       # Baseline implementations (Sec. 4.1)
|   |-- local_only.py                # LocalOnly (no collaboration)
|   |-- random_pair.py               # Random-P2P
|   |-- fedid_central_kd.py          # FedID-CentralKD (centralized distillation)
|   |-- fedavg.py                    # FedAvg (traditional FL)
|   |-- fedskd.py                    # FedSKD
|   |-- no_trust.py                  # Ablation: no trust scoring
|   +-- no_privacy.py                # Ablation: no privacy guardrails
|
|-- simulations/                     # LinUCB CPM validation suite (Fig. 2, Supp. Fig. 1)
|   +-- linucb_cpm_validation/
|       |-- bandit_engines/          # Multiple bandit implementations
|       |   |-- linucb_basic.py      #   Standard LinUCB
|       |   |-- linucb_enhanced.py   #   Enhanced LinUCB with decay
|       |   |-- oracle_engine.py     #   Oracle (upper bound)
|       |   |-- random_baseline.py   #   Random (lower bound)
|       |   +-- heterogeneity_greedy.py # Heuristic baseline
|       |-- synthetic_environment.py # Synthetic federation environment
|       |-- profile_builders.py      # Agent profile generation
|       |-- reward_models.py         # Reward signal models
|       +-- comprehensive_evaluation_protocol.py
|
|-- knexa_fl_release/                # Minimal reproduction package (artifact-based)
|   |-- cpm_linucb.py               # Compact LinUCB (numpy-only)
|   |-- simulate_cpm.py             # Synthetic CPM simulation driver
|   |-- reproduce_paper.py          # Recompute Table 4 from stored artifacts
|   +-- split_serializer.py         # Deterministic split generator
|
|-- artifacts/                       # Stored experimental artifacts
|   |-- baselines/summaries/         # Baseline result summaries (6-client config)
|   |-- knexa_fl/logs/               # KNEXA-FL run logs
|   |-- roster/client_roster.json    # Federation roster
|   +-- data/DATA_MANIFEST.md        # Data provenance documentation
|
|-- configs/                         # Configuration files
|   |-- default_config.yaml          # Default framework configuration
|   +-- example_run.yaml             # Example 6-agent experiment config
|
|-- scripts/                         # Experiment launch scripts
|   |-- run_knexa_fl.sh              # Run KNEXA-FL (LinUCB matchmaking)
|   |-- run_random_p2p.sh            # Run Random-P2P baseline
|   |-- run_central_kd.sh            # Run Central-KD baseline
|   |-- run_synthetic_cpm.sh         # Run CPM simulation
|   |-- reproduce.sh                 # Reproduce metrics from artifacts
|   +-- make_splits.sh               # Generate federation data splits
|
|-- tests/                           # Test suite
|-- techforum_code/                  # Live guardrail demonstration
+-- requirements.txt                 # Python dependencies
```

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce Paper Metrics from Artifacts (CPU-only)

Recompute the aggregate metrics in Table 4 directly from stored artifacts:

```bash
bash scripts/reproduce.sh
```

### 3. Run CPM Simulation (CPU-only)

Reproduce the LinUCB learning dynamics study (Fig. 2):

```bash
bash scripts/run_synthetic_cpm.sh
```

### 4. Run Full Experiments (GPU required)

The full experiments require GPU hardware (A100/H100-class recommended) and will download models from HuggingFace.

**KNEXA-FL (LinUCB matchmaking):**
```bash
export PYTHONPATH=$(pwd)
bash scripts/run_knexa_fl.sh --rounds 25 --seed 42 --clients 6
```

**Baselines:**
```bash
# Random P2P
bash scripts/run_random_p2p.sh --rounds 25 --seed 42 --clients 6

# Centralized KD
bash scripts/run_central_kd.sh --rounds 25 --seed 42 --clients 6

# Local-Only (via Flower simulation)
python baselines/local_only.py
```

### 5. Use the Framework Programmatically

```python
from knexa_fl.cpm import CPMOrchestrator, LinUCB
from knexa_fl.agents import KnexaAgent, AgentConfig, GuardrailFilter
from knexa_fl.p2p import AdaptiveKnowledgeDistillation, KDConfig

# See knexa_fl/main_example.py for a complete end-to-end example
```

## Federation Configuration

The default 6-client configuration uses heterogeneous LLM backbones:

| Client | Backbone | Parameters | Architecture |
|--------|----------|------------|--------------|
| C0 | Qwen-0.5B | 620M | Qwen |
| C1 | Cerebras-GPT-590M | 590M | Cerebras |
| C2 | BLOOM-560M | 560M | BLOOM |
| C3 | Pythia-410M | 410M | Pythia (GPT-NeoX) |
| C4 | Qwen-0.5B | 620M | Qwen (duplicate backbone) |
| C5 | Cerebras-GPT-590M | 590M | Cerebras (duplicate backbone) |

Duplicate backbones are treated as independent agents with distinct LoRA adapters, seeds, and disjoint non-IID data partitions (Dirichlet alpha = 0.1).

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | 8 | Adaptive per architecture |
| LoRA alpha | 32 | 4x rank |
| Learning rate (local) | 3e-5 | Local PEFT training |
| Learning rate (KD) | 5e-5 | Knowledge distillation |
| KD temperature | 1.5 | Soft target temperature |
| LinUCB lambda | 0.01 | Regularization |
| LinUCB beta0 | 1.0 | Exploration coefficient |
| DP clip norm | 1.0 | Gradient clipping |
| Dirichlet alpha | 0.1 | Non-IID partitioning |

See `src/globals.py` for the complete parameter set and `configs/default_config.yaml` for YAML-based configuration.

## Reproducibility

- **Determinism**: All experiments use fixed seeds (default: `--seed 42`).
- **Environment**: Python >= 3.10. CPM simulation is CPU-only. Full experiments require CUDA-capable GPUs.
- **Datasets**: HumanEval and MBPP, loaded via HuggingFace `datasets`. No manual download required.

## Tests

```bash
pytest tests/ -v
```

## Citation

If you use this code or build on our work, please cite:

```bibtex
@inproceedings{singh2026knexa,
  title={Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation},
  author={Singh, Inderjeet and Vissol-Gaudin, Eleonore and Otung, Andikan and Sekiya, Motoyoshi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Maintainer: Inderjeet Singh (corresponding author) - inderjeet.singh@fujitsu.com

Fujitsu Research of Europe, Slough, United Kingdom

# KNEXA‑FL

This directory provides reference code and artifacts used to reproduce the core empirical results from our paper:

- Title: “Learning to Collaborate: An Orchestrated‑Decentralized Framework for Peer‑to‑Peer LLM Federation”
- Venue: AAAI‑2026 (camera‑ready)

Active maintenance and citation
- Maintainer: Inderjeet Singh (corresponding author). This release will be updated as we finalize ablations and additional analyses; minor version bumps may appear as new experiments are integrated.
- Citation: If you build on this code or artifacts, please cite the paper as indicated in the camera‑ready version.

Scope and design
- Focused and self‑contained: this package is organized around components that are sufficient to recompute the main tables and curves reported in the paper under realistic resource constraints.
- Deterministic: seeds and synthetic environments are fixed for repeatability.
- Two complementary tracks:
  1) CPM simulation (LinUCB) — reproduces the learning‑dynamics study, including learning curves and sub‑linear regret relative to a random baseline.
  2) Artifact‑based summary — recomputes the aggregate metrics (Pass@k, CodeBLEU) from stored artifacts. When required artifacts are not present, the code reports which components are missing so that they can be produced from the corresponding experimental runs.

Contents
- `knexa_fl_release/`
  - `cpm_linucb.py`: Compact LinUCB‑based CPM used in the synthetic simulation.
  - `simulate_cpm.py`: Driver for the synthetic federation; produces CSV files for learning and regret curves.
  - `reproduce_paper.py`: Recomputes the final summary table (Pass@k, CodeBLEU) directly from artifacts.
- `artifacts/`
  - `baselines/summaries/*.json`: Baseline summaries for LocalOnly, Random‑P2P, Central‑KD, FedID‑CentralKD, and Heuristic‑P2P (six‑client configuration).
  - `knexa_fl/logs/*.log`: KNEXA‑FL log with final aggregated metrics for the six‑client configuration.
  - `roster/client_roster.json`: Six‑client federation roster, including repeated backbones.
- `scripts/`
  - `run_synthetic_cpm.sh`: One‑command entry point for the CPM simulation.
  - `reproduce.sh`: One‑command entry point to recompute the reported metrics from artifacts.
  - `make_splits.sh`: Deterministic split generator (requires local HumanEval and MBPP JSONL files).
- `requirements.txt`: Minimal dependencies required to run the release scripts.

Quick start
- Create a virtual environment and install requirements:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r knexa-fl-release/requirements.txt`
- Run the CPM simulation:
  - `bash knexa-fl-release/scripts/run_synthetic_cpm.sh`
  - Outputs to `knexa-fl-release/results/simulation/`:
    - `learning_curve.csv` (mean Pass@1 vs. rounds)
    - `regret_curve.csv` (cumulative regret vs. rounds)
- Reproduce the paper’s final metrics from artifacts:
  - `bash knexa-fl-release/scripts/reproduce.sh`
  - The script parses the packaged artifacts in this folder to recompute the aggregate table.
  - If any required artifacts are missing, it exits with a clear message describing which pieces are absent and indicating that they must be obtained from the corresponding experimental runs.

Full real‑model runs
- The six‑client heterogeneous federation experiments rely on parameter‑efficient fine‑tuning over 410M–604M‑parameter backbones, curated HumanEval and MBPP splits, and accelerator hardware such as A100/H100‑class GPUs.
- The underlying training and orchestration pipelines follow standard practice for large‑scale experimentation and are described at a high level in the paper.
- This release is centered on elements that are directly useful for empirical verification:
  - A CPM‑based simulation that mirrors the learning‑dynamics study.
  - An artifact‑based recomputation of the aggregate metrics, using logs and summaries obtained from representative runs.

Federation configuration (six clients)
- The roster file (`artifacts/roster/client_roster.json`) specifies six clients:
  - C0 Qwen‑0.5B, C1 Cerebras‑590M, C2 BLOOM‑560M, C3 Pythia‑410M, C4 Qwen‑0.5B (duplicate of C0), C5 Cerebras‑590M (duplicate of C1).
- Repeated backbones are treated as independent endpoints: they use distinct random seeds, separate LoRA adapters, and disjoint non‑IID data partitions (Dirichlet α = 0.1). This configuration is intended to reflect realistic deployments with repeated backbones while preserving heterogeneity across clients.

Reported values and artifacts
- This release does not print fixed “camera‑ready” values unless they are computed directly from the artifacts contained here.
- `reproduce.sh` parses the artifacts in this directory tree. If the artifact set is incomplete, the code explains which components are missing and indicates that they must come from upstream experimental runs.

Reproducibility notes
- Determinism: primary experiments use fixed seeds (e.g., `--seed 42`).
- Environment: Python ≥ 3.10; the CPM simulation is CPU‑only, and GPUs are not required to run this release.
- No external network access is needed: the simulation uses synthetic data, and artifact parsing operates on local files.

Generating federation splits (optional)
- To regenerate the federation splits, provide local dataset files (JSONL) for HumanEval and MBPP.
- Run:
  - `bash knexa-fl-release/scripts/make_splits.sh /path/to/HumanEval.jsonl /path/to/MBPP.jsonl`
  - The script writes JSON lists to `artifacts/data/splits/`: `client_Ci_train.json`, `client_Ci_val.json`, and `global_test.json`.
- The resulting counts follow `artifacts/roster/client_roster.json` and the non‑IID Dirichlet α = 0.1 policy. The script fails clearly if the input pool is insufficient.

Citation
- If you use this release, please cite the AAAI‑2026 paper as specified in the camera‑ready manuscript.

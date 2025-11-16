Data Manifest (HumanEval + MBPP)

Purpose
- This release is independent and prints metrics computed from packaged artifacts. Raw datasets are not redistributed here; please use the original dataset sources for licensing and access details.
- This manifest documents dataset sources, licenses, and verified retrieval instructions for internal replication on compatible hardware.

Datasets
- HumanEval
  - Source: OpenAI HumanEval benchmark
  - License: Refer to the original project’s license
  - Typical access: via public mirrors or references included in many evaluation toolkits
- MBPP (Mostly Basic Programming Problems)
  - Source: Google Research
  - License: Refer to the original project’s license
  - Typical access: via Hugging Face datasets `mbpp` (community maintained variants exist)

Federation Splits
- Non‑IID partitions are sampled via Dirichlet with α=0.1.
- Counts per client (train/val) are reflected in `artifacts/roster/client_roster.json`.
- For deterministic replication:
  - Fix `seed=42` and persist the sampled indices.
  - Assign per‑client train/val using the roster counts.

Verification
- Upon fetching datasets, we recommend:
  1) Verify total counts match those used in the paper (HumanEval+MBPP combined training, 116‑problem global test set).
  2) Confirm non‑IID splits across six clients follow the referenced counts.
  3) Log the final aggregate metrics and compare against this package’s computed values, allowing minor differences depending on environment.

Rationale
- Raw datasets are not redistributed in this package; please refer to the original dataset sources for licensing and access details.
- The artifact summaries provided here are sufficient to verify the reported metrics without embedding the full datasets.

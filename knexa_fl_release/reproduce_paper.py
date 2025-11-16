#!/usr/bin/env python3
"""Reproduce the paper’s final aggregate metrics from artifacts.

Behavior:
- Parse packaged logs/JSON summaries to recompute the table.
- If required artifacts are missing, exit with an error and
  point to the scripts used to regenerate them in the full repo.
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ReportRow:
    pass1: float
    pass5: float
    pass10: float
    codebleu: float


# No hard-coded reported values; we only print values computed from artifacts.


def _baseline_summary_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts" / "baselines" / "summaries"


def _load_baseline_summary(method_name: str, *, min_clients: int = 6) -> Optional[ReportRow]:
    base = _baseline_summary_dir()
    if not base.is_dir():
        return None
    for name in sorted(os.listdir(base)):
        if not name.endswith('.json'):
            continue
        path = base / name
        try:
            with open(path, "r") as f:
                obj = json.load(f)
        except Exception:
            continue
        num_clients = obj.get("num_clients")
        method = obj.get("method")
        pass_section = obj.get("pass_at_k_metrics", {})
        p1 = pass_section.get("pass@1", {}).get("final")
        p5 = pass_section.get("pass@5", {}).get("final")
        p10 = pass_section.get("pass@10", {}).get("final")
        codebleu = obj.get("final_metrics", {}).get("codebleu")
        if not (
            method == method_name and
            num_clients is not None and int(num_clients) >= min_clients and
            p1 is not None and p5 is not None and p10 is not None and codebleu is not None
        ):
            continue
        return ReportRow(pass1=float(p1), pass5=float(p5), pass10=float(p10), codebleu=float(codebleu))
    return None


def _try_knexa_fl_summary() -> Optional[ReportRow]:
    # Parse from KNEXA‑FL logs: look for FINAL_PASS_AT_K lines
    release_base = Path(__file__).resolve().parents[1] / "artifacts" / "knexa_fl" / "logs"
    base = str(release_base)
    if not os.path.isdir(base):
        return None
    out: Optional[ReportRow] = None
    for name in sorted(os.listdir(base)):
        if not name.endswith(".log"):
            continue
        path = os.path.join(base, name)
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    if "FINAL_PASS_AT_K completed" in line and "average_pass_at_k" in line:
                        # Example snippet:
                        # average_pass_at_k={'pass@1': 0.0999, 'pass@5': 0.3108, 'pass@10': 0.4166, 'codebleu': 0.3447}
                        t = line[line.find("average_pass_at_k=") + len("average_pass_at_k=") :]
                        t = t.strip()
                        # Trim trailing characters after the closing '}'
                        end = t.rfind('}')
                        if end != -1:
                            t = t[: end + 1]
                        try:
                            import ast
                            kv = ast.literal_eval(t)  # safe parse for Python dict literal
                            # Prefer logs with clients_evaluated >= 6
                            ce_pos = line.find("clients_evaluated=")
                            ce_val = None
                            if ce_pos != -1:
                                ce_str = line[ce_pos + len("clients_evaluated=") :].split(",")[0]
                                try:
                                    ce_val = int(ce_str)
                                except Exception:
                                    ce_val = None
                            row = ReportRow(
                                pass1=float(kv.get("pass@1", 0.0)),
                                pass5=float(kv.get("pass@5", 0.0)),
                                pass10=float(kv.get("pass@10", 0.0)),
                                codebleu=float(kv.get("codebleu", 0.0)),
                            )
                            if ce_val is None or ce_val >= 6:
                                out = row
                        except Exception:
                            pass
        except Exception:
            continue
    return out

def main():
    # Always compute from artifacts; fail if required artifacts are missing.
    table: Dict[str, ReportRow] = {}

    knx = _try_knexa_fl_summary()

    missing = []

    baseline_methods = [
        "LocalOnly",
        "FedID-CentralKD",
        "Central-KD",
        "Heuristic-P2P",
        "Random-P2P",
    ]

    for method in baseline_methods:
        row = _load_baseline_summary(method)
        if row:
            table[method] = row
        else:
            missing.append(f"{method} (baselines/summaries not found or no ≥6-client run)")

    if knx:
        table["KNEXA-FL"] = knx
    else:
        missing.append("KNEXA-FL (logs with FINAL_PASS_AT_K and clients_evaluated ≥6 not found)")

    if missing:
        print("Error: Required artifacts missing for:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease generate the corresponding artifacts using your experimental pipelines (e.g., by running KNEXA-FL, Random-P2P, Central-KD, FedID-CentralKD, and Heuristic-P2P evaluation jobs on the target hardware) and place the resulting summaries/logs under this release's artifacts directory.")
        raise SystemExit(1)

    print("Final average performance on the 116-problem global test set:")
    print("Method           Pass@1   Pass@5   Pass@10  CodeBLEU")
    for method in [
        "LocalOnly",
        "FedID-CentralKD",
        "Central-KD",
        "Heuristic-P2P",
        "Random-P2P",
        "KNEXA-FL",
    ]:
        row = table[method]
        print(f"{method:15} {100*row.pass1:6.2f}%  {100*row.pass5:6.2f}%  {100*row.pass10:7.2f}%  {row.codebleu:7.3f}")


if __name__ == "__main__":
    main()

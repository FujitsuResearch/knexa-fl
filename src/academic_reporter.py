#!/usr/bin/env python3
"""
Academic Results Reporter for KNEXA-FL
Minimal stub for compatibility
"""

import json
import logging
from pathlib import Path
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

def generate_academic_report(experiment_id: str, results_dir: Union[str, Path]) -> Path:
    """
    Generate comprehensive academic report for KNEXA-FL experiment
    
    Args:
        experiment_id: Unique experiment identifier
        results_dir: Directory containing experiment results
        
    Returns:
        Path to generated paper materials
    """
    results_dir = Path(results_dir)
    paper_dir = results_dir / "paper_materials"
    paper_dir.mkdir(exist_ok=True)
    
    # Create basic structure
    (paper_dir / "tables").mkdir(exist_ok=True)
    (paper_dir / "figures").mkdir(exist_ok=True)
    (paper_dir / "data").mkdir(exist_ok=True)
    
    # Create minimal summary file
    summary = {
        "experiment_id": experiment_id,
        "results_dir": str(results_dir),
        "generated": True,
        "status": "Academic report generation simplified - full LaTeX generation removed"
    }
    
    # Write summary
    with open(paper_dir / "MASTER_SUMMARY.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Write basic metrics if final_results.json exists
    results_file = results_dir / "final_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Save key metrics in CSV format
        if "metrics" in results:
            metrics_file = paper_dir / "data" / "key_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(results["metrics"], f, indent=2)
    
    logger.info(f"Academic report stub generated for {experiment_id} at {paper_dir}")
    return paper_dir

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python academic_reporter.py <experiment_id> <results_dir>")
        sys.exit(1)
    
    experiment_id = sys.argv[1]
    results_dir = sys.argv[2]
    
    paper_dir = generate_academic_report(experiment_id, results_dir)
    print(f"Academic report generated at: {paper_dir}")
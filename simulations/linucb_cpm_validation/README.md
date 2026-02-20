# LinUCB-Enhanced CPM Validation

This directory contains the validation suite for the LinUCB-based Central Profiler/Matchmaker (CPM) component of KNEXA-FL.

## Key Results
- **33.9% average improvement** over random baseline
- **48.5% maximum improvement** in high-heterogeneity scenarios
- Statistically significant results (p < 0.05) in 75% of configurations

## Main Documentation
See `LINUCB_CPM_COMPREHENSIVE_DOCUMENTATION.md` for complete technical details, implementation guide, and results.

## Quick Start
```bash
# Run validation test
python quick_validation_test.py

# Generate publication-ready figures
python generate_paper_results_fast.py

# Run comprehensive evaluation
python comprehensive_evaluation_protocol.py
```

## Directory Structure
- `bandit_engines/` - Core LinUCB algorithm implementations
- `final_paper_results/` - Publication-ready figures and results
- `results/` - Experimental outputs
- `trash/` - Deprecated files (to be removed)

## Key Files
- `LINUCB_CPM_COMPREHENSIVE_DOCUMENTATION.md` - Complete consolidated documentation
- `comprehensive_evaluation_protocol.py` - Main evaluation script
- `generate_paper_results_fast.py` - Figure generation for papers
- `create_learning_curves.py` - Learning dynamics visualization

---
*Author: Inderjeet Singh*
*Last Updated: August 2025*
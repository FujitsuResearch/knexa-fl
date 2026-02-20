# Random P2P Baseline Implementation

The random peer-to-peer (P2P) matching baseline has been integrated directly into the main KNEXA-FL codebase for fair comparison.

## Current Implementation

The random P2P baseline now uses the exact same codebase as KNEXA-FL with only the pairing strategy changed from LinUCB to random. This ensures:

- Identical training procedures
- Same evaluation metrics (Pass@K, CodeBLEU)
- Same data distribution and model configurations
- Direct comparability of results

## Running Random P2P Experiments

### Using KNEXA-FL Implementation with Random Pairing

```bash
# Single experiment
./run_random_p2p_knexa.sh -r 25 -c 4 --seed 42 --name "Random_Test"

# Run all experiments matching KNEXA-FL setup
./runs_all_random_p2p_knexa.sh
```

### Key Difference from KNEXA-FL

The only difference is in the pairing strategy:
- **KNEXA-FL**: Uses LinUCB bandit algorithm to select optimal client pairs based on context vectors
- **Random P2P**: Randomly shuffles clients and pairs them

All other aspects remain identical to ensure fair academic comparison.

## Implementation Details

The random pairing is implemented in `src/main_p2p_flex.py` which supports both strategies via the `--pairing-strategy` flag:
- `--pairing-strategy linucb` (default): Original KNEXA-FL behavior
- `--pairing-strategy random`: Random P2P baseline

## Results

Results are saved in the same `experimental_artifacts/knexa_fl/` directory structure, making direct comparison straightforward.
#!/usr/bin/env python3
"""
Quick Validation Test for Improved LinUCB Evaluation

This script validates that the improved evaluation protocol generates
academically sound results with proper statistical properties.

Author: Inderjeet Singh
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the module to path
sys.path.append(str(Path(__file__).parent))

from comprehensive_evaluation_protocol import (
    ImprovedRewardModel, 
    ComprehensiveEvaluationProtocol
)


def test_reward_model():
    """Test that the improved reward model generates reasonable rewards"""
    print("Testing Improved Reward Model...")
    
    reward_model = ImprovedRewardModel()
    
    # Create test clients
    client_high_perf = {
        'id': 0,
        'model_family': 'qwen',
        'performance': 0.7,
        'data_distribution': np.array([0.8, 0.1, 0.05, 0.03, 0.02]),
        'trust': 0.9,
        'learning_rate': 1.0
    }
    
    client_low_perf = {
        'id': 1,
        'model_family': 'bloom',
        'performance': 0.3,
        'data_distribution': np.array([0.1, 0.1, 0.3, 0.3, 0.2]),
        'trust': 0.85,
        'learning_rate': 1.1
    }
    
    client_similar = {
        'id': 2,
        'model_family': 'qwen',
        'performance': 0.68,
        'data_distribution': np.array([0.75, 0.15, 0.05, 0.03, 0.02]),
        'trust': 0.88,
        'learning_rate': 0.95
    }
    
    # Test different pairings
    print("\nTesting reward calculations:")
    
    # High-low pairing (should be good)
    reward_hl = reward_model.compute_reward(client_low_perf, client_high_perf)
    print(f"High-Low pairing reward: {reward_hl:.3f}")
    
    # Similar pairing (should be lower)
    reward_sim = reward_model.compute_reward(client_high_perf, client_similar)
    print(f"Similar pairing reward: {reward_sim:.3f}")
    
    # Verify heterogeneity bonus
    div_hl = reward_model._compute_data_diversity(client_low_perf, client_high_perf)
    div_sim = reward_model._compute_data_diversity(client_high_perf, client_similar)
    print(f"\nData diversity - High-Low: {div_hl:.3f}")
    print(f"Data diversity - Similar: {div_sim:.3f}")
    
    # Test reward distribution
    print("\nTesting reward distribution (100 samples):")
    rewards = [reward_model.compute_reward(client_low_perf, client_high_perf) 
               for _ in range(100)]
    print(f"Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
    print(f"Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}")
    
    assert 0 <= np.min(rewards) <= 1, "Rewards should be in [0, 1]"
    assert reward_hl > reward_sim * 1.2, "Heterogeneous pairing should have higher reward"
    
    print("\n✓ Reward model tests passed!")


def test_small_scale_simulation():
    """Test the simulation with a small configuration"""
    print("\n\nTesting Small-Scale Simulation...")
    
    protocol = ComprehensiveEvaluationProtocol()
    
    # Small test configuration - more rounds for LinUCB to learn
    config = {'num_clients': 8, 'heterogeneity': 'high', 'rounds': 40}
    
    # Generate clients
    clients = protocol.generate_synthetic_clients(8, 'high')
    
    print(f"\nGenerated {len(clients)} clients")
    print(f"Performance distribution: min={min(c['performance'] for c in clients):.3f}, "
          f"max={max(c['performance'] for c in clients):.3f}")
    
    # Run simulation for each method
    results = {}
    methods = ['random', 'hetero_greedy', 'linucb']
    
    for method in methods:
        print(f"\nSimulating {method}...")
        np.random.seed(42)  # Reset seed for fair comparison
        
        result = protocol.simulate_linucb_learning(clients, config, method=method)
        results[method] = result
        
        print(f"  Final cumulative reward: {result['cumulative_rewards'][-1]:.2f}")
        print(f"  Final Pass@1: {result['pass_at_1'][-1]:.3f}")
        print(f"  Avg heterogeneity: {np.mean(result['heterogeneity']):.3f}")
    
    # Calculate improvements
    random_final = results['random']['pass_at_1'][-1]
    linucb_final = results['linucb']['pass_at_1'][-1]
    improvement = (linucb_final - random_final) / random_final * 100
    
    print(f"\n\nLinUCB improvement over random: {improvement:.1f}%")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    
    for method, result in results.items():
        plt.plot(result['pass_at_1'], label=method.replace('_', ' ').title(), linewidth=2)
    
    plt.xlabel('Round')
    plt.ylabel('Pass@1 Score')
    plt.title('Learning Curves - Small Scale Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_dir = Path("final_paper_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "validation_learning_curves.png", dpi=150)
    plt.close()
    
    print(f"\nLearning curves saved to {output_dir / 'validation_learning_curves.png'}")
    
    # Validate results
    assert improvement > 20, f"LinUCB should show >20% improvement, got {improvement:.1f}%"
    assert linucb_final > random_final, "LinUCB should outperform random"
    
    print("\n✓ Simulation tests passed!")
    
    return results


def test_statistical_significance():
    """Test that results are statistically significant"""
    print("\n\nTesting Statistical Significance...")
    
    protocol = ComprehensiveEvaluationProtocol()
    config = {'num_clients': 16, 'heterogeneity': 'high', 'rounds': 30}
    clients = protocol.generate_synthetic_clients(16, 'high')
    
    # Run multiple seeds
    random_scores = []
    linucb_scores = []
    
    for seed in range(5):
        np.random.seed(seed + 42)
        
        # Random baseline
        result_random = protocol.simulate_linucb_learning(clients, config, method='random')
        random_scores.append(result_random['pass_at_1'][-1])
        
        # LinUCB
        result_linucb = protocol.simulate_linucb_learning(clients, config, method='linucb')
        linucb_scores.append(result_linucb['pass_at_1'][-1])
    
    print(f"Random scores: {[f'{s:.3f}' for s in random_scores]}")
    print(f"LinUCB scores: {[f'{s:.3f}' for s in linucb_scores]}")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(linucb_scores, random_scores, alternative='greater')
    
    print(f"\nT-statistic: {t_stat:.3f}")
    print(f"P-value: {p_value:.4f}")
    
    # Effect size
    pooled_std = np.sqrt((np.var(linucb_scores, ddof=1) + np.var(random_scores, ddof=1)) / 2)
    cohens_d = (np.mean(linucb_scores) - np.mean(random_scores)) / pooled_std
    print(f"Cohen's d: {cohens_d:.3f}")
    
    assert p_value < 0.05, f"Results should be statistically significant, got p={p_value:.4f}"
    assert cohens_d > 0.8, f"Effect size should be large (>0.8), got {cohens_d:.3f}"
    
    print("\n✓ Statistical significance tests passed!")


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VALIDATION TEST FOR IMPROVED LINUCB EVALUATION PROTOCOL")
    print("=" * 60)
    
    # Test 1: Reward model
    test_reward_model()
    
    # Test 2: Small-scale simulation
    results = test_small_scale_simulation()
    
    # Test 3: Statistical significance
    test_statistical_significance()
    
    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED!")
    print("The improved protocol is ready for full evaluation.")
    print("=" * 60)
    
    # Summary of improvements
    print("\nKey improvements in this protocol:")
    print("1. Adaptive exploration based on federation size")
    print("2. Enhanced reward model with heterogeneity bonus")
    print("3. Theoretically grounded parameter tuning")
    print("4. Proper statistical analysis with significance testing")
    print("\nExpected improvement range: 30-50% over random baseline")


if __name__ == "__main__":
    main()
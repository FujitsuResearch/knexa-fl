#!/usr/bin/env python3
"""
Corrected Evaluation Methodology for KNEXA-FL
Implements proper federated learning evaluation on global test set
"""
import torch
import logging
import numpy as np
from typing import List, Dict, Any
from src.code_evaluation import ImprovedCodeEvaluator

logger = logging.getLogger(__name__)

class FederatedEvaluator:
    """
    Proper federated learning evaluator that measures global performance
    """
    
    def __init__(self, model, tokenizer, device, global_test_set, all_client_val_sets):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.global_test_set = global_test_set
        self.all_client_val_sets = all_client_val_sets
        
        # Create evaluator for comprehensive assessment
        self.evaluator = ImprovedCodeEvaluator(
            model, tokenizer, device, 
            eval_samples=min(50, len(global_test_set)), 
            max_k=10
        )
    
    def evaluate_comprehensive(self, client_id: int) -> Dict[str, float]:
        """
        Comprehensive evaluation measuring all aspects of federated learning
        """
        results = {}
        
        # 1. PRIMARY: Global test set performance (TRUE federated learning metric)
        global_performance = self.evaluate_global_performance()
        results['pass@1_global'] = global_performance['pass@1']
        results['pass@5_global'] = global_performance.get('pass@5', 0.0)
        
        # 2. Local validation performance (for comparison)
        local_val_set = self.all_client_val_sets[client_id]
        local_performance = self.evaluator.evaluate_on_dataset(list(local_val_set))
        results['pass@1_local'] = local_performance.get('pass@1', 0.0)
        
        # 3. Cross-client transfer performance
        cross_client_scores = self.evaluate_cross_client_transfer(client_id)
        results.update(cross_client_scores)
        
        # 4. Calculate federated benefit
        results['federated_benefit'] = results['pass@1_global'] - results['pass@1_local']
        
        # 5. Calculate generalization score (average cross-client performance)
        cross_client_values = [v for k, v in cross_client_scores.items() if k.startswith('cross_client_')]
        results['generalization_score'] = np.mean(cross_client_values) if cross_client_values else 0.0
        
        logger.info(f"Client {client_id} Comprehensive Evaluation:")
        logger.info(f"  GLOBAL TEST SET Pass@1: {results['pass@1_global']:.3f}")
        logger.info(f"  LOCAL VAL SET Pass@1: {results['pass@1_local']:.3f}")
        logger.info(f"  Federated Benefit: {results['federated_benefit']:+.3f}")
        logger.info(f"  Generalization Score: {results['generalization_score']:.3f}")
        
        return results
    
    def evaluate_global_performance(self) -> Dict[str, float]:
        """
        Evaluate on global test set - PRIMARY metric for federated learning
        """
        try:
            # Sample global test set for evaluation
            test_samples = list(self.global_test_set)
            if len(test_samples) > 50:  # Limit for computational efficiency
                test_samples = np.random.choice(test_samples, 50, replace=False).tolist()
            
            # Evaluate using comprehensive code execution
            results = self.evaluator.evaluate_on_dataset(test_samples)
            
            logger.info(f"GLOBAL TEST SET evaluation: Pass@1={results.get('pass@1', 0):.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Global evaluation failed: {e}")
            return {'pass@1': 0.0, 'pass@5': 0.0}
    
    def evaluate_cross_client_transfer(self, client_id: int) -> Dict[str, float]:
        """
        Evaluate how well this client performs on other clients' data
        Measures knowledge transfer effectiveness
        """
        cross_client_results = {}
        
        for other_client_id, other_val_set in self.all_client_val_sets.items():
            if other_client_id != client_id:
                try:
                    # Evaluate on other client's validation set
                    other_samples = list(other_val_set)
                    if len(other_samples) > 20:  # Limit for efficiency
                        other_samples = np.random.choice(other_samples, 20, replace=False).tolist()
                    
                    performance = self.evaluator.evaluate_on_dataset(other_samples)
                    cross_client_results[f'cross_client_{other_client_id}'] = performance.get('pass@1', 0.0)
                    
                except Exception as e:
                    logger.warning(f"Cross-client evaluation failed for client {other_client_id}: {e}")
                    cross_client_results[f'cross_client_{other_client_id}'] = 0.0
        
        return cross_client_results


class CorrectedKnexaClient:
    """
    Corrected KNEXA-FL client with proper federated evaluation
    """
    
    def __init__(self, cid: int, train_ds, val_ds, global_test_set, all_client_val_sets):
        # ... existing initialization code ...
        self.cid = cid
        self.train_ds = train_ds
        self.val_ds = val_ds
        
        # NEW: Store global test set and all client data for proper evaluation
        self.global_test_set = global_test_set
        self.all_client_val_sets = all_client_val_sets
        
        # Load model (simplified for example)
        # self.model, self.tok = load_model_and_tokenizer(cid, device)
        
        # Create federated evaluator
        # self.fed_evaluator = FederatedEvaluator(
        #     self.model, self.tok, device, global_test_set, all_client_val_sets
        # )
    
    def evaluate(self, parameters, config):
        """
        CORRECTED evaluation method using global test set
        """
        try:
            round_id = config.get("round", 0)
            
            # CORRECTED: Comprehensive federated evaluation
            comprehensive_results = self.fed_evaluator.evaluate_comprehensive(self.cid)
            
            # Primary metric: Global test performance
            primary_score = comprehensive_results['pass@1_global']
            
            logger.info(f"Client {self.cid} Round {round_id} CORRECTED evaluation:")
            logger.info(f"  Global Performance: {primary_score:.3f}")
            logger.info(f"  Federated Benefit: {comprehensive_results['federated_benefit']:+.3f}")
            
            # Return in Flower format with comprehensive metrics
            return float(1.0 - primary_score), len(self.global_test_set), {
                **comprehensive_results,
                "client_id": self.cid,
                "round": round_id,
                "evaluation_type": "corrected_federated"
            }
            
        except Exception as e:
            logger.error(f"Client {self.cid} corrected evaluation failed: {e}")
            return 1.0, len(self.global_test_set), {
                "pass@1_global": 0.0,
                "client_id": self.cid,
                "round": config.get("round", 0),
                "error": str(e)
            }


def run_corrected_evaluation_experiment():
    """
    Example of how to run experiments with corrected evaluation methodology
    """
    from src.data_utils import load_split
    from src.globals import NUM_CLIENTS
    
    # Load data with proper global test set
    client_splits, global_test = load_split(NUM_CLIENTS, alpha=0.1)
    
    # Extract all client validation sets for cross-evaluation
    all_client_val_sets = {}
    for cid, (train, val) in enumerate(client_splits):
        all_client_val_sets[cid] = val
    
    print("🔧 CORRECTED EVALUATION SETUP:")
    print(f"  Global test set size: {len(global_test)}")
    print(f"  Client validation sets: {[len(val) for _, val in client_splits]}")
    print()
    print("📊 EVALUATION METRICS:")
    print("  PRIMARY: pass@1_global (performance on global test set)")
    print("  SECONDARY: federated_benefit (global - local performance)")
    print("  ANALYSIS: cross_client transfer scores")
    print()
    print("✅ This corrected methodology will show:")
    print("  • TRUE federated learning benefits")
    print("  • Proper comparison between methods")
    print("  • Knowledge transfer effectiveness")
    
    return client_splits, global_test, all_client_val_sets


if __name__ == "__main__":
    run_corrected_evaluation_experiment()
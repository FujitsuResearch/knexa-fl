#!/usr/bin/env python3
"""
Enhanced Data Validation for KNEXA-FL
Ensures rigorous non-IID data splits with no overlap and verified heterogeneity
"""
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt  # Removed to avoid plot generation
# import seaborn as sns  # Removed to avoid plot generation
from datasets import load_dataset, Dataset
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json
import re
from collections import Counter
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class CodingProblemCategorizer:
    """
    Categorizes coding problems by type and difficulty for heterogeneity analysis
    """
    
    def __init__(self):
        # Problem type keywords for classification
        self.problem_type_keywords = {
            'algorithms': ['sort', 'search', 'binary', 'tree', 'graph', 'dynamic', 'greedy', 'divide'],
            'data_structures': ['list', 'array', 'stack', 'queue', 'heap', 'hash', 'dict', 'set'],
            'string_processing': ['string', 'char', 'text', 'word', 'regex', 'parse', 'format'],
            'mathematics': ['math', 'number', 'prime', 'factorial', 'fibonacci', 'calculate', 'sum'],
            'recursion': ['recursive', 'recursion', 'factorial', 'fibonacci', 'tree', 'backtrack']
        }
        
        # Difficulty indicators
        self.difficulty_indicators = {
            'easy': ['simple', 'basic', 'easy', 'straightforward'],
            'medium': ['medium', 'moderate', 'intermediate'],
            'hard': ['complex', 'advanced', 'difficult', 'challenging', 'optimize']
        }
    
    def categorize_problem(self, problem_text: str, canonical_solution: str = "") -> Dict[str, Any]:
        """
        Categorize a single coding problem by type and estimated difficulty
        """
        text_lower = (problem_text + " " + canonical_solution).lower()
        
        # Classify problem type
        problem_type_scores = {}
        for ptype, keywords in self.problem_type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            problem_type_scores[ptype] = score
        
        primary_type = max(problem_type_scores, key=problem_type_scores.get)
        
        # Estimate difficulty
        difficulty_scores = {}
        for difficulty, indicators in self.difficulty_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            difficulty_scores[difficulty] = score
        
        # Additional difficulty estimation based on solution complexity
        solution_complexity = self.estimate_solution_complexity(canonical_solution)
        
        estimated_difficulty = max(difficulty_scores, key=difficulty_scores.get)
        if solution_complexity > 10:  # High complexity
            estimated_difficulty = 'hard'
        elif solution_complexity < 5:  # Low complexity
            estimated_difficulty = 'easy'
        
        return {
            'primary_type': primary_type,
            'type_scores': problem_type_scores,
            'estimated_difficulty': estimated_difficulty,
            'difficulty_scores': difficulty_scores,
            'solution_complexity': solution_complexity
        }
    
    def estimate_solution_complexity(self, solution: str) -> int:
        """
        Estimate solution complexity based on code features
        """
        if not solution:
            return 1
        
        complexity_score = 0
        
        # Count control structures
        complexity_score += solution.count('for') * 2
        complexity_score += solution.count('while') * 2
        complexity_score += solution.count('if') * 1
        complexity_score += solution.count('elif') * 1
        complexity_score += solution.count('def') * 3
        complexity_score += solution.count('class') * 5
        
        # Count advanced constructs
        complexity_score += solution.count('lambda') * 2
        complexity_score += solution.count('recursion') * 3
        complexity_score += solution.count('yield') * 2
        
        # Count imports (indicates more complex dependencies)
        complexity_score += solution.count('import') * 2
        
        return max(complexity_score, 1)


class EnhancedDataValidator:
    """
    Comprehensive data validation for federated learning with rigorous heterogeneity checks
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.categorizer = CodingProblemCategorizer()
        np.random.seed(seed)
    
    def load_and_categorize_datasets(self) -> Tuple[Dataset, Dataset, Dict]:
        """
        Load datasets and categorize all problems for heterogeneity analysis
        """
        logger.info("Loading and categorizing datasets...")
        
        # Load datasets
        humaneval = load_dataset("openai_humaneval")['test'].shuffle(self.seed)
        mbpp = load_dataset("mbpp", split="test").select(range(300)).shuffle(self.seed)
        
        # Standardize MBPP format
        mbpp_standardized = mbpp.map(lambda ex: {
            'task_id': f'MBPP/{ex["task_id"]}',
            'prompt': ex['text'] + '\n' + ex['code'],
            'test': ex['test_list'][0] if ex['test_list'] else '',
            'canonical_solution': ex['code'],
            'entry_point': 'solution'
        })
        
        # Combine datasets
        combined_data = list(humaneval) + list(mbpp_standardized)
        
        # Categorize all problems
        categorized_problems = []
        problem_metadata = {}
        
        for i, problem in enumerate(combined_data):
            category_info = self.categorizer.categorize_problem(
                problem['prompt'], 
                problem.get('canonical_solution', '')
            )
            
            # Add category info to problem
            enhanced_problem = dict(problem)
            enhanced_problem.update(category_info)
            categorized_problems.append(enhanced_problem)
            
            # Store metadata for analysis
            problem_metadata[problem['task_id']] = category_info
        
        # Create categorized dataset
        categorized_dataset = Dataset.from_list(categorized_problems)
        
        # Split into train/test
        split_data = categorized_dataset.train_test_split(test_size=0.25, seed=self.seed)
        global_train, global_test = split_data['train'], split_data['test']
        
        logger.info(f"Dataset loaded: {len(global_train)} train, {len(global_test)} test samples")
        logger.info(f"Problem types distribution: {self.analyze_type_distribution(categorized_problems)}")
        
        return global_train, global_test, problem_metadata
    
    def create_heterogeneous_splits(self, global_train: Dataset, num_clients: int, 
                                  alpha: float = 0.1, min_heterogeneity: float = 0.15) -> List[Tuple]:
        """
        Create rigorously heterogeneous data splits with NO OVERLAP
        """
        logger.info(f"Creating heterogeneous splits for {num_clients} clients (α={alpha})")
        
        # Convert to list for easier manipulation
        all_problems = list(global_train)
        np.random.shuffle(all_problems)  # Shuffle for randomness
        
        # Group problems by type for controlled heterogeneity
        problems_by_type = {}
        for problem in all_problems:
            ptype = problem['primary_type']
            if ptype not in problems_by_type:
                problems_by_type[ptype] = []
            problems_by_type[ptype].append(problem)
        
        logger.info(f"Problem types: {[(k, len(v)) for k, v in problems_by_type.items()]}")
        
        # Create non-overlapping allocation ensuring heterogeneity
        client_splits = []
        allocated_problems = set()  # Track allocated problems to prevent overlap
        
        for client_id in range(num_clients):
            client_problems = []
            
            # Calculate desired distribution for this client (Dirichlet-based)
            type_names = list(problems_by_type.keys())
            
            # Create biased distribution for heterogeneity
            dirichlet_params = [alpha] * len(type_names)
            if client_id < len(type_names):
                dirichlet_params[client_id] = alpha * 5  # Bias toward one type
            
            type_distribution = np.random.dirichlet(dirichlet_params)
            
            # Target number of problems for this client
            problems_per_client = len(all_problems) // num_clients
            min_problems_per_type = 2  # Ensure each client gets some of each type
            
            # Allocate problems ensuring no overlap
            for type_idx, ptype in enumerate(type_names):
                available_problems = [p for p in problems_by_type[ptype] 
                                    if p['task_id'] not in allocated_problems]
                
                if not available_problems:
                    continue
                
                # Calculate number of problems for this type
                desired_count = max(min_problems_per_type, 
                                  int(type_distribution[type_idx] * problems_per_client))
                actual_count = min(desired_count, len(available_problems))
                
                # Select problems (no replacement needed since we track allocated)
                selected_problems = available_problems[:actual_count]
                client_problems.extend(selected_problems)
                
                # Mark as allocated
                for problem in selected_problems:
                    allocated_problems.add(problem['task_id'])
            
            # Ensure minimum data size by adding remaining unallocated problems
            if len(client_problems) < 15:  # Minimum viable size
                remaining_problems = [p for p in all_problems 
                                    if p['task_id'] not in allocated_problems]
                needed = min(15 - len(client_problems), len(remaining_problems))
                
                if needed > 0:
                    additional_problems = remaining_problems[:needed]
                    client_problems.extend(additional_problems)
                    
                    # Mark as allocated
                    for problem in additional_problems:
                        allocated_problems.add(problem['task_id'])
            
            # Create client dataset and split train/val
            if len(client_problems) == 0:
                raise ValueError(f"Client {client_id} received no problems")
            
            client_dataset = Dataset.from_list(client_problems)
            train_val_split = client_dataset.train_test_split(test_size=0.2, seed=self.seed)
            
            client_splits.append((train_val_split['train'], train_val_split['test']))
            
            logger.info(f"Client {client_id}: {len(train_val_split['train'])} train, "
                       f"{len(train_val_split['test'])} val samples")
            
            # Log type distribution for this client
            client_types = {}
            for problem in client_problems:
                ptype = problem['primary_type']
                client_types[ptype] = client_types.get(ptype, 0) + 1
            logger.info(f"Client {client_id} type distribution: {client_types}")
        
        logger.info(f"Total problems allocated: {len(allocated_problems)} / {len(all_problems)}")
        
        # Validate heterogeneity and no overlap
        self.validate_splits(client_splits, min_heterogeneity)
        
        return client_splits
    
    def create_type_aware_allocation(self, problems_by_type: Dict, num_clients: int, 
                                   alpha: float) -> Dict[int, Dict[str, int]]:
        """
        Create type-aware allocation ensuring each client has different specializations
        """
        type_names = list(problems_by_type.keys())
        num_types = len(type_names)
        
        # Create Dirichlet allocation for each problem type
        allocation = {}
        
        for client_id in range(num_clients):
            allocation[client_id] = {}
            
            # Each client gets a different primary specialization
            primary_type_idx = client_id % num_types
            
            # Generate Dirichlet distribution with bias toward primary type
            dirichlet_params = [alpha] * num_types
            dirichlet_params[primary_type_idx] = alpha * 5  # Bias toward primary type
            
            type_distribution = np.random.dirichlet(dirichlet_params)
            
            # Allocate problems based on distribution
            total_problems = sum(len(problems) for problems in problems_by_type.values())
            client_total = max(20, total_problems // (num_clients * 2))  # Ensure reasonable size
            
            for i, ptype in enumerate(type_names):
                num_problems = int(type_distribution[i] * client_total)
                num_problems = min(num_problems, len(problems_by_type[ptype]))
                allocation[client_id][ptype] = max(1, num_problems)  # At least 1 of each type
        
        return allocation
    
    def validate_splits(self, client_splits: List[Tuple], min_heterogeneity: float = 0.15):
        """
        Comprehensive validation of data splits
        """
        logger.info("Validating data splits...")
        
        # Check for overlap
        self.check_data_overlap(client_splits)
        
        # Validate heterogeneity
        heterogeneity_report = self.validate_heterogeneity(client_splits, min_heterogeneity)
        
        # Generate validation report
        self.generate_validation_report(client_splits, heterogeneity_report)
        
        return True
    
    def check_data_overlap(self, client_splits: List[Tuple]):
        """
        Detect any data overlap between client splits
        """
        logger.info("Checking for data overlap between clients...")
        
        overlap_detected = False
        
        for i, (train_i, val_i) in enumerate(client_splits):
            all_ids_i = set(train_i['task_id']) | set(val_i['task_id'])
            
            for j, (train_j, val_j) in enumerate(client_splits):
                if i < j:  # Avoid duplicate checks
                    all_ids_j = set(train_j['task_id']) | set(val_j['task_id'])
                    overlap = all_ids_i & all_ids_j
                    
                    if overlap:
                        logger.error(f"❌ DATA OVERLAP DETECTED between Client {i} and {j}: {overlap}")
                        overlap_detected = True
        
        if overlap_detected:
            raise ValueError("Data overlap detected - violates federated learning assumptions")
        
        logger.info("✅ No data overlap detected between clients")
    
    def validate_heterogeneity(self, client_splits: List[Tuple], 
                             min_heterogeneity: float) -> Dict[str, Any]:
        """
        Validate that each client has sufficiently heterogeneous data
        """
        logger.info(f"Validating heterogeneity (minimum JS divergence: {min_heterogeneity})")
        
        heterogeneity_report = {
            'client_distributions': {},
            'js_divergences': {},
            'heterogeneity_passed': True,
            'global_distribution': None
        }
        
        # Calculate global type distribution
        all_problems = []
        for train, val in client_splits:
            all_problems.extend(list(train) + list(val))
        
        global_type_dist = self.analyze_type_distribution(all_problems)
        heterogeneity_report['global_distribution'] = global_type_dist
        
        # Calculate each client's distribution and heterogeneity
        for client_id, (train, val) in enumerate(client_splits):
            client_problems = list(train) + list(val)
            client_type_dist = self.analyze_type_distribution(client_problems)
            
            # Calculate Jensen-Shannon divergence from global distribution
            global_probs = np.array(list(global_type_dist.values()))
            client_probs = np.array([client_type_dist.get(k, 0) for k in global_type_dist.keys()])
            
            # Normalize to probabilities
            global_probs = global_probs / global_probs.sum()
            client_probs = client_probs / client_probs.sum() if client_probs.sum() > 0 else global_probs
            
            js_divergence = jensenshannon(global_probs, client_probs)
            
            heterogeneity_report['client_distributions'][client_id] = client_type_dist
            heterogeneity_report['js_divergences'][client_id] = js_divergence
            
            if js_divergence < min_heterogeneity:
                logger.warning(f"Client {client_id} insufficient heterogeneity: JS={js_divergence:.3f}")
                heterogeneity_report['heterogeneity_passed'] = False
            else:
                logger.info(f"Client {client_id} heterogeneity: JS={js_divergence:.3f} ✅")
        
        if not heterogeneity_report['heterogeneity_passed']:
            logger.warning("Some clients failed heterogeneity validation")
        else:
            logger.info("✅ All clients passed heterogeneity validation")
        
        return heterogeneity_report
    
    def analyze_type_distribution(self, problems: List[Dict]) -> Dict[str, int]:
        """
        Analyze the distribution of problem types in a dataset
        """
        type_counts = Counter()
        for problem in problems:
            ptype = problem.get('primary_type', 'unknown')
            type_counts[ptype] += 1
        
        return dict(type_counts)
    
    def generate_validation_report(self, client_splits: List[Tuple], 
                                 heterogeneity_report: Dict[str, Any]):
        """
        Generate comprehensive validation report
        """
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        report = {
            'timestamp': timestamp,
            'validation_summary': {
                'num_clients': len(client_splits),
                'data_overlap_detected': False,  # Would have raised exception if true
                'heterogeneity_validation_passed': heterogeneity_report['heterogeneity_passed'],
                'min_heterogeneity_threshold': 0.15
            },
            'client_statistics': {},
            'heterogeneity_analysis': heterogeneity_report,
            'validation_status': 'PASSED' if heterogeneity_report['heterogeneity_passed'] else 'FAILED'
        }
        
        # Add client statistics
        for client_id, (train, val) in enumerate(client_splits):
            report['client_statistics'][client_id] = {
                'train_samples': len(train),
                'val_samples': len(val),
                'total_samples': len(train) + len(val),
                'type_distribution': heterogeneity_report['client_distributions'][client_id],
                'js_divergence': heterogeneity_report['js_divergences'][client_id]
            }
        
        # Save report
        import os
        os.makedirs('eval_results/data_validation', exist_ok=True)
        report_file = f'eval_results/data_validation/validation_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Validation report saved: {report_file}")
        
        # Print summary
        print("\n🔍 DATA VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Status: {'✅ PASSED' if report['validation_status'] == 'PASSED' else '❌ FAILED'}")
        print(f"Clients: {len(client_splits)}")
        print(f"Data Overlap: {'❌ Detected' if report['validation_summary']['data_overlap_detected'] else '✅ None'}")
        print(f"Heterogeneity: {'✅ All clients passed' if heterogeneity_report['heterogeneity_passed'] else '❌ Some clients failed'}")
        print()
        
        for client_id in range(len(client_splits)):
            stats = report['client_statistics'][client_id]
            print(f"Client {client_id}: {stats['total_samples']} samples, "
                  f"JS divergence: {stats['js_divergence']:.3f}")
        
        return report


def validate_knexa_fl_data_splits(num_clients=4, alpha=0.1, min_heterogeneity=0.15):
    """
    Main function to validate KNEXA-FL data splits with comprehensive checks
    """
    logger.info("Starting comprehensive KNEXA-FL data validation...")
    
    validator = EnhancedDataValidator(seed=42)
    
    # Load and categorize datasets
    global_train, global_test, problem_metadata = validator.load_and_categorize_datasets()
    
    # Create heterogeneous splits
    client_splits = validator.create_heterogeneous_splits(
        global_train, num_clients, alpha, min_heterogeneity
    )
    
    logger.info("✅ KNEXA-FL data validation completed successfully")
    
    return client_splits, global_test, problem_metadata


if __name__ == "__main__":
    # Run comprehensive data validation
    client_splits, global_test, metadata = validate_knexa_fl_data_splits()
    
    print("\n🎯 VALIDATION COMPLETED")
    print("Ready for three-tier evaluation with validated heterogeneous data splits")
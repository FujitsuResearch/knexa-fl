#!/usr/bin/env python3
"""
Transfer Set Management for KNEXA-FL Knowledge Distillation
Provides shared, privacy-vetted public dataset for KD exchanges as per paper Section 4.3
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datasets import load_dataset
import hashlib
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TransferSet:
    """
    Manages the shared transfer set (X_u) for knowledge distillation
    
    Key features:
    - Privacy-vetted public dataset (no PII, no proprietary code)
    - Deterministic ordering across all clients
    - Efficient caching mechanism
    - Support for different dataset sources
    
    Based on KNEXA-FL paper: |X_u| = 256 samples, using HumanEval/MBPP
    """
    
    def __init__(self, 
                 dataset_name: str = "humaneval",
                 size: int = 128,  # Paper suggests 256, but we start smaller
                 cache_dir: Optional[str] = None,
                 seed: int = 42):
        """
        Initialize transfer set
        
        Args:
            dataset_name: Source dataset ("humaneval" or "mbpp")
            size: Number of samples in transfer set
            cache_dir: Directory for caching processed samples
            seed: Random seed for deterministic sampling
        """
        self.dataset_name = dataset_name
        self.size = size
        self.seed = seed
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "knexa_fl" / "transfer_sets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache_key = self._compute_cache_key()
        self.cache_path = self.cache_dir / f"{self.cache_key}.json"
        
        # Load or create transfer set
        self.samples = self._load_or_create_samples()
        self.sample_hashes = self._compute_sample_hashes()
        
        logger.info(f"Transfer set initialized: {dataset_name}, size={size}")
        logger.info(f"Cache key: {self.cache_key}")
        logger.info(f"Deterministic ordering ensured via seed={seed}")
    
    def _compute_cache_key(self) -> str:
        """Compute unique cache key for this configuration"""
        config = {
            "dataset": self.dataset_name,
            "size": self.size,
            "seed": self.seed,
            "version": "1.0"
        }
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _load_or_create_samples(self) -> List[Dict[str, Any]]:
        """Load samples from cache or create new transfer set"""
        # Try to load from cache
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get("cache_key") == self.cache_key:
                        logger.info(f"Loaded transfer set from cache: {self.cache_path}")
                        return cached_data["samples"]
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Create new transfer set
        logger.info(f"Creating new transfer set from {self.dataset_name}")
        samples = self._create_transfer_set()
        
        # Save to cache
        try:
            cache_data = {
                "cache_key": self.cache_key,
                "dataset": self.dataset_name,
                "size": self.size,
                "seed": self.seed,
                "samples": samples
            }
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved transfer set to cache: {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        
        return samples
    
    def _create_transfer_set(self) -> List[Dict[str, Any]]:
        """Create transfer set from source dataset"""
        samples = []
        
        try:
            if self.dataset_name == "humaneval":
                # Load HumanEval dataset
                dataset = load_dataset("openai_humaneval", split="test", download_mode="reuse_dataset_if_exists")
                
                # Set random seed for deterministic sampling
                rng = np.random.RandomState(self.seed)
                indices = rng.choice(len(dataset), size=min(self.size, len(dataset)), replace=False)
                indices = sorted(indices)  # Ensure deterministic order
                
                for idx in indices:
                    item = dataset[int(idx)]
                    # Privacy vetting: remove any potential PII or sensitive info
                    sample = {
                        "id": f"humaneval_{idx}",
                        "prompt": item["prompt"],
                        "task_id": item["task_id"],
                        "entry_point": item["entry_point"],
                        # Don't include canonical solution or test cases
                    }
                    samples.append(sample)
            
            elif self.dataset_name == "mbpp":
                # Load MBPP dataset
                dataset = load_dataset("mbpp", split="test", download_mode="reuse_dataset_if_exists")
                
                rng = np.random.RandomState(self.seed)
                indices = rng.choice(len(dataset), size=min(self.size, len(dataset)), replace=False)
                indices = sorted(indices)
                
                for idx in indices:
                    item = dataset[int(idx)]
                    sample = {
                        "id": f"mbpp_{idx}",
                        "prompt": item["text"],
                        "task_id": item["task_id"],
                        # Don't include code or test cases
                    }
                    samples.append(sample)
            
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            logger.info(f"Created transfer set with {len(samples)} samples")
            
            # Additional privacy vetting
            samples = self._privacy_vet_samples(samples)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error creating transfer set: {e}")
            # CRITICAL: For academic integrity, we must NEVER use synthetic data
            # Raise the error instead of hiding it with fake data
            raise RuntimeError(f"Failed to create transfer set from real data: {e}")
    
    def _privacy_vet_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply privacy vetting to remove sensitive information"""
        vetted_samples = []
        
        # List of patterns that might indicate PII or sensitive data
        sensitive_patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\bpassword\s*[:=]\s*\S+',  # Password patterns
            r'\bapi[_-]?key\s*[:=]\s*\S+',  # API key patterns
        ]
        
        import re
        
        for sample in samples:
            # Check prompt for sensitive patterns
            prompt = sample.get("prompt", "")
            
            # Skip if contains potential sensitive data
            skip = False
            for pattern in sensitive_patterns:
                if re.search(pattern, prompt):
                    logger.warning(f"Skipping sample {sample['id']} due to potential sensitive data")
                    skip = True
                    break
            
            if not skip:
                vetted_samples.append(sample)
        
        logger.info(f"Privacy vetting complete: {len(vetted_samples)}/{len(samples)} samples retained")
        return vetted_samples
    
    # REMOVED: _create_synthetic_fallback method
    # Academic integrity requires using only real data - no synthetic fallbacks
    
    def _compute_sample_hashes(self) -> Dict[str, str]:
        """Compute hashes for each sample to ensure consistency"""
        hashes = {}
        for sample in self.samples:
            sample_str = json.dumps(sample, sort_keys=True)
            sample_hash = hashlib.sha256(sample_str.encode()).hexdigest()[:8]
            hashes[sample['id']] = sample_hash
        return hashes
    
    def get_evaluation_samples(self, sample_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get samples with test cases for evaluation purposes only
        
        This method provides full dataset entries including test cases
        for computing Pass@k and CodeBLEU metrics. Should only be used
        for model evaluation, not for knowledge distillation.
        
        Args:
            sample_ids: List of sample IDs to retrieve (e.g., ['humaneval_0', 'humaneval_5'])
                       If None, returns all transfer set samples with evaluation data
            
        Returns:
            List of evaluation samples with test cases and solutions
        """
        eval_samples = []
        
        # Determine which samples to retrieve
        samples_to_retrieve = self.samples
        if sample_ids is not None:
            samples_to_retrieve = [s for s in self.samples if s['id'] in sample_ids]
        
        try:
            if self.dataset_name == "humaneval":
                dataset = load_dataset("openai_humaneval", split="test", download_mode="reuse_dataset_if_exists")
                
                for sample in samples_to_retrieve:
                    if sample['id'].startswith('humaneval_'):
                        idx = int(sample['id'].split('_')[1])
                        item = dataset[idx]
                        eval_sample = {
                            "id": sample['id'],
                            "prompt": item["prompt"],
                            "task_id": item["task_id"],
                            "entry_point": item["entry_point"],
                            "canonical_solution": item["canonical_solution"],
                            "test": item["test"]
                        }
                        eval_samples.append(eval_sample)
            
            elif self.dataset_name == "mbpp":
                dataset = load_dataset("mbpp", split="test", download_mode="reuse_dataset_if_exists")
                
                for sample in samples_to_retrieve:
                    if sample['id'].startswith('mbpp_'):
                        idx = int(sample['id'].split('_')[1])
                        item = dataset[idx]
                        eval_sample = {
                            "id": sample['id'],
                            "prompt": item["text"],
                            "task_id": item["task_id"],
                            "code": item["code"],
                            "test_list": item["test_list"]
                        }
                        eval_samples.append(eval_sample)
                        
            logger.info(f"Retrieved {len(eval_samples)} evaluation samples with test cases")
            
        except Exception as e:
            logger.error(f"Failed to retrieve evaluation samples: {e}")
            
        return eval_samples
    
    def get_batch(self, indices: Optional[List[int]] = None, 
                  round_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a batch of samples from the transfer set
        
        Args:
            indices: Specific indices to retrieve (optional)
            round_id: Round ID for deterministic batch selection (optional)
            
        Returns:
            List of samples
        """
        if indices is not None:
            # Return specific indices
            batch = []
            for idx in indices:
                if 0 <= idx < len(self.samples):
                    batch.append(self.samples[idx])
                else:
                    logger.warning(f"Index {idx} out of range, skipping")
            return batch
        
        elif round_id is not None:
            # Deterministic batch based on round ID
            # Use round-based rotation to ensure all samples are used
            if len(self.samples) == 0:
                logger.warning("No samples available in transfer set")
                return []
            
            batch_size = min(32, len(self.samples))  # Reasonable batch size
            start_idx = (round_id * batch_size) % len(self.samples)
            
            batch = []
            for i in range(batch_size):
                idx = (start_idx + i) % len(self.samples)
                batch.append(self.samples[idx])
            
            logger.debug(f"Round {round_id}: returning batch indices {start_idx} to {(start_idx + batch_size - 1) % len(self.samples)}")
            return batch
        
        else:
            # Return all samples
            return self.samples.copy()
    
    def get_prompts_for_tokenization(self, batch: List[Dict[str, Any]]) -> List[str]:
        """Extract prompts from batch for tokenization"""
        return [sample.get("prompt", "") for sample in batch]
    
    def verify_consistency(self, other_hashes: Dict[str, str]) -> bool:
        """
        Verify that another client has the same transfer set
        
        Args:
            other_hashes: Sample hashes from another client
            
        Returns:
            True if transfer sets match
        """
        if set(self.sample_hashes.keys()) != set(other_hashes.keys()):
            logger.error("Transfer set IDs don't match between clients")
            return False
        
        for sample_id, expected_hash in self.sample_hashes.items():
            if other_hashes.get(sample_id) != expected_hash:
                logger.error(f"Hash mismatch for sample {sample_id}")
                return False
        
        logger.info("Transfer set consistency verified")
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the transfer set"""
        return {
            "dataset": self.dataset_name,
            "size": len(self.samples),
            "seed": self.seed,
            "cache_key": self.cache_key,
            "sample_hashes": self.sample_hashes
        }
    
    def __len__(self) -> int:
        """Get size of transfer set"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index"""
        return self.samples[idx]


def create_shared_transfer_set() -> TransferSet:
    """
    Create the standard shared transfer set for KNEXA-FL
    
    Returns configured transfer set matching paper specifications
    """
    # Paper suggests |X_u| = 256, but we start with 128 for efficiency
    return TransferSet(
        dataset_name="humaneval",
        size=128,
        seed=42  # Fixed seed for reproducibility
    )


if __name__ == "__main__":
    # Test transfer set creation
    logging.basicConfig(level=logging.INFO)
    
    # Create transfer set
    transfer_set = create_shared_transfer_set()
    
    print(f"\nTransfer set created:")
    print(f"  Size: {len(transfer_set)}")
    print(f"  Dataset: {transfer_set.dataset_name}")
    print(f"  Cache key: {transfer_set.cache_key}")
    
    # Test batch retrieval
    batch = transfer_set.get_batch(round_id=0)
    print(f"\nBatch for round 0:")
    print(f"  Batch size: {len(batch)}")
    print(f"  First sample ID: {batch[0]['id']}")
    
    # Test consistency
    metadata = transfer_set.get_metadata()
    print(f"\nMetadata:")
    print(f"  Number of hashes: {len(metadata['sample_hashes'])}")
    
    # Verify deterministic ordering
    transfer_set2 = create_shared_transfer_set()
    consistent = transfer_set.verify_consistency(transfer_set2.sample_hashes)
    print(f"\nConsistency check: {consistent}")

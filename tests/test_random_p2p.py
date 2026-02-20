#!/usr/bin/env python3
"""
Unit tests for Random P2P Baseline
Tests actual training to ensure models are learning
"""

import pytest
import torch
import numpy as np
import json
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.client import KnexaClient
from src.data_utils import load_split

try:
    from baselines.random_p2p_real import (
        RandomP2PClient,
        RandomP2PCoordinator,
        RandomP2PExperiment,
        load_federated_datasets,
    )
    from baselines.random_p2p_flower import RandomP2PFlowerStrategy
except ModuleNotFoundError:
    pytest.skip(
        "Optional Random P2P baseline modules are not included in this repository snapshot.",
        allow_module_level=True,
    )


class TestRandomP2PBaseline:
    """Test suite for Random P2P baseline implementations"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_random_p2p_coordinator(self):
        """Test random pairing generation"""
        coordinator = RandomP2PCoordinator(num_clients=6, seed=42)
        
        # Test pairing generation
        pairs, unpaired = coordinator.generate_pairs(round_num=0)
        
        # Check that all clients are accounted for
        paired_clients = set()
        for s, t, _, _ in pairs:
            paired_clients.add(s)
            paired_clients.add(t)
        
        all_clients = paired_clients.union(set(unpaired))
        assert all_clients == set(range(6))
        
        # Check KD parameters
        for _, _, alpha, temp in pairs:
            assert alpha in [0.2, 0.3, 0.4]
            assert temp >= 1.5
        
        # Test odd number of clients
        coordinator_odd = RandomP2PCoordinator(num_clients=5, seed=42)
        pairs_odd, unpaired_odd = coordinator_odd.generate_pairs(round_num=0)
        assert len(unpaired_odd) == 1
        assert len(pairs_odd) == 2
    
    def test_model_training_decreases_loss(self, temp_dir):
        """Test that actual model training decreases loss"""
        # Small test with 4 clients, 2 rounds
        num_clients = 4
        num_rounds = 2
        
        # Load small dataset
        datasets, test_ds = load_federated_datasets(num_clients, alpha=0.1)
        
        # Initialize clients
        device = torch.device("cpu")  # Use CPU for tests
        clients = {}
        for i in range(num_clients):
            train_ds, val_ds = datasets[i]
            # Limit dataset size for faster testing
            train_ds = train_ds.select(range(min(10, len(train_ds))))
            val_ds = val_ds.select(range(min(5, len(val_ds))))
            
            clients[i] = RandomP2PClient(
                client_id=i,
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds.select(range(5)),
                device=device
            )
        
        # Get initial losses
        initial_losses = []
        for client in clients.values():
            val_loss, _ = client.evaluate()
            initial_losses.append(val_loss)
        
        # Run training
        coordinator = RandomP2PCoordinator(num_clients)
        
        for round_num in range(1, num_rounds + 1):
            pairs, unpaired = coordinator.generate_pairs(round_num)
            
            # Train paired clients
            for student_id, teacher_id, alpha, temperature in pairs:
                student = clients[student_id]
                teacher = clients[teacher_id]
                
                # Student learns from teacher
                student.knowledge_distillation_training(
                    teacher_model=teacher.model,
                    alpha=alpha,
                    temperature=temperature,
                    round_num=round_num
                )
                
                # Teacher trains locally
                teacher.local_training(round_num)
            
            # Train unpaired clients
            for client_id in unpaired:
                clients[client_id].local_training(round_num)
        
        # Get final losses
        final_losses = []
        for client in clients.values():
            val_loss, _ = client.evaluate()
            final_losses.append(val_loss)
        
        # Check that average loss decreased
        avg_initial_loss = np.mean(initial_losses)
        avg_final_loss = np.mean(final_losses)
        
        assert avg_final_loss < avg_initial_loss, \
            f"Training should decrease loss: initial={avg_initial_loss:.4f}, final={avg_final_loss:.4f}"
        
        # Check that at least some clients improved
        improvements = [initial - final for initial, final in zip(initial_losses, final_losses)]
        assert sum(imp > 0 for imp in improvements) >= num_clients // 2, \
            "At least half the clients should show improvement"
    
    def test_weights_actually_change(self):
        """Test that model weights actually change during training"""
        # Create minimal dataset
        from datasets import Dataset
        dummy_data = [
            {"prompt": "def hello():", "canonical_solution": "return 'world'"},
            {"prompt": "def add(a, b):", "canonical_solution": "return a + b"}
        ]
        train_ds = Dataset.from_list(dummy_data)
        val_ds = Dataset.from_list(dummy_data)
        test_ds = Dataset.from_list(dummy_data)
        
        # Create client
        device = torch.device("cpu")
        client = RandomP2PClient(
            client_id=0,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            device=device
        )
        
        # Get initial weights
        initial_weights = {
            name: param.clone().detach() 
            for name, param in client.model.named_parameters()
        }
        
        # Train for one round
        client.local_training(round_num=1)
        
        # Get final weights
        final_weights = {
            name: param.clone().detach() 
            for name, param in client.model.named_parameters()
        }
        
        # Check that some weights changed
        weights_changed = False
        for name in initial_weights:
            if not torch.allclose(initial_weights[name], final_weights[name], atol=1e-6):
                weights_changed = True
                break
        
        assert weights_changed, "Model weights should change after training"
    
    def test_flower_strategy_pairing(self):
        """Test Flower strategy creates proper pairings"""
        strategy = RandomP2PFlowerStrategy(
            num_clients=6,
            kd_alpha_grid=[0.2, 0.3, 0.4],
            temp_default=1.5,
            seed=42
        )
        
        # Mock client manager
        class MockClientManager:
            def all(self):
                return {i: f"client_{i}" for i in range(6)}
        
        client_manager = MockClientManager()
        
        # Test configure_fit
        fit_configs = strategy.configure_fit(
            server_round=1,
            parameters=None,
            client_manager=client_manager
        )
        
        # Check that all clients are configured
        assert len(fit_configs) == 6
        
        # Check role configurations
        student_count = 0
        teacher_count = 0
        unpaired_count = 0
        
        for client_proxy, fit_ins in fit_configs:
            config = fit_ins.config
            # Find the role config key
            role_key = [k for k in config.keys() if k.startswith("role_")][0]
            role_config = config[role_key]
            
            if role_config["role"] == "student":
                student_count += 1
                assert "teacher_cid" in role_config
                assert "alpha" in role_config
                assert "T" in role_config
            elif role_config["role"] == "teacher":
                teacher_count += 1
            elif role_config["role"] == "local_only":
                unpaired_count += 1
        
        # With 6 clients, we should have 3 students and 3 teachers
        assert student_count == 3
        assert teacher_count == 3
        assert unpaired_count == 0
    
    def test_knexa_client_integration(self):
        """Test that KnexaClient can be used with random P2P"""
        # Load minimal data
        client_splits, global_test = load_split(num_clients=2)
        train_ds, val_ds = client_splits[0]
        
        # Limit dataset size
        train_ds = train_ds.select(range(min(5, len(train_ds))))
        val_ds = val_ds.select(range(min(2, len(val_ds))))
        global_test = global_test.select(range(min(2, len(global_test))))
        
        # Create KnexaClient
        client = KnexaClient(
            cid=0,
            train_ds=train_ds,
            val_ds=val_ds,
            global_test_ds=global_test
        )
        
        # Test that client has required methods
        assert hasattr(client, 'fit')
        assert hasattr(client, 'evaluate')
        assert hasattr(client, 'get_parameters')
        
        # Test basic functionality
        params = client.get_parameters(config={})
        assert params is not None
        assert hasattr(params, 'tensors')
    
    @pytest.mark.slow
    def test_full_experiment_sanity(self, temp_dir):
        """Test full experiment runs without errors (slow test)"""
        experiment = RandomP2PExperiment(
            num_clients=2,
            num_rounds=2,
            alpha=0.1,
            output_dir=temp_dir
        )
        
        # Run experiment
        report = experiment.run_experiment()
        
        # Check report structure
        assert report['experiment_config']['method'] == 'random_p2p_baseline_real'
        assert report['experiment_config']['num_clients'] == 2
        assert report['experiment_config']['num_rounds'] == 2
        
        # Check that metrics exist
        assert 'final_metrics' in report
        assert 'avg_accuracy' in report['final_metrics']
        assert 'avg_pass_rate' in report['final_metrics']
        
        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, "results", "latest_summary.json"))
        assert os.path.exists(os.path.join(temp_dir, "results", "training_curves.json"))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

"""
KNEXA-FL Example Orchestration Script
Demonstrates the core framework functionality
"""

import yaml
import time
import numpy as np
from typing import List, Dict, Any

# Import core components
from core.cpm.orchestrator import CPMOrchestrator
from core.agents.agent import KnexaAgent, AgentConfig
from core.p2p.knowledge_exchange import P2PKnowledgeExchange
from core.cpm.privacy_profile import PrivacyParameters

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_mock_dataset(size: int, agent_id: str) -> List[Dict[str, Any]]:
    """Create mock dataset for demonstration"""
    dataset = []
    for i in range(size):
        dataset.append({
            'id': f"{agent_id}_sample_{i}",
            'text': f"Example training text for {agent_id} sample {i}",
            'prompt': f"Generate code for task {i}:",
            'label': f"def solution_{i}():\n    return {i}"
        })
    return dataset

def create_transfer_set(size: int) -> List[Dict[str, Any]]:
    """Create transfer set for knowledge distillation"""
    transfer_set = []
    for i in range(size):
        transfer_set.append({
            'id': f"transfer_{i}",
            'prompt': f"Solve problem {i}: "
        })
    return transfer_set

def main():
    """Main orchestration example"""
    print("=" * 60)
    print("KNEXA-FL Framework Demonstration")
    print("=" * 60)
    
    # Load configuration
    config = load_config("configs/default_config.yaml")
    
    # Initialize CPM
    print("\n1. Initializing Central Profiler/Matchmaker (CPM)")
    privacy_params = PrivacyParameters(
        k_anonymity=config['cpm']['privacy']['k_anonymity'],
        dp_epsilon=config['cpm']['privacy']['dp_epsilon'],
        dp_delta=config['cpm']['privacy']['dp_delta']
    )
    
    cpm = CPMOrchestrator(
        max_pairs_per_round=config['cpm']['max_pairs_per_round'],
        context_dim=config['cpm']['context_dimension'],
        privacy_params=privacy_params
    )
    print("✓ CPM initialized with privacy guarantees")
    
    # Create mock agents (in real implementation, these would be actual LLM agents)
    print("\n2. Creating federation agents")
    agents = {}
    num_agents = 4
    
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        
        # Mock model and tokenizer (replace with actual in production)
        class MockModel:
            def parameters(self):
                return [torch.randn(100, 100)]
            def to(self, device):
                return self
            def train(self):
                pass
            def eval(self):
                pass
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 4, 5]])
            def __call__(self, **kwargs):
                class Output:
                    loss = torch.tensor(2.0, requires_grad=True)
                return Output()
        
        class MockTokenizer:
            vocab_size = 50000
            eos_token_id = 1
            def __call__(self, text, **kwargs):
                class TokenizerOutput(dict):
                    def __init__(self):
                        super().__init__({'input_ids': torch.randint(0, 50000, (1, 10))})
                    def to(self, device):
                        return self
                return TokenizerOutput()
            def decode(self, tokens, **kwargs):
                return "Generated response"
        
        # Create agent
        train_data = create_mock_dataset(50, agent_id)
        val_data = create_mock_dataset(10, agent_id)
        
        agent = KnexaAgent(
            agent_id=agent_id,
            model=MockModel(),
            tokenizer=MockTokenizer(),
            train_dataset=train_data,
            val_dataset=val_data,
            config=AgentConfig(device="cpu")  # Use CPU for demo
        )
        
        agents[agent_id] = agent
        print(f"✓ Created {agent_id}")
    
    # Initialize P2P exchange module
    print("\n3. Initializing P2P Knowledge Exchange")
    p2p_exchange = P2PKnowledgeExchange()
    transfer_set = create_transfer_set(20)
    
    # Run federation rounds
    num_rounds = 5
    print(f"\n4. Running {num_rounds} federation rounds")
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Step 1: Local training
        print("Step 1: Local training")
        for agent_id, agent in agents.items():
            metrics = agent.local_training_step(num_epochs=1)
            print(f"  {agent_id}: loss={metrics['avg_loss']:.4f}")
        
        # Step 2: Update profiles
        print("Step 2: Updating agent profiles")
        for agent_id, agent in agents.items():
            profile = agent.generate_profile()
            cpm.update_agent_profile(agent_id, profile)
        
        # Step 3: Request matching
        print("Step 3: CPM matching")
        available_agents = list(agents.keys())
        pairings = cpm.request_matching(available_agents, round_num)
        
        if not pairings:
            print("  No pairings generated")
            continue
        
        print(f"  Generated {len(pairings)} pairings:")
        for pairing in pairings:
            print(f"    {pairing.teacher_id} → {pairing.student_id} "
                  f"(α={pairing.alpha:.2f}, T={pairing.temperature:.1f})")
        
        # Step 4: Execute P2P knowledge exchange
        print("Step 4: P2P knowledge exchange")
        for pairing in pairings:
            teacher = agents[pairing.teacher_id]
            student = agents[pairing.student_id]
            
            # Execute transfer
            result = p2p_exchange.execute_knowledge_transfer(
                teacher_agent=teacher,
                student_agent=student,
                transfer_samples=transfer_set[:10],  # Use subset
                alpha=pairing.alpha,
                temperature=pairing.temperature
            )
            
            if result['success']:
                print(f"  ✓ {pairing.teacher_id} → {pairing.student_id}: Success")
                
                # Update CPM with feedback
                feedback = {
                    'performance_delta': np.random.uniform(-0.1, 0.3),
                    'trust_change': 0.05 if result['success'] else -0.05
                }
                cpm.update_feedback(
                    pairing.student_id,
                    pairing.teacher_id,
                    feedback
                )
            else:
                print(f"  ✗ {pairing.teacher_id} → {pairing.student_id}: Failed")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("FEDERATION COMPLETE")
    print("=" * 60)
    
    # CPM statistics
    cpm_stats = cpm.get_statistics()
    print("\nCPM Statistics:")
    print(f"  Total rounds: {cpm_stats['num_rounds']}")
    print(f"  Total pairings: {cpm_stats['total_pairings']}")
    print(f"  LinUCB average reward: {cpm_stats['bandit_stats']['average_reward']:.4f}")
    
    # P2P statistics
    p2p_stats = p2p_exchange.get_exchange_statistics()
    print("\nP2P Exchange Statistics:")
    print(f"  Total exchanges: {p2p_stats['total_exchanges']}")
    print(f"  Success rate: {p2p_stats['success_rate']:.2%}")
    print(f"  Average duration: {p2p_stats['avg_duration']:.2f}s")
    
    # Agent statistics
    print("\nAgent Final Performance:")
    for agent_id, agent in agents.items():
        stats = agent.get_statistics()
        print(f"  {agent_id}: performance={stats['current_performance']:.4f}")
    
    print("\n✓ Demonstration complete!")

if __name__ == "__main__":
    # Import torch here to avoid issues if not installed
    try:
        import torch
    except ImportError:
        print("Note: PyTorch not installed. Using mock implementation.")
        # Create mock torch for demonstration
        class MockTorch:
            def randn(self, *args):
                return np.random.randn(*args)
            def randint(self, low, high, size):
                return np.random.randint(low, high, size)
            def tensor(self, data):
                return np.array(data)
            class nn:
                class utils:
                    @staticmethod
                    def clip_grad_norm_(params, max_norm):
                        pass
        torch = MockTorch()
    
    main()
#!/usr/bin/env python3
"""
Comprehensive Experiment Result Management System for KNEXA-FL
Principal ML Engineer-grade result tracking and organization
"""
import json
import yaml
import pickle
import shutil
import logging
import hashlib
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Removed to avoid plot generation
# import seaborn as sns  # Removed to avoid plot generation
from dataclasses import dataclass, asdict
import torch

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_name: str
    method: str = "KNEXA-FL"
    num_clients: int = 4
    num_rounds: int = 25
    learning_rate_local: float = 5e-5
    learning_rate_kd: float = 1e-4
    batch_size_local: int = 16
    batch_size_kd: int = 8
    alpha_dirichlet: float = 0.1
    temperature_kd: float = 2.0
    alpha_kd: float = 0.5
    enable_pass_at_k: bool = True
    pass_at_k_timing: str = "strategic"
    seed: int = 42
    local_pretrain_rounds: int = 0  # Number of initial local-only rounds before P2P collaboration
    additional_params: Dict[str, Any] = None


def format_learning_rate(lr_value: float) -> str:
    """Format learning rate for folder name (e.g., 1.0e-04 -> 1e-4)."""
    lr_str = f"{lr_value:.0e}"
    # Convert 1e-04 to 1e-4
    lr_str = re.sub(r'e-0+', 'e-', lr_str)
    lr_str = re.sub(r'e\+0+', 'e', lr_str)
    return lr_str


def build_experiment_folder_name(config: ExperimentConfig, timestamp: str) -> str:
    """Build new folder name based on hyperparameters following KNEXA-FL naming convention."""
    parts = [timestamp]
    
    # Method (m)
    parts.append(config.method)
    
    # Seed (s)
    parts.append(f"s{config.seed}")
    
    # Local pretrain rounds (lpr)
    parts.append(f"lpr{config.local_pretrain_rounds}")
    
    # Number of rounds (r)
    parts.append(f"r{config.num_rounds}")
    
    # Batch size local (bs)
    parts.append(f"bs{config.batch_size_local}")
    
    # Learning rate local (lr)
    lr_str = format_learning_rate(config.learning_rate_local)
    parts.append(f"lr{lr_str}")
    
    # Alpha dirichlet (ad)
    parts.append(f"ad{config.alpha_dirichlet}")
    
    # Optional tags
    tags = []
    
    # Pass@k enabled
    if config.enable_pass_at_k:
        tags.append('+pak')
        
        # Pass@k timing if not strategic
        if config.pass_at_k_timing != 'strategic':
            tags.append(f'_pak-{config.pass_at_k_timing}')
    
    # Additional params
    if config.additional_params:
        for key, value in sorted(config.additional_params.items()):
            tags.append(f'_ap-{key}-{value}')
    
    # Join all parts
    folder_name = '_'.join(parts)
    if tags:
        folder_name += ''.join(tags)
    
    return folder_name


@dataclass
class LossMetrics:
    """Container for comprehensive loss and evaluation metrics"""
    # Training losses
    local_training_losses: List[float] = None  # Losses during local training on client's own data
    local_validation_losses: List[float] = None  # Evaluation losses on client's validation set
    global_evaluation_losses: List[float] = None  # Evaluation losses on global test set
    
    # Evaluation metrics
    local_pass_at_k: Dict[str, List[float]] = None  # Pass@k on local validation set over rounds
    global_pass_at_k: Dict[str, List[float]] = None  # Pass@k on global test set over rounds
    local_codebleu: List[float] = None  # CodeBLEU on local validation set
    global_codebleu: List[float] = None  # CodeBLEU on global test set
    
    # Metadata
    round_numbers: List[int] = None  # Round numbers for alignment
    timestamps: List[str] = None  # Timestamps for each measurement
    
    def __post_init__(self):
        # Initialize empty lists if None
        if self.local_training_losses is None:
            self.local_training_losses = []
        if self.local_validation_losses is None:
            self.local_validation_losses = []
        if self.global_evaluation_losses is None:
            self.global_evaluation_losses = []
        if self.local_pass_at_k is None:
            self.local_pass_at_k = {"pass@1": [], "pass@5": [], "pass@10": [], "codebleu": []}
        if self.global_pass_at_k is None:
            self.global_pass_at_k = {"pass@1": [], "pass@5": [], "pass@10": [], "codebleu": []}
        if self.local_codebleu is None:
            self.local_codebleu = []
        if self.global_codebleu is None:
            self.global_codebleu = []
        if self.round_numbers is None:
            self.round_numbers = []
        if self.timestamps is None:
            self.timestamps = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LossMetrics to dictionary"""
        return asdict(self)


class ExperimentConfigMixin:
    """Mixin for ExperimentConfig methods"""
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if hasattr(self, 'additional_params') and self.additional_params:
            data.update(self.additional_params)
        return data
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Apply mixin to ExperimentConfig
ExperimentConfig.to_dict = ExperimentConfigMixin.to_dict
ExperimentConfig.get_hash = ExperimentConfigMixin.get_hash


@dataclass
class SystemInfo:
    """System information for reproducibility"""
    python_version: str
    torch_version: str
    cuda_version: str
    gpu_info: List[str]
    cpu_info: str
    memory_gb: float
    git_commit: str
    git_branch: str
    git_dirty: bool
    
    @classmethod
    def capture(cls) -> 'SystemInfo':
        """Capture current system information"""
        import platform
        import psutil
        
        # Python and package versions
        python_version = platform.python_version()
        torch_version = torch.__version__
        cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        # GPU info
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append(torch.cuda.get_device_name(i))
        
        # CPU and memory
        cpu_info = platform.processor()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Git info
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
            git_status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
            git_dirty = bool(git_status)
        except:
            git_commit = "unknown"
            git_branch = "unknown"
            git_dirty = False
        
        return cls(
            python_version=python_version,
            torch_version=torch_version,
            cuda_version=cuda_version,
            gpu_info=gpu_info,
            cpu_info=cpu_info,
            memory_gb=memory_gb,
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty
        )


class ExperimentManager:
    """
    Comprehensive experiment management system
    Handles result storage, organization, and analysis
    """
    
    def __init__(self, base_dir: str = "experimental_artifacts/knexa_fl"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Updated directory structure for unified system
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        self.summary_dir = self.base_dir / "summaries"
        self.summary_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.base_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.best_models_dir = self.base_dir / "best_models"
        self.best_models_dir.mkdir(exist_ok=True)
        
        # Load experiment registry
        self.registry_file = self.base_dir / "experiment_registry.json"
        self.registry = self._load_registry()
        
        # Comprehensive client metrics tracking
        self.client_metrics: Dict[str, Dict[int, LossMetrics]] = {}  # experiment_id -> client_id -> metrics
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load or create experiment registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"experiments": {}}
    
    def _save_registry(self):
        """Save experiment registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create new experiment with unified directory structure
        Returns experiment ID
        """
        # Generate experiment ID using new hyperparameter-based naming convention
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_id = build_experiment_folder_name(config, timestamp)
        
        # Create experiment directory with new unified structure
        exp_dir = self.runs_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create unified subdirectories
        subdirs = [
            "raw_data/round_results",
            "raw_data/metrics",
            "checkpoints",
            "code_generation/summaries",
            "code_generation/archive", 
            # "plots/training",  # Removed plot directories
            # "plots/performance",
            # "plots/communication",
            # "plots/system",
            # "paper_materials/figures",
            "paper_materials/tables",
            "paper_materials/sections",
            "paper_materials/data",
            "logs"
        ]
        
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        # Save system info
        system_info = SystemInfo.capture()
        system_info_path = exp_dir / "system_info.json"
        with open(system_info_path, 'w') as f:
            json.dump(asdict(system_info), f, indent=2)
        
        # Generate config hash
        config_hash = config.get_hash()
        
        # Create README
        readme_path = exp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"# Experiment: {experiment_id}\n\n")
            f.write(f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Method**: {config.method}\n\n")
            f.write(f"**Configuration Hash**: {config_hash}\n\n")
            f.write("## Configuration\n\n")
            f.write("```yaml\n")
            f.write(yaml.dump(config.to_dict(), default_flow_style=False))
            f.write("```\n\n")
            f.write("## Directory Structure\n\n")
            f.write("- `checkpoints/`: Model checkpoints\n")
            f.write("- `logs/`: Training logs\n")
            f.write("- `round_results/`: Per-round results\n")
            f.write("- `metrics/`: Evaluation metrics\n")
            f.write("- `plots/`: Visualizations\n")
            f.write("- `code_generation/`: Pass@k code generation logs\n")
        
        # Register experiment
        self.registry["experiments"][experiment_id] = {
            "created": datetime.now().isoformat(),
            "config": config.to_dict(),
            "status": "created",
            "path": str(exp_dir)
        }
        self._save_registry()
        
        logger.info(f"Created experiment: {experiment_id}")
        logger.info(f"Experiment directory: {exp_dir}")
        
        return experiment_id
    
    def get_experiment_dir(self, experiment_id: str) -> Path:
        """Get experiment directory"""
        return self.runs_dir / experiment_id
    
    def save_round_result(self, experiment_id: str, round_id: int, result: Dict[str, Any]):
        """Save results for a specific round"""
        exp_dir = self.get_experiment_dir(experiment_id)
        round_results_dir = exp_dir / "raw_data" / "round_results"
        round_results_dir.mkdir(parents=True, exist_ok=True)
        
        round_file = round_results_dir / f"round_{round_id:03d}.json"
        
        # Make serializable
        serializable_result = self._make_serializable(result)
        
        with open(round_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        # Also save compressed pickle for large data
        pickle_file = round_results_dir / f"round_{round_id:03d}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(result, f)
    
    def init_client_metrics(self, experiment_id: str, num_clients: int):
        """Initialize comprehensive metrics tracking for all clients"""
        if experiment_id not in self.client_metrics:
            self.client_metrics[experiment_id] = {}
        
        for client_id in range(num_clients):
            self.client_metrics[experiment_id][client_id] = LossMetrics()
        
        logger.info(f"Initialized comprehensive metrics tracking for {num_clients} clients in experiment {experiment_id}")
    
    def record_training_loss(self, experiment_id: str, client_id: int, round_id: int, losses: List[float]):
        """Record training losses for a client during local training"""
        if experiment_id not in self.client_metrics:
            self.init_client_metrics(experiment_id, client_id + 1)
        
        metrics = self.client_metrics[experiment_id][client_id]
        metrics.local_training_losses.extend(losses)
        metrics.round_numbers.extend([round_id] * len(losses))
        metrics.timestamps.extend([datetime.now().isoformat()] * len(losses))
        
        # Save to file
        self._save_client_metrics(experiment_id, client_id)
    
    def record_validation_metrics(self, experiment_id: str, client_id: int, round_id: int, 
                                local_loss: float, global_loss: float,
                                local_pass_at_k: Dict[str, float], global_pass_at_k: Dict[str, float],
                                local_codebleu: float = None, global_codebleu: float = None):
        """Record comprehensive evaluation metrics for both local and global datasets"""
        if experiment_id not in self.client_metrics:
            self.init_client_metrics(experiment_id, client_id + 1)
        
        metrics = self.client_metrics[experiment_id][client_id]
        timestamp = datetime.now().isoformat()
        
        # Record losses
        metrics.local_validation_losses.append(local_loss)
        metrics.global_evaluation_losses.append(global_loss)
        
        # Record pass@k metrics
        for k, value in local_pass_at_k.items():
            if k in metrics.local_pass_at_k:
                metrics.local_pass_at_k[k].append(value)
        
        for k, value in global_pass_at_k.items():
            if k in metrics.global_pass_at_k:
                metrics.global_pass_at_k[k].append(value)
        
        # Record CodeBLEU if available
        if local_codebleu is not None:
            metrics.local_codebleu.append(local_codebleu)
        
        if global_codebleu is not None:
            metrics.global_codebleu.append(global_codebleu)
        
        # Add metadata (only one entry per round for validation metrics)
        if round_id not in metrics.round_numbers:
            metrics.round_numbers.append(round_id)
            metrics.timestamps.append(timestamp)
        
        # Save to file
        self._save_client_metrics(experiment_id, client_id)
        
        logger.debug(f"Recorded validation metrics for client {client_id}, round {round_id}")
    
    def _save_client_metrics(self, experiment_id: str, client_id: int):
        """Save client metrics to disk"""
        exp_dir = self.get_experiment_dir(experiment_id)
        metrics_dir = exp_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        client_metrics_file = metrics_dir / f"client_{client_id}_metrics.json"
        metrics = self.client_metrics[experiment_id][client_id]
        
        with open(client_metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def generate_loss_plots(self, experiment_id: str):
        """Placeholder for plot generation - removed to avoid complications"""
        logger.info(f"Plot generation disabled for experiment {experiment_id} - results saved in JSON/CSV format")
        return  # Skip all plot generation
    
    '''def _plot_training_losses(self, experiment_id: str, plots_dir: Path, num_clients: int):
        """Plot training losses over iterations for each client"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
        
        for i, (client_id, metrics) in enumerate(self.client_metrics[experiment_id].items()):
            if i >= 4:  # Only plot first 4 clients in this view
                break
            
            ax = axes[i]
            if metrics.local_training_losses:
                # Create step numbers for x-axis
                steps = list(range(len(metrics.local_training_losses)))
                ax.plot(steps, metrics.local_training_losses, 
                       color=colors[i], linewidth=1.5, alpha=0.8)
                
                # Add smoothed trend line
                if len(steps) > 5:
                    # Simple moving average
                    window = max(1, len(steps) // 10)
                    smoothed = pd.Series(metrics.local_training_losses).rolling(window=window).mean()
                    ax.plot(steps, smoothed, color=colors[i], linewidth=3, alpha=0.9, 
                           label=f'Client {client_id} (smoothed)')
                
                ax.set_title(f'Client {client_id} Training Loss')
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_losses_per_client.png', dpi=300, bbox_inches='tight')
        plt.close()'''
    
    '''def _plot_validation_losses(self, experiment_id: str, plots_dir: Path, num_clients: int):
        """Plot validation losses (local vs global) over rounds"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
        
        for client_id, metrics in self.client_metrics[experiment_id].items():
            color = colors[client_id]
            
            # Local validation losses
            if metrics.local_validation_losses and metrics.round_numbers:
                ax1.plot(metrics.round_numbers[:len(metrics.local_validation_losses)], 
                        metrics.local_validation_losses, 
                        marker='o', linewidth=2, alpha=0.8, color=color, 
                        label=f'Client {client_id}')
            
            # Global evaluation losses
            if metrics.global_evaluation_losses and metrics.round_numbers:
                ax2.plot(metrics.round_numbers[:len(metrics.global_evaluation_losses)], 
                        metrics.global_evaluation_losses, 
                        marker='s', linewidth=2, alpha=0.8, color=color, 
                        label=f'Client {client_id}')
        
        ax1.set_title('Local Validation Loss Over Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Global Evaluation Loss Over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'validation_losses_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()'''
    
    '''def _plot_pass_at_k_evolution(self, experiment_id: str, plots_dir: Path, num_clients: int):
        """Plot Pass@k evolution for local vs global datasets"""
        metrics_to_plot = ['pass@1', 'pass@5', 'pass@10']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
        
        for j, metric in enumerate(metrics_to_plot):
            # Local Pass@k
            ax_local = axes[0, j]
            for client_id, client_metrics in self.client_metrics[experiment_id].items():
                if metric in client_metrics.local_pass_at_k and client_metrics.local_pass_at_k[metric]:
                    rounds = client_metrics.round_numbers[:len(client_metrics.local_pass_at_k[metric])]
                    ax_local.plot(rounds, client_metrics.local_pass_at_k[metric], 
                                marker='o', linewidth=2, alpha=0.8, color=colors[client_id], 
                                label=f'Client {client_id}')
            
            ax_local.set_title(f'Local {metric.upper()}')
            ax_local.set_xlabel('Round')
            ax_local.set_ylabel(f'{metric.upper()} Score')
            ax_local.legend()
            ax_local.grid(True, alpha=0.3)
            
            # Global Pass@k
            ax_global = axes[1, j]
            for client_id, client_metrics in self.client_metrics[experiment_id].items():
                if metric in client_metrics.global_pass_at_k and client_metrics.global_pass_at_k[metric]:
                    rounds = client_metrics.round_numbers[:len(client_metrics.global_pass_at_k[metric])]
                    ax_global.plot(rounds, client_metrics.global_pass_at_k[metric], 
                                 marker='s', linewidth=2, alpha=0.8, color=colors[client_id], 
                                 label=f'Client {client_id}')
            
            ax_global.set_title(f'Global {metric.upper()}')
            ax_global.set_xlabel('Round')
            ax_global.set_ylabel(f'{metric.upper()} Score')
            ax_global.legend()
            ax_global.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'pass_at_k_evolution_local_vs_global.png', dpi=300, bbox_inches='tight')
        plt.close()'''
    
    '''def _plot_codebleu_evolution(self, experiment_id: str, plots_dir: Path, num_clients: int):
        """Plot CodeBLEU evolution for local vs global datasets"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
        
        for client_id, metrics in self.client_metrics[experiment_id].items():
            color = colors[client_id]
            
            # Local CodeBLEU
            if metrics.local_codebleu and metrics.round_numbers:
                rounds = metrics.round_numbers[:len(metrics.local_codebleu)]
                ax1.plot(rounds, metrics.local_codebleu, 
                        marker='o', linewidth=2, alpha=0.8, color=color, 
                        label=f'Client {client_id}')
            
            # Global CodeBLEU
            if metrics.global_codebleu and metrics.round_numbers:
                rounds = metrics.round_numbers[:len(metrics.global_codebleu)]
                ax2.plot(rounds, metrics.global_codebleu, 
                        marker='s', linewidth=2, alpha=0.8, color=color, 
                        label=f'Client {client_id}')
        
        ax1.set_title('Local CodeBLEU Over Rounds')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('CodeBLEU Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Global CodeBLEU Over Rounds')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('CodeBLEU Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'codebleu_evolution_local_vs_global.png', dpi=300, bbox_inches='tight')
        plt.close()'''
    
    '''def _plot_combined_dashboard(self, experiment_id: str, plots_dir: Path, num_clients: int):
        """Create a comprehensive dashboard view"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_clients))
        
        # Average metrics across all clients
        avg_local_losses = []
        avg_global_losses = []
        avg_local_pass1 = []
        avg_global_pass1 = []
        max_rounds = 0
        
        for client_id, metrics in self.client_metrics[experiment_id].items():
            max_rounds = max(max_rounds, len(metrics.round_numbers))
        
        for round_idx in range(max_rounds):
            local_losses = []
            global_losses = []
            local_pass1 = []
            global_pass1 = []
            
            for client_id, metrics in self.client_metrics[experiment_id].items():
                if round_idx < len(metrics.local_validation_losses):
                    local_losses.append(metrics.local_validation_losses[round_idx])
                if round_idx < len(metrics.global_evaluation_losses):
                    global_losses.append(metrics.global_evaluation_losses[round_idx])
                if 'pass@1' in metrics.local_pass_at_k and round_idx < len(metrics.local_pass_at_k['pass@1']):
                    local_pass1.append(metrics.local_pass_at_k['pass@1'][round_idx])
                if 'pass@1' in metrics.global_pass_at_k and round_idx < len(metrics.global_pass_at_k['pass@1']):
                    global_pass1.append(metrics.global_pass_at_k['pass@1'][round_idx])
            
            if local_losses:
                avg_local_losses.append(np.mean(local_losses))
            if global_losses:
                avg_global_losses.append(np.mean(global_losses))
            if local_pass1:
                avg_local_pass1.append(np.mean(local_pass1))
            if global_pass1:
                avg_global_pass1.append(np.mean(global_pass1))
        
        # Plot average metrics
        rounds = list(range(1, len(avg_local_losses) + 1))
        
        # Average Loss Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        if avg_local_losses:
            ax1.plot(rounds, avg_local_losses, marker='o', linewidth=3, label='Local Validation', alpha=0.8)
        if avg_global_losses:
            ax1.plot(list(range(1, len(avg_global_losses) + 1)), avg_global_losses, 
                    marker='s', linewidth=3, label='Global Evaluation', alpha=0.8)
        ax1.set_title('Average Loss Across All Clients')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average Pass@1 Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        if avg_local_pass1:
            ax2.plot(list(range(1, len(avg_local_pass1) + 1)), avg_local_pass1, 
                    marker='o', linewidth=3, label='Local Validation', alpha=0.8)
        if avg_global_pass1:
            ax2.plot(list(range(1, len(avg_global_pass1) + 1)), avg_global_pass1, 
                    marker='s', linewidth=3, label='Global Evaluation', alpha=0.8)
        ax2.set_title('Average Pass@1 Across All Clients')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Pass@1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Individual client progression (Pass@1 only for clarity)
        for i in range(min(num_clients, 8)):  # Max 8 clients
            ax = fig.add_subplot(gs[1 + i//4, i%4])
            client_id = list(self.client_metrics[experiment_id].keys())[i]
            metrics = self.client_metrics[experiment_id][client_id]
            
            if 'pass@1' in metrics.local_pass_at_k and metrics.local_pass_at_k['pass@1']:
                rounds_local = list(range(1, len(metrics.local_pass_at_k['pass@1']) + 1))
                ax.plot(rounds_local, metrics.local_pass_at_k['pass@1'], 
                       marker='o', linewidth=2, alpha=0.8, label='Local', color=colors[i])
            
            if 'pass@1' in metrics.global_pass_at_k and metrics.global_pass_at_k['pass@1']:
                rounds_global = list(range(1, len(metrics.global_pass_at_k['pass@1']) + 1))
                ax.plot(rounds_global, metrics.global_pass_at_k['pass@1'], 
                       marker='s', linewidth=2, alpha=0.8, label='Global', color=colors[i])
            
            ax.set_title(f'Client {client_id} Pass@1')
            ax.set_xlabel('Round')
            ax.set_ylabel('Pass@1')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.savefig(plots_dir / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()'''
    
    def generate_comprehensive_report(self, experiment_id: str):
        """Generate comprehensive final evaluation report for paper submission"""
        exp_dir = self.get_experiment_dir(experiment_id)
        
        # Load experiment configuration
        config_file = exp_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Load system information
        system_info_file = exp_dir / "system_info.json"
        if system_info_file.exists():
            with open(system_info_file, 'r') as f:
                system_info = json.load(f)
        else:
            system_info = {}
        
        # Compile comprehensive report
        report = self._compile_comprehensive_report(experiment_id, config, system_info)
        
        # Save as JSON (structured data for programmatic access)
        report_json_file = exp_dir / "comprehensive_evaluation_report.json"
        with open(report_json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as human-readable text/markdown
        report_md_file = exp_dir / "comprehensive_evaluation_report.md"
        self._write_markdown_report(report, report_md_file)
        
        # Save LaTeX table snippets for paper
        latex_dir = exp_dir / "paper_materials"
        latex_dir.mkdir(exist_ok=True)
        self._generate_latex_tables(report, latex_dir)
        
        logger.info(f"Generated comprehensive evaluation report for experiment: {experiment_id}")
        return report
    
    def _compile_comprehensive_report(self, experiment_id: str, config: Dict[str, Any], system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Compile all experimental data into comprehensive report"""
        
        # Basic experiment information
        experiment_info = {
            "experiment_id": experiment_id,
            "creation_timestamp": self.registry["experiments"][experiment_id].get("created", "unknown"),
            "completion_timestamp": self.registry["experiments"][experiment_id].get("completed", "unknown"),
            "status": self.registry["experiments"][experiment_id].get("status", "unknown")
        }
        
        # Comprehensive hyperparameters and settings
        hyperparameters = {
            "method": config.get("method", "KNEXA-FL"),
            "num_clients": config.get("num_clients", 4),
            "num_rounds": config.get("num_rounds", 25),
            "learning_rates": {
                "local": config.get("learning_rate_local", 5e-5),
                "knowledge_distillation": config.get("learning_rate_kd", 1e-4)
            },
            "batch_sizes": {
                "local": config.get("batch_size_local", 16),
                "knowledge_distillation": config.get("batch_size_kd", 8)
            },
            "federated_learning": {
                "alpha_dirichlet": config.get("alpha_dirichlet", 0.1),
                "temperature_kd": config.get("temperature_kd", 2.0),
                "alpha_kd": config.get("alpha_kd", 0.5)
            },
            "evaluation": {
                "enable_pass_at_k": config.get("enable_pass_at_k", True),
                "pass_at_k_timing": config.get("pass_at_k_timing", "strategic")
            },
            "reproducibility": {
                "seed": config.get("seed", 42)
            }
        }
        
        # Dataset information 
        dataset_info = {
            "name": "HumanEval + MBPP",
            "humaneval_source": "openai_humaneval",
            "mbpp_source": "mbpp (first 300 problems)",
            "total_problems": "HumanEval: 164, MBPP: 300",
            "data_split_method": "Dirichlet distribution for non-IID",
            "validation_ratio": "20% of each client's data",
            "global_test_set": "25% of total data reserved for global evaluation"
        }
        
        # System and environment
        environment_info = {
            "hardware": {
                "gpu_info": system_info.get("gpu_info", ["Unknown"]),
                "cpu_info": system_info.get("cpu_info", "Unknown"),
                "memory_gb": system_info.get("memory_gb", 0)
            },
            "software": {
                "python_version": system_info.get("python_version", "Unknown"),
                "torch_version": system_info.get("torch_version", "Unknown"),
                "cuda_version": system_info.get("cuda_version", "Unknown")
            },
            "reproducibility": {
                "git_commit": system_info.get("git_commit", "Unknown"),
                "git_branch": system_info.get("git_branch", "Unknown"),
                "git_dirty": system_info.get("git_dirty", False)
            }
        }
        
        # Comprehensive client-specific results
        client_results = {}
        if experiment_id in self.client_metrics:
            for client_id, metrics in self.client_metrics[experiment_id].items():
                
                # Calculate statistics for each metric type
                training_loss_stats = self._calculate_stats(metrics.local_training_losses)
                local_val_loss_stats = self._calculate_stats(metrics.local_validation_losses)
                global_eval_loss_stats = self._calculate_stats(metrics.global_evaluation_losses)
                
                # Pass@k final and progression
                local_pass_at_k_final = {}
                global_pass_at_k_final = {}
                pass_at_k_progression = {}
                
                for k in ['pass@1', 'pass@5', 'pass@10', 'codebleu']:
                    if k in metrics.local_pass_at_k and metrics.local_pass_at_k[k]:
                        local_pass_at_k_final[k] = metrics.local_pass_at_k[k][-1]
                        pass_at_k_progression[f'local_{k}'] = metrics.local_pass_at_k[k]
                    
                    if k in metrics.global_pass_at_k and metrics.global_pass_at_k[k]:
                        global_pass_at_k_final[k] = metrics.global_pass_at_k[k][-1]
                        pass_at_k_progression[f'global_{k}'] = metrics.global_pass_at_k[k]
                
                # CodeBLEU final values
                local_codebleu_final = metrics.local_codebleu[-1] if metrics.local_codebleu else None
                global_codebleu_final = metrics.global_codebleu[-1] if metrics.global_codebleu else None
                
                client_results[f"client_{client_id}"] = {
                    "losses": {
                        "training": {
                            "all_values": metrics.local_training_losses,
                            "statistics": training_loss_stats
                        },
                        "local_validation": {
                            "all_values": metrics.local_validation_losses,
                            "statistics": local_val_loss_stats
                        },
                        "global_evaluation": {
                            "all_values": metrics.global_evaluation_losses,
                            "statistics": global_eval_loss_stats
                        }
                    },
                    "evaluation_metrics": {
                        "local_dataset": {
                            "pass_at_k_final": local_pass_at_k_final,
                            "codebleu_final": local_codebleu_final,
                            "pass_at_k_progression": {k: v for k, v in pass_at_k_progression.items() if k.startswith('local_')}
                        },
                        "global_dataset": {
                            "pass_at_k_final": global_pass_at_k_final,
                            "codebleu_final": global_codebleu_final,
                            "pass_at_k_progression": {k: v for k, v in pass_at_k_progression.items() if k.startswith('global_')}
                        }
                    },
                    "metadata": {
                        "round_numbers": metrics.round_numbers,
                        "timestamps": metrics.timestamps,
                        "total_training_steps": len(metrics.local_training_losses)
                    }
                }
        
        # Aggregate statistics across all clients
        aggregate_stats = self._calculate_aggregate_statistics(experiment_id)
        
        # Compile final report
        comprehensive_report = {
            "experiment_info": experiment_info,
            "hyperparameters": hyperparameters,
            "dataset_info": dataset_info,
            "environment_info": environment_info,
            "client_results": client_results,
            "aggregate_statistics": aggregate_stats,
            "academic_integrity_statement": "All results are from actual model executions. No synthetic or fabricated values were used.",
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return comprehensive_report
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values"""
        if not values:
            return {"count": 0}
        
        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "median": float(np.median(values_array)),
            "q25": float(np.percentile(values_array, 25)),
            "q75": float(np.percentile(values_array, 75)),
            "final_value": float(values[-1]),
            "improvement": float(values[-1] - values[0]) if len(values) > 1 else 0.0
        }
    
    def _calculate_aggregate_statistics(self, experiment_id: str) -> Dict[str, Any]:
        """Calculate aggregate statistics across all clients"""
        if experiment_id not in self.client_metrics:
            return {}
        
        # Collect all final values
        final_local_pass1 = []
        final_global_pass1 = []
        final_local_pass5 = []
        final_global_pass5 = []
        final_local_pass10 = []
        final_global_pass10 = []
        final_local_codebleu = []
        final_global_codebleu = []
        
        avg_training_loss_final = []
        avg_local_val_loss_final = []
        avg_global_eval_loss_final = []
        
        for client_id, metrics in self.client_metrics[experiment_id].items():
            # Pass@k metrics
            if 'pass@1' in metrics.local_pass_at_k and metrics.local_pass_at_k['pass@1']:
                final_local_pass1.append(metrics.local_pass_at_k['pass@1'][-1])
            if 'pass@1' in metrics.global_pass_at_k and metrics.global_pass_at_k['pass@1']:
                final_global_pass1.append(metrics.global_pass_at_k['pass@1'][-1])
            
            if 'pass@5' in metrics.local_pass_at_k and metrics.local_pass_at_k['pass@5']:
                final_local_pass5.append(metrics.local_pass_at_k['pass@5'][-1])
            if 'pass@5' in metrics.global_pass_at_k and metrics.global_pass_at_k['pass@5']:
                final_global_pass5.append(metrics.global_pass_at_k['pass@5'][-1])
            
            if 'pass@10' in metrics.local_pass_at_k and metrics.local_pass_at_k['pass@10']:
                final_local_pass10.append(metrics.local_pass_at_k['pass@10'][-1])
            if 'pass@10' in metrics.global_pass_at_k and metrics.global_pass_at_k['pass@10']:
                final_global_pass10.append(metrics.global_pass_at_k['pass@10'][-1])
            
            # CodeBLEU
            if metrics.local_codebleu:
                final_local_codebleu.append(metrics.local_codebleu[-1])
            if metrics.global_codebleu:
                final_global_codebleu.append(metrics.global_codebleu[-1])
            
            # Losses
            if metrics.local_training_losses:
                avg_training_loss_final.append(metrics.local_training_losses[-1])
            if metrics.local_validation_losses:
                avg_local_val_loss_final.append(metrics.local_validation_losses[-1])
            if metrics.global_evaluation_losses:
                avg_global_eval_loss_final.append(metrics.global_evaluation_losses[-1])
        
        return {
            "final_performance_averages": {
                "local_dataset": {
                    "pass@1": {"mean": float(np.mean(final_local_pass1)), "std": float(np.std(final_local_pass1))} if final_local_pass1 else None,
                    "pass@5": {"mean": float(np.mean(final_local_pass5)), "std": float(np.std(final_local_pass5))} if final_local_pass5 else None,
                    "pass@10": {"mean": float(np.mean(final_local_pass10)), "std": float(np.std(final_local_pass10))} if final_local_pass10 else None,
                    "codebleu": {"mean": float(np.mean(final_local_codebleu)), "std": float(np.std(final_local_codebleu))} if final_local_codebleu else None,
                    "validation_loss": {"mean": float(np.mean(avg_local_val_loss_final)), "std": float(np.std(avg_local_val_loss_final))} if avg_local_val_loss_final else None
                },
                "global_dataset": {
                    "pass@1": {"mean": float(np.mean(final_global_pass1)), "std": float(np.std(final_global_pass1))} if final_global_pass1 else None,
                    "pass@5": {"mean": float(np.mean(final_global_pass5)), "std": float(np.std(final_global_pass5))} if final_global_pass5 else None,
                    "pass@10": {"mean": float(np.mean(final_global_pass10)), "std": float(np.std(final_global_pass10))} if final_global_pass10 else None,
                    "codebleu": {"mean": float(np.mean(final_global_codebleu)), "std": float(np.std(final_global_codebleu))} if final_global_codebleu else None,
                    "evaluation_loss": {"mean": float(np.mean(avg_global_eval_loss_final)), "std": float(np.std(avg_global_eval_loss_final))} if avg_global_eval_loss_final else None
                }
            },
            "training_loss_final": {"mean": float(np.mean(avg_training_loss_final)), "std": float(np.std(avg_training_loss_final))} if avg_training_loss_final else None,
            "num_clients_analyzed": len(self.client_metrics[experiment_id])
        }
    
    def _write_markdown_report(self, report: Dict[str, Any], output_file: Path):
        """Write comprehensive report as human-readable markdown"""
        
        with open(output_file, 'w') as f:
            f.write("# KNEXA-FL Comprehensive Evaluation Report\n\n")
            
            # Experiment Info
            f.write("## Experiment Information\n\n")
            exp_info = report["experiment_info"]
            f.write(f"- **Experiment ID**: {exp_info['experiment_id']}\n")
            f.write(f"- **Created**: {exp_info['creation_timestamp']}\n")
            f.write(f"- **Completed**: {exp_info['completion_timestamp']}\n")
            f.write(f"- **Status**: {exp_info['status']}\n\n")
            
            # Hyperparameters
            f.write("## Experimental Configuration\n\n")
            hp = report["hyperparameters"]
            f.write(f"- **Method**: {hp['method']}\n")
            f.write(f"- **Number of Clients**: {hp['num_clients']}\n")
            f.write(f"- **Number of Rounds**: {hp['num_rounds']}\n")
            f.write(f"- **Local Learning Rate**: {hp['learning_rates']['local']}\n")
            f.write(f"- **KD Learning Rate**: {hp['learning_rates']['knowledge_distillation']}\n")
            f.write(f"- **Local Batch Size**: {hp['batch_sizes']['local']}\n")
            f.write(f"- **KD Batch Size**: {hp['batch_sizes']['knowledge_distillation']}\n")
            f.write(f"- **Dirichlet Alpha**: {hp['federated_learning']['alpha_dirichlet']}\n")
            f.write(f"- **KD Temperature**: {hp['federated_learning']['temperature_kd']}\n")
            f.write(f"- **KD Alpha**: {hp['federated_learning']['alpha_kd']}\n")
            f.write(f"- **Seed**: {hp['reproducibility']['seed']}\n\n")
            
            # Dataset Info
            f.write("## Dataset Information\n\n")
            ds = report["dataset_info"]
            f.write(f"- **Dataset**: {ds['name']}\n")
            f.write(f"- **HumanEval Source**: {ds['humaneval_source']}\n")
            f.write(f"- **MBPP Source**: {ds['mbpp_source']}\n")
            f.write(f"- **Total Problems**: {ds['total_problems']}\n")
            f.write(f"- **Data Split**: {ds['data_split_method']}\n")
            f.write(f"- **Validation Ratio**: {ds['validation_ratio']}\n")
            f.write(f"- **Global Test Set**: {ds['global_test_set']}\n\n")
            
            # Environment
            f.write("## System Environment\n\n")
            env = report["environment_info"]
            f.write(f"- **GPU**: {', '.join(env['hardware']['gpu_info'])}\n")
            f.write(f"- **CPU**: {env['hardware']['cpu_info']}\n")
            f.write(f"- **Memory**: {env['hardware']['memory_gb']:.1f} GB\n")
            f.write(f"- **Python**: {env['software']['python_version']}\n")
            f.write(f"- **PyTorch**: {env['software']['torch_version']}\n")
            f.write(f"- **CUDA**: {env['software']['cuda_version']}\n")
            f.write(f"- **Git Commit**: {env['reproducibility']['git_commit'][:8]}\n")
            f.write(f"- **Git Branch**: {env['reproducibility']['git_branch']}\n\n")
            
            # Aggregate Results
            f.write("## Summary Results\n\n")
            agg = report["aggregate_statistics"]
            if "final_performance_averages" in agg:
                local_avg = agg["final_performance_averages"]["local_dataset"]
                global_avg = agg["final_performance_averages"]["global_dataset"]
                
                f.write("### Average Performance (Final Round)\n\n")
                f.write("| Metric | Local Dataset | Global Dataset |\n")
                f.write("|--------|---------------|----------------|\n")
                
                for metric in ["pass@1", "pass@5", "pass@10", "codebleu"]:
                    local_val = local_avg.get(metric)
                    global_val = global_avg.get(metric)
                    
                    local_str = f"{local_val['mean']:.3f} ± {local_val['std']:.3f}" if local_val else "N/A"
                    global_str = f"{global_val['mean']:.3f} ± {global_val['std']:.3f}" if global_val else "N/A"
                    
                    f.write(f"| {metric} | {local_str} | {global_str} |\n")
                
                f.write("\n")
            
            # Client-specific detailed results
            f.write("## Per-Client Detailed Results\n\n")
            client_results = report["client_results"]
            for client_name, client_data in client_results.items():
                f.write(f"### {client_name.replace('_', ' ').title()}\n\n")
                
                # Loss statistics
                losses = client_data["losses"]
                f.write("#### Loss Statistics\n\n")
                f.write("| Loss Type | Final Value | Mean | Std | Min | Max | Improvement |\n")
                f.write("|-----------|-------------|------|-----|-----|-----|-------------|\n")
                
                for loss_type, loss_data in losses.items():
                    stats = loss_data["statistics"]
                    if stats.get("count", 0) > 0:
                        f.write(f"| {loss_type.replace('_', ' ').title()} | ")
                        f.write(f"{stats['final_value']:.4f} | ")
                        f.write(f"{stats['mean']:.4f} | ")
                        f.write(f"{stats['std']:.4f} | ")
                        f.write(f"{stats['min']:.4f} | ")
                        f.write(f"{stats['max']:.4f} | ")
                        f.write(f"{stats['improvement']:.4f} |\n")
                
                f.write("\n")
                
                # Evaluation metrics
                eval_metrics = client_data["evaluation_metrics"]
                f.write("#### Evaluation Metrics (Final Values)\n\n")
                f.write("| Metric | Local Dataset | Global Dataset |\n")
                f.write("|--------|---------------|----------------|\n")
                
                local_final = eval_metrics["local_dataset"]["pass_at_k_final"]
                global_final = eval_metrics["global_dataset"]["pass_at_k_final"]
                
                for metric in ["pass@1", "pass@5", "pass@10", "codebleu"]:
                    local_val = local_final.get(metric, "N/A")
                    global_val = global_final.get(metric, "N/A")
                    
                    local_str = f"{local_val:.3f}" if local_val != "N/A" else "N/A"
                    global_str = f"{global_val:.3f}" if global_val != "N/A" else "N/A"
                    
                    f.write(f"| {metric} | {local_str} | {global_str} |\n")
                
                f.write("\n")
            
            # Academic integrity statement
            f.write("## Academic Integrity Statement\n\n")
            f.write(report["academic_integrity_statement"] + "\n\n")
            
            f.write(f"**Report Generated**: {report['generation_timestamp']}\n")
    
    def _generate_latex_tables(self, report: Dict[str, Any], latex_dir: Path):
        """Generate LaTeX table snippets for paper inclusion"""
        
        # Table 1: Experimental Configuration
        config_table = []
        config_table.append("\\begin{table}[h]")
        config_table.append("\\centering")
        config_table.append("\\caption{KNEXA-FL Experimental Configuration}")
        config_table.append("\\label{tab:experiment_config}")
        config_table.append("\\begin{tabular}{ll}")
        config_table.append("\\toprule")
        config_table.append("Parameter & Value \\\\")
        config_table.append("\\midrule")
        
        hp = report["hyperparameters"]
        config_table.append(f"Number of Clients & {hp['num_clients']} \\\\")
        config_table.append(f"Number of Rounds & {hp['num_rounds']} \\\\")
        config_table.append(f"Local Learning Rate & {hp['learning_rates']['local']} \\\\")
        config_table.append(f"KD Learning Rate & {hp['learning_rates']['knowledge_distillation']} \\\\")
        config_table.append(f"Local Batch Size & {hp['batch_sizes']['local']} \\\\")
        config_table.append(f"KD Batch Size & {hp['batch_sizes']['knowledge_distillation']} \\\\")
        config_table.append(f"Dirichlet $\\alpha$ & {hp['federated_learning']['alpha_dirichlet']} \\\\")
        config_table.append(f"KD Temperature & {hp['federated_learning']['temperature_kd']} \\\\")
        config_table.append(f"Seed & {hp['reproducibility']['seed']} \\\\")
        
        config_table.append("\\bottomrule")
        config_table.append("\\end{tabular}")
        config_table.append("\\end{table}")
        
        with open(latex_dir / "experimental_config_table.tex", 'w') as f:
            f.write("\n".join(config_table))
        
        # Table 2: Performance Results
        if "aggregate_statistics" in report and "final_performance_averages" in report["aggregate_statistics"]:
            results_table = []
            results_table.append("\\begin{table}[h]")
            results_table.append("\\centering")
            results_table.append("\\caption{KNEXA-FL Performance Results}")
            results_table.append("\\label{tab:performance_results}")
            results_table.append("\\begin{tabular}{lcc}")
            results_table.append("\\toprule")
            results_table.append("Metric & Local Dataset & Global Dataset \\\\")
            results_table.append("\\midrule")
            
            agg = report["aggregate_statistics"]["final_performance_averages"]
            local_avg = agg["local_dataset"]
            global_avg = agg["global_dataset"]
            
            for metric in ["pass@1", "pass@5", "pass@10", "codebleu"]:
                local_val = local_avg.get(metric)
                global_val = global_avg.get(metric)
                
                local_str = f"{local_val['mean']:.3f} $\\pm$ {local_val['std']:.3f}" if local_val else "N/A"
                global_str = f"{global_val['mean']:.3f} $\\pm$ {global_val['std']:.3f}" if global_val else "N/A"
                
                metric_display = metric.replace('@', '@')  # Keep @ as is for LaTeX
                results_table.append(f"{metric_display} & {local_str} & {global_str} \\\\")
            
            results_table.append("\\bottomrule")
            results_table.append("\\end{tabular}")
            results_table.append("\\end{table}")
            
            with open(latex_dir / "performance_results_table.tex", 'w') as f:
                f.write("\n".join(results_table))
    
    def save_final_results(self, experiment_id: str, results: Dict[str, Any]):
        """Save final experiment results with comprehensive summary"""
        exp_dir = self.get_experiment_dir(experiment_id)
        
        # Save full results
        results_file = exp_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_serializable(results), f, indent=2)
        
        # Save pickle version
        pickle_file = exp_dir / "final_results.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate and save summary
        summary = self._generate_summary(results)
        summary_file = exp_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update registry
        self.registry["experiments"][experiment_id]["status"] = "completed"
        self.registry["experiments"][experiment_id]["completed"] = datetime.now().isoformat()
        self.registry["experiments"][experiment_id]["summary"] = summary
        self._save_registry()
        
        # Copy summary to central location
        central_summary = self.summary_dir / f"{experiment_id}_summary.json"
        shutil.copy(summary_file, central_summary)
        
        # Generate comprehensive plots
        self.generate_loss_plots(experiment_id)
        
        # Generate comprehensive final evaluation report
        self.generate_comprehensive_report(experiment_id)
        
        logger.info(f"Saved final results and generated comprehensive report for experiment: {experiment_id}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary from results"""
        summary = {
            "experiment_time": results.get("experiment_time", 0),
            "num_rounds": results.get("num_rounds", 0),
            "num_clients": results.get("num_clients", 0),
            "success": results.get("success", False)
        }
        
        # Extract key metrics
        if "federated_metrics" in results:
            fm = results["federated_metrics"]
            summary["final_metrics"] = {
                "global_performance": fm.get("convergence", {}).get("global_performance_history", [])[-1] if fm.get("convergence", {}).get("global_performance_history") else 0,
                "convergence_round": fm.get("convergence", {}).get("convergence_round"),
                "total_communication_mb": fm.get("communication", {}).get("total_bytes_transferred", 0) / (1024*1024),
                "average_transfer_quality": fm.get("knowledge_transfer", {}).get("average_quality", 0),
                "fairness_gini": fm.get("fairness", {}).get("performance_gini_coefficient", 0)
            }
        
        # Extract pass@k and CodeBLEU metrics if available
        if "round_results" in results:
            pass_at_k_history = {"pass@1": [], "pass@5": [], "pass@10": []}
            codebleu_history = {"codebleu_mean": [], "codebleu_max": [], "codebleu_std": []}
            
            for round_data in results["round_results"].values():
                if "strategic_pass_at_k" in round_data and "average" in round_data["strategic_pass_at_k"]:
                    avg_metrics = round_data["strategic_pass_at_k"]["average"]
                    
                    # Extract pass@k metrics
                    for k in ["pass@1", "pass@5", "pass@10"]:
                        if k in avg_metrics:
                            pass_at_k_history[k].append(avg_metrics[k])
                    
                    # Extract CodeBLEU metrics
                    for cb_metric in ["codebleu_mean", "codebleu_max", "codebleu_std"]:
                        if cb_metric in avg_metrics:
                            codebleu_history[cb_metric].append(avg_metrics[cb_metric])
            
            summary["pass_at_k_metrics"] = {
                k: {
                    "final": v[-1] if v else 0,
                    "improvement": (v[-1] - v[0]) if len(v) > 1 else 0,
                    "history": v
                }
                for k, v in pass_at_k_history.items()
            }
            
            # Add CodeBLEU metrics to summary
            summary["codebleu_metrics"] = {
                k: {
                    "final": v[-1] if v else 0,
                    "improvement": (v[-1] - v[0]) if len(v) > 1 else 0,
                    "history": v
                }
                for k, v in codebleu_history.items()
                if v  # Only include metrics that have data
            }
        
        # Model distribution
        if "model_configuration" in results:
            summary["model_distribution"] = results["model_configuration"]
        
        return summary
    
    def create_experiment_report(self, experiment_id: str):
        """Create comprehensive experiment report with visualizations"""
        exp_dir = self.get_experiment_dir(experiment_id)
        
        # Load results
        with open(exp_dir / "final_results.json", 'r') as f:
            results = json.load(f)
        
        # Create report directory
        report_dir = exp_dir / "report"
        report_dir.mkdir(exist_ok=True)
        
        # Visualization generation disabled - results saved in JSON/CSV format
        # self._create_convergence_plots(results, report_dir)
        # self._create_pass_at_k_plots(results, report_dir)
        # self._create_codebleu_plots(results, report_dir)
        # self._create_communication_plots(results, report_dir)
        # self._create_fairness_plots(results, report_dir)
        logger.info("Plot generation disabled - all results saved in JSON/CSV format")
        
        # Generate LaTeX report
        self._generate_latex_report(experiment_id, results, report_dir)
        
        # Generate markdown report
        self._generate_markdown_report(experiment_id, results, report_dir)
        
        logger.info(f"Created comprehensive report for experiment: {experiment_id}")
    
    '''def _create_convergence_plots(self, results: Dict[str, Any], output_dir: Path):
        """Create convergence visualizations"""
        if "federated_metrics" not in results:
            return
        
        fm = results["federated_metrics"]
        
        # Global performance convergence
        if "convergence" in fm and "global_performance_history" in fm["convergence"]:
            plt.figure(figsize=(10, 6))
            history = fm["convergence"]["global_performance_history"]
            plt.plot(history, 'b-', linewidth=2, label='Global Performance')
            plt.axhline(y=history[-1], color='r', linestyle='--', alpha=0.5, label=f'Final: {history[-1]:.3f}')
            plt.xlabel('Round')
            plt.ylabel('Performance (Pass@1)')
            plt.title('Global Model Performance Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'convergence_global.png', dpi=300)
            plt.close()'''
    
    '''def _create_pass_at_k_plots(self, results: Dict[str, Any], output_dir: Path):
        """Create pass@k visualizations"""
        # Extract pass@k data from round results
        pass_at_k_data = {"pass@1": [], "pass@5": [], "pass@10": []}
        rounds = []
        
        if "round_results" in results:
            for round_id, round_data in sorted(results["round_results"].items()):
                if "strategic_pass_at_k" in round_data and "average" in round_data["strategic_pass_at_k"]:
                    rounds.append(int(round_id))
                    avg_metrics = round_data["strategic_pass_at_k"]["average"]
                    for k in ["pass@1", "pass@5", "pass@10"]:
                        pass_at_k_data[k].append(avg_metrics.get(k, 0))
        
        if rounds:
            plt.figure(figsize=(10, 6))
            for metric, values in pass_at_k_data.items():
                if values:
                    plt.plot(rounds, values, marker='o', linewidth=2, label=metric)
            
            plt.xlabel('Round')
            plt.ylabel('Pass@k Score')
            plt.title('Code Generation Performance (Pass@k) Over Rounds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'pass_at_k_evolution.png', dpi=300)
            plt.close()'''
    
    '''def _create_codebleu_plots(self, results: Dict[str, Any], output_dir: Path):
        """Create CodeBLEU visualizations"""
        # Extract CodeBLEU data from round results
        codebleu_data = {"codebleu_mean": [], "codebleu_max": [], "codebleu_std": []}
        rounds = []
        
        if "round_results" in results:
            for round_id, round_data in sorted(results["round_results"].items()):
                if "strategic_pass_at_k" in round_data and "average" in round_data["strategic_pass_at_k"]:
                    avg_metrics = round_data["strategic_pass_at_k"]["average"]
                    # Check if any CodeBLEU metrics are present
                    if any(cb_metric in avg_metrics for cb_metric in codebleu_data.keys()):
                        rounds.append(int(round_id))
                        for cb_metric in codebleu_data.keys():
                            codebleu_data[cb_metric].append(avg_metrics.get(cb_metric, 0))
        
        if rounds and any(codebleu_data.values()):
            plt.figure(figsize=(10, 6))
            
            # Plot mean with error bars using std
            if codebleu_data["codebleu_mean"] and codebleu_data["codebleu_std"]:
                means = codebleu_data["codebleu_mean"]
                stds = codebleu_data["codebleu_std"]
                plt.errorbar(rounds, means, yerr=stds, marker='o', linewidth=2, 
                           capsize=5, label='CodeBLEU Mean ± Std')
            
            # Plot max values
            if codebleu_data["codebleu_max"]:
                plt.plot(rounds, codebleu_data["codebleu_max"], marker='s', 
                        linewidth=2, linestyle='--', alpha=0.7, label='CodeBLEU Max')
            
            plt.xlabel('Round')
            plt.ylabel('CodeBLEU Score')
            plt.title('Code Similarity (CodeBLEU) Over Rounds')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)  # CodeBLEU scores are between 0 and 1
            plt.tight_layout()
            plt.savefig(output_dir / 'codebleu_evolution.png', dpi=300)
            plt.close()'''
    
    '''def _create_communication_plots(self, results: Dict[str, Any], output_dir: Path):
        """Create communication efficiency visualizations"""
        if "federated_metrics" not in results or "communication" not in results["federated_metrics"]:
            return
        
        comm = results["federated_metrics"]["communication"]
        
        # Communication pattern heatmap
        if "transfer_matrix" in comm:
            matrix = np.array(comm["transfer_matrix"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Blues')
            plt.xlabel('To Client')
            plt.ylabel('From Client')
            plt.title('Knowledge Transfer Communication Pattern')
            plt.tight_layout()
            plt.savefig(output_dir / 'communication_pattern.png', dpi=300)
            plt.close()'''
    
    '''def _create_fairness_plots(self, results: Dict[str, Any], output_dir: Path):
        """Create fairness visualizations"""
        if "federated_metrics" not in results or "fairness" not in results["federated_metrics"]:
            return
        
        fairness = results["federated_metrics"]["fairness"]
        
        # Client performance distribution
        if "client_performance_history" in fairness:
            history = fairness["client_performance_history"]
            num_clients = len(history)
            
            plt.figure(figsize=(10, 6))
            for client_id in range(num_clients):
                client_perf = [h[client_id] for h in history if client_id < len(h)]
                plt.plot(client_perf, linewidth=2, label=f'Client {client_id}')
            
            plt.xlabel('Round')
            plt.ylabel('Performance')
            plt.title('Per-Client Performance Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'client_performance_evolution.png', dpi=300)
            plt.close()'''
    
    def _generate_latex_report(self, experiment_id: str, results: Dict[str, Any], output_dir: Path):
        """Generate LaTeX report for paper inclusion"""
        latex_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}

\\title{{KNEXA-FL Experiment Report}}
\\author{{Experiment ID: {experiment_id}}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\section{{Summary}}
Experiment completed in {results.get('experiment_time', 0):.1f} seconds with {results.get('num_rounds', 0)} rounds and {results.get('num_clients', 0)} clients.

\\section{{Key Results}}
\\begin{{itemize}}
"""
        
        if "summary" in results:
            summary = results["summary"]
            if "final_metrics" in summary:
                fm = summary["final_metrics"]
                latex_content += f"\\item Final global performance: {fm.get('global_performance', 0):.3f}\n"
                latex_content += f"\\item Convergence round: {fm.get('convergence_round', 'N/A')}\n"
                latex_content += f"\\item Total communication: {fm.get('total_communication_mb', 0):.1f} MB\n"
                latex_content += f"\\item Fairness (Gini): {fm.get('fairness_gini', 0):.3f}\n"
        
        latex_content += """\\end{itemize}

\\section{Visualizations}
\\begin{figure}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{convergence_global.png}
\\caption{Global performance convergence}
\\end{figure}

\\end{document}
"""
        
        with open(output_dir / 'report.tex', 'w') as f:
            f.write(latex_content)
    
    def _generate_markdown_report(self, experiment_id: str, results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive markdown report"""
        config = self.registry["experiments"][experiment_id]["config"]
        
        md_content = f"""# KNEXA-FL Experiment Report

**Experiment ID**: `{experiment_id}`  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status**: {self.registry["experiments"][experiment_id]["status"]}

## Configuration

| Parameter | Value |
|-----------|-------|
| Method | {config.get('method', 'KNEXA-FL')} |
| Clients | {config.get('num_clients', 4)} |
| Rounds | {config.get('num_rounds', 25)} |
| Learning Rate (Local) | {config.get('learning_rate_local', 5e-5)} |
| Learning Rate (KD) | {config.get('learning_rate_kd', 1e-4)} |
| Batch Size (Local) | {config.get('batch_size_local', 16)} |
| Batch Size (KD) | {config.get('batch_size_kd', 8)} |
| Dirichlet α | {config.get('alpha_dirichlet', 0.1)} |
| KD Temperature | {config.get('temperature_kd', 2.0)} |
| KD α | {config.get('alpha_kd', 0.5)} |

## Results Summary
"""
        
        if "summary" in results:
            summary = results["summary"]
            md_content += f"""
### Performance Metrics
- **Experiment Duration**: {summary.get('experiment_time', 0):.1f} seconds
- **Success**: {'✅ Yes' if summary.get('success', False) else '❌ No'}
"""
            
            if "final_metrics" in summary:
                fm = summary["final_metrics"]
                md_content += f"""
### Final Metrics
- **Global Performance**: {fm.get('global_performance', 0):.3f}
- **Convergence Round**: {fm.get('convergence_round', 'N/A')}
- **Total Communication**: {fm.get('total_communication_mb', 0):.1f} MB
- **Average Transfer Quality**: {fm.get('average_transfer_quality', 0):.3f}
- **Fairness (Gini)**: {fm.get('fairness_gini', 0):.3f}
"""
            
            if "pass_at_k_metrics" in summary:
                pk = summary["pass_at_k_metrics"]
                md_content += f"""
### Code Generation Performance (Pass@k)
| Metric | Final Score | Improvement |
|--------|------------|-------------|
| Pass@1 | {pk.get('pass@1', {}).get('final', 0):.3f} | {pk.get('pass@1', {}).get('improvement', 0):+.3f} |
| Pass@5 | {pk.get('pass@5', {}).get('final', 0):.3f} | {pk.get('pass@5', {}).get('improvement', 0):+.3f} |
| Pass@10 | {pk.get('pass@10', {}).get('final', 0):.3f} | {pk.get('pass@10', {}).get('improvement', 0):+.3f} |
"""
            
            if "codebleu_metrics" in summary:
                cb = summary["codebleu_metrics"]
                md_content += f"""
### Code Similarity (CodeBLEU)
| Metric | Final Score | Improvement |
|--------|------------|-------------|
| CodeBLEU Mean | {cb.get('codebleu_mean', {}).get('final', 0):.3f} | {cb.get('codebleu_mean', {}).get('improvement', 0):+.3f} |
| CodeBLEU Max | {cb.get('codebleu_max', {}).get('final', 0):.3f} | {cb.get('codebleu_max', {}).get('improvement', 0):+.3f} |
| CodeBLEU Std | {cb.get('codebleu_std', {}).get('final', 0):.3f} | {cb.get('codebleu_std', {}).get('improvement', 0):+.3f} |
"""
        
        md_content += """
## Visualizations

### Global Performance Convergence
![Global Performance](convergence_global.png)

### Pass@k Evolution
![Pass@k Evolution](pass_at_k_evolution.png)

### CodeBLEU Evolution
![CodeBLEU Evolution](codebleu_evolution.png)

### Communication Pattern
![Communication Pattern](communication_pattern.png)

### Client Performance Evolution
![Client Performance](client_performance_evolution.png)

## Files Generated

- `config.yaml`: Experiment configuration
- `system_info.json`: System information for reproducibility
- `final_results.json`: Complete results in JSON format
- `final_results.pkl`: Complete results in pickle format
- `experiment_summary.json`: Condensed summary
- `round_results/`: Per-round detailed results
- `logs/`: Training logs
- `code_generation/`: Pass@k code generation logs
- `report/`: This report and visualizations

---
*Generated by KNEXA-FL Experiment Manager*
"""
        
        with open(output_dir / 'report.md', 'w') as f:
            f.write(md_content)
    
    def compare_experiments(self, experiment_ids: List[str], output_dir: Optional[str] = None):
        """Compare multiple experiments"""
        if output_dir is None:
            output_dir = self.summary_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load all experiment summaries
        experiments = []
        for exp_id in experiment_ids:
            summary_file = self.summary_dir / f"{exp_id}_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    summary['experiment_id'] = exp_id
                    experiments.append(summary)
        
        # Create comparison table
        comparison_data = []
        for exp in experiments:
            row = {
                'Experiment': exp['experiment_id'][:20] + '...',
                'Duration (s)': exp.get('experiment_time', 0),
                'Rounds': exp.get('num_rounds', 0),
                'Clients': exp.get('num_clients', 0)
            }
            
            if 'final_metrics' in exp:
                fm = exp['final_metrics']
                row.update({
                    'Final Performance': fm.get('global_performance', 0),
                    'Convergence Round': fm.get('convergence_round', 'N/A'),
                    'Communication (MB)': fm.get('total_communication_mb', 0),
                    'Fairness (Gini)': fm.get('fairness_gini', 0)
                })
            
            if 'pass_at_k_metrics' in exp:
                pk = exp['pass_at_k_metrics']
                row.update({
                    'Pass@1': pk.get('pass@1', {}).get('final', 0),
                    'Pass@5': pk.get('pass@5', {}).get('final', 0),
                    'Pass@10': pk.get('pass@10', {}).get('final', 0)
                })
            
            if 'codebleu_metrics' in exp:
                cb = exp['codebleu_metrics']
                row.update({
                    'CodeBLEU Mean': cb.get('codebleu_mean', {}).get('final', 0),
                    'CodeBLEU Max': cb.get('codebleu_max', {}).get('final', 0)
                })
            
            comparison_data.append(row)
        
        # Save comparison table
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_dir / 'comparison_table.csv', index=False)
        
        # Plot generation disabled
        # self._create_comparison_plots(experiments, output_dir)
        logger.info("Comparison plots disabled - results saved in CSV format")
        
        logger.info(f"Created experiment comparison in: {output_dir}")
        
        return df
    
    def _create_comparison_plots(self, experiments: List[Dict], output_dir: Path):
        '''
        """Create comparison visualizations"""
        # Performance comparison bar chart
        plt.figure(figsize=(12, 6))
        
        exp_names = [exp['experiment_id'][:15] + '...' for exp in experiments]
        performances = [exp.get('final_metrics', {}).get('global_performance', 0) for exp in experiments]
        
        plt.bar(exp_names, performances)
        plt.xlabel('Experiment')
        plt.ylabel('Final Performance')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300)
        plt.close()
        
        # Pass@k comparison
        if any('pass_at_k_metrics' in exp for exp in experiments):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, k in enumerate(['pass@1', 'pass@5', 'pass@10']):
                values = [exp.get('pass_at_k_metrics', {}).get(k, {}).get('final', 0) for exp in experiments]
                axes[i].bar(exp_names, values)
                axes[i].set_xlabel('Experiment')
                axes[i].set_ylabel(f'{k} Score')
                axes[i].set_title(f'{k} Comparison')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'pass_at_k_comparison.png', dpi=300)
            plt.close()
        
        # CodeBLEU comparison
        if any('codebleu_metrics' in exp for exp in experiments):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            for i, metric in enumerate(['codebleu_mean', 'codebleu_max']):
                values = [exp.get('codebleu_metrics', {}).get(metric, {}).get('final', 0) for exp in experiments]
                axes[i].bar(exp_names, values)
                axes[i].set_xlabel('Experiment')
                axes[i].set_ylabel(f'{metric.replace("_", " ").title()} Score')
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].set_ylim(0, 1)  # CodeBLEU scores are between 0 and 1
            
            plt.tight_layout()
            plt.savefig(output_dir / 'codebleu_comparison.png', dpi=300)
            plt.close()'''
        pass  # Plot generation disabled
    
    def get_best_experiment(self, metric: str = "global_performance") -> Optional[str]:
        """Find best experiment by metric"""
        best_id = None
        best_value = -float('inf')
        
        for exp_id, exp_info in self.registry["experiments"].items():
            if exp_info["status"] == "completed" and "summary" in exp_info:
                summary = exp_info["summary"]
                
                # Extract metric value
                value = None
                if metric == "global_performance":
                    value = summary.get("final_metrics", {}).get("global_performance", -float('inf'))
                elif metric.startswith("pass@"):
                    value = summary.get("pass_at_k_metrics", {}).get(metric, {}).get("final", -float('inf'))
                elif metric.startswith("codebleu"):
                    value = summary.get("codebleu_metrics", {}).get(metric, {}).get("final", -float('inf'))
                
                if value is not None and value > best_value:
                    best_value = value
                    best_id = exp_id
        
        return best_id
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj


# Convenience functions
def create_experiment(config: Union[Dict[str, Any], ExperimentConfig], base_dir: str = "experimental_artifacts/knexa_fl") -> str:
    """Create new experiment with config"""
    manager = ExperimentManager(base_dir)
    
    if isinstance(config, dict):
        config = ExperimentConfig(**config)
    
    return manager.create_experiment(config)


def save_experiment_results(experiment_id: str, results: Dict[str, Any], base_dir: str = "experimental_artifacts/knexa_fl"):
    """Save experiment results"""
    manager = ExperimentManager(base_dir)
    manager.save_final_results(experiment_id, results)


def generate_experiment_report(experiment_id: str, base_dir: str = "experimental_artifacts/knexa_fl"):
    """Generate comprehensive report for experiment"""
    manager = ExperimentManager(base_dir)
    manager.create_experiment_report(experiment_id)
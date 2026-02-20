#!/usr/bin/env python3
"""
FedAvg Experiment Result Management System
Comprehensive research-grade result tracking and reporting for FedAvg baseline

Author: Inderjeet Singh
Academic Standard: Publication-ready experimental documentation
"""

import json
import yaml
import pickle
import shutil
import logging
import hashlib
import subprocess
import psutil
import platform
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import torch
import sys
import os

logger = logging.getLogger(__name__)

@dataclass
class FedAvgConfig:
    """Configuration for FedAvg experiment run"""
    experiment_name: str
    algorithm: str = "FedAvg"
    reference: str = "McMahan et al., 2017"
    num_clients: int = 4
    num_rounds: int = 30
    learning_rate_local: float = 1e-4
    local_epochs: int = 1
    batch_size_local: int = 16
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 4
    min_evaluate_clients: int = 4
    model_configuration: str = "stable_vram_cached"
    gpu_resources: float = 0.25
    cpu_resources: float = 1.0
    timeout_seconds: int = 120
    seed: int = 42
    quick_mode: bool = False
    verbose: bool = False
    additional_params: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.additional_params:
            data.update(self.additional_params)
        return data
    
    def get_hash(self) -> str:
        """Generate unique hash for this configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

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
    environment: str
    working_directory: str
    
    @classmethod
    def capture(cls) -> 'SystemInfo':
        """Capture current system information"""
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
        
        # Environment info
        environment = os.getenv('VIRTUAL_ENV', 'unknown')
        working_directory = os.getcwd()
        
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
            git_dirty=git_dirty,
            environment=environment,
            working_directory=working_directory
        )

@dataclass
class DatasetInfo:
    """Information about the dataset used in experiment"""
    dataset_name: str
    train_samples_per_client: List[int]
    validation_samples_per_client: List[int]
    total_train_samples: int
    total_validation_samples: int
    data_distribution: str = "non-IID"
    task_type: str = "code_generation"
    evaluation_metrics: List[str] = None
    
    @classmethod
    def from_splits(cls, client_splits, global_test=None):
        """Create dataset info from client splits"""
        train_samples = []
        val_samples = []
        
        for train_ds, val_ds in client_splits:
            train_samples.append(len(train_ds))
            val_samples.append(len(val_ds))
        
        return cls(
            dataset_name="HumanEval + MBPP (Code Generation)",
            train_samples_per_client=train_samples,
            validation_samples_per_client=val_samples,
            total_train_samples=sum(train_samples),
            total_validation_samples=sum(val_samples),
            data_distribution="non-IID (heterogeneous)",
            task_type="code_generation",
            evaluation_metrics=["pass@1", "code_quality", "functional_correctness"]
        )

@dataclass  
class ModelInfo:
    """Information about models used in experiment"""
    model_configuration: str
    models_per_client: Dict[int, Dict[str, Any]]
    heterogeneous: bool = True
    parameter_efficient: bool = True
    fine_tuning_method: str = "LoRA/DoRA"
    
    @classmethod
    def from_config(cls, model_config: str, num_clients: int):
        """Create model info from configuration"""
        from src.globals import MODEL_MAP, LLM_REGISTRY, HETEROGENEOUS_CONFIGS
        
        # Get the specific configuration
        if model_config in HETEROGENEOUS_CONFIGS:
            config_models = HETEROGENEOUS_CONFIGS[model_config]
        else:
            config_models = MODEL_MAP
            
        models_per_client = {}
        for cid in range(num_clients):
            model_name = config_models[cid % len(config_models)]
            model_info = LLM_REGISTRY.get(model_name, {"params": "unknown", "arch": "unknown"})
            
            models_per_client[cid] = {
                "model_name": model_name,
                "parameters": model_info.get("params", "unknown"),
                "architecture": model_info.get("arch", "unknown"),
                "type": model_info.get("type", "decoder"),
                "license": model_info.get("license", "unknown")
            }
        
        return cls(
            model_configuration=model_config,
            models_per_client=models_per_client,
            heterogeneous=len(set(config_models.values())) > 1,
            parameter_efficient=True,
            fine_tuning_method="LoRA/DoRA"
        )

class FedAvgExperimentManager:
    """
    Comprehensive FedAvg experiment management system
    Handles result storage, organization, and publication-ready reporting
    """
    
    def __init__(self, base_dir: str = "experimental_artifacts/baselines/fedavg/results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Directory structure
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        self.summary_dir = self.base_dir / "summaries"
        self.summary_dir.mkdir(exist_ok=True)
        
        self.plots_dir = self.base_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.base_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load experiment registry
        self.registry_file = self.base_dir / "fedavg_registry.json"
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load or create FedAvg experiment registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {"fedavg_experiments": {}}
    
    def _save_registry(self):
        """Save experiment registry"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def start_experiment(self, config: FedAvgConfig, client_splits=None, global_test=None) -> str:
        """Start a new FedAvg experiment and return experiment ID"""
        
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.get_hash()
        experiment_id = f"FedAvg_{timestamp}_{config_hash}"
        
        # Create experiment directory
        exp_dir = self.runs_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "metrics").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "round_results").mkdir(exist_ok=True)
        
        # Capture system info
        system_info = SystemInfo.capture()
        
        # Capture dataset info
        dataset_info = DatasetInfo.from_splits(client_splits, global_test) if client_splits else None
        
        # Capture model info
        model_info = ModelInfo.from_config(config.model_configuration, config.num_clients)
        
        # Save experiment metadata
        metadata = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "config": config.to_dict(),
            "system_info": asdict(system_info),
            "dataset_info": asdict(dataset_info) if dataset_info else None,
            "model_info": asdict(model_info),
            "status": "running"
        }
        
        with open(exp_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save configuration
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        # Save system info
        with open(exp_dir / "system_info.json", "w") as f:
            json.dump(asdict(system_info), f, indent=2)
        
        # Create README for experiment
        self._create_experiment_readme(exp_dir, config, system_info, dataset_info, model_info)
        
        # Update registry
        self.registry["fedavg_experiments"][experiment_id] = {
            "config": config.to_dict(),
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "directory": str(exp_dir)
        }
        self._save_registry()
        
        logger.info(f"Started FedAvg experiment: {experiment_id}")
        return experiment_id
    
    def _create_experiment_readme(self, exp_dir: Path, config: FedAvgConfig, 
                                system_info: SystemInfo, dataset_info: DatasetInfo, 
                                model_info: ModelInfo):
        """Create comprehensive README for experiment"""
        
        readme_content = f"""# FedAvg Experiment: {config.experiment_name}

## Overview
This experiment implements the FedAvg (Federated Averaging) baseline algorithm as described in McMahan et al. 2017: "Communication-Efficient Learning of Deep Networks from Decentralized Data".

## Experiment Configuration

### Algorithm Details
- **Algorithm**: {config.algorithm}
- **Reference**: {config.reference}
- **Experiment Name**: {config.experiment_name}
- **Number of Rounds**: {config.num_rounds}
- **Number of Clients**: {config.num_clients}

### Hyperparameters
- **Local Learning Rate**: {config.learning_rate_local}
- **Local Epochs per Round**: {config.local_epochs}
- **Batch Size (Local)**: {config.batch_size_local}
- **Fraction of Clients for Training**: {config.fraction_fit}
- **Fraction of Clients for Evaluation**: {config.fraction_evaluate}
- **Minimum Clients for Training**: {config.min_fit_clients}
- **Minimum Clients for Evaluation**: {config.min_evaluate_clients}
- **Random Seed**: {config.seed}

### Model Configuration
- **Configuration Name**: {config.model_configuration}
- **Heterogeneous Models**: {model_info.heterogeneous}
- **Parameter-Efficient Fine-Tuning**: {model_info.parameter_efficient}
- **Fine-Tuning Method**: {model_info.fine_tuning_method}

### Models per Client
"""
        
        for cid, model_data in model_info.models_per_client.items():
            readme_content += f"- **Client {cid}**: {model_data['model_name']} ({model_data['parameters']}, {model_data['architecture']})\n"
        
        readme_content += f"""
### Dataset Information
- **Dataset**: {dataset_info.dataset_name if dataset_info else 'Unknown'}
- **Task Type**: {dataset_info.task_type if dataset_info else 'Unknown'}
- **Data Distribution**: {dataset_info.data_distribution if dataset_info else 'Unknown'}
- **Total Training Samples**: {dataset_info.total_train_samples if dataset_info else 'Unknown'}
- **Total Validation Samples**: {dataset_info.total_validation_samples if dataset_info else 'Unknown'}

### Training Samples per Client
"""
        
        if dataset_info:
            for i, samples in enumerate(dataset_info.train_samples_per_client):
                readme_content += f"- **Client {i}**: {samples} training samples\n"
        
        readme_content += f"""
### System Information
- **Python Version**: {system_info.python_version}
- **PyTorch Version**: {system_info.torch_version}
- **CUDA Version**: {system_info.cuda_version}
- **GPU(s)**: {', '.join(system_info.gpu_info) if system_info.gpu_info else 'None'}
- **CPU**: {system_info.cpu_info}
- **Memory**: {system_info.memory_gb:.1f} GB
- **Git Commit**: {system_info.git_commit}
- **Git Branch**: {system_info.git_branch}
- **Git Status**: {'Clean' if not system_info.git_dirty else 'Modified'}

### Resource Allocation
- **GPU Resources per Client**: {config.gpu_resources}
- **CPU Resources per Client**: {config.cpu_resources}
- **Round Timeout**: {config.timeout_seconds} seconds

## Files Structure
- `experiment_metadata.json`: Complete experiment metadata
- `config.yaml`: Experiment configuration
- `system_info.json`: System information for reproducibility
- `checkpoints/`: Model checkpoints and states
- `logs/`: Training and execution logs  
- `metrics/`: Performance metrics and measurements
- `plots/`: Visualization and analysis plots
- `round_results/`: Per-round detailed results
- `final_report.md`: Comprehensive final report

## Reproducibility
This experiment is designed for full reproducibility. All hyperparameters, system information, random seeds, and model configurations are logged. To reproduce:

1. Use the same system configuration or similar hardware
2. Install dependencies matching the versions in `system_info.json`
3. Use the configuration in `config.yaml`
4. Run with the same random seed: {config.seed}

## Author
**Inderjeet Singh**  
Implementation follows academic standards for research publication.

## Citation
If using this baseline in research, please cite:
```
McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. Y. (2017). 
Communication-efficient learning of deep networks from decentralized data. 
Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS).
```
"""
        
        with open(exp_dir / "README.md", "w") as f:
            f.write(readme_content)
    
    def log_round_results(self, experiment_id: str, round_num: int, results: Dict[str, Any]):
        """Log results for a specific round"""
        exp_dir = self.runs_dir / experiment_id
        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {experiment_id}")
            return
        
        # Save round results
        round_file = exp_dir / "round_results" / f"round_{round_num:03d}.json"
        with open(round_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save as pickle for complex objects
        round_pickle = exp_dir / "round_results" / f"round_{round_num:03d}.pkl"
        with open(round_pickle, "wb") as f:
            pickle.dump(results, f)
    
    def complete_experiment(self, experiment_id: str, final_results: Dict[str, Any]):
        """Mark experiment as complete and generate final report"""
        exp_dir = self.runs_dir / experiment_id
        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {experiment_id}")
            return
        
        # Update metadata
        metadata_file = exp_dir / "experiment_metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        metadata["end_time"] = datetime.now().isoformat()
        metadata["status"] = "completed"
        metadata["final_results"] = final_results
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save final results
        with open(exp_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        with open(exp_dir / "final_results.pkl", "wb") as f:
            pickle.dump(final_results, f)
        
        # Generate comprehensive final report
        self._generate_final_report(experiment_id, metadata, final_results)
        
        # Update registry
        if experiment_id in self.registry["fedavg_experiments"]:
            self.registry["fedavg_experiments"][experiment_id]["status"] = "completed"
            self.registry["fedavg_experiments"][experiment_id]["end_time"] = datetime.now().isoformat()
            self.registry["fedavg_experiments"][experiment_id]["final_results"] = final_results
        self._save_registry()
        
        logger.info(f"Completed FedAvg experiment: {experiment_id}")
        
        # Generate summary for registry
        self._generate_experiment_summary(experiment_id)
    
    def _generate_final_report(self, experiment_id: str, metadata: Dict[str, Any], 
                             final_results: Dict[str, Any]):
        """Generate comprehensive final experiment report"""
        exp_dir = self.runs_dir / experiment_id
        
        # Collect all round results
        round_results = []
        round_dir = exp_dir / "round_results"
        if round_dir.exists():
            for round_file in sorted(round_dir.glob("round_*.json")):
                try:
                    with open(round_file, "r") as f:
                        round_data = json.load(f)
                        round_results.append(round_data)
                except Exception as e:
                    logger.warning(f"Failed to load round results from {round_file}: {e}")
        
        # Calculate experiment statistics
        stats = self._calculate_experiment_statistics(round_results, final_results)
        
        # Generate report
        report_content = f"""# FedAvg Experiment Final Report

## Experiment Overview
- **Experiment ID**: {experiment_id}
- **Algorithm**: {metadata['config']['algorithm']}
- **Reference**: {metadata['config']['reference']}
- **Start Time**: {metadata['start_time']}
- **End Time**: {metadata['end_time']}
- **Duration**: {self._calculate_duration(metadata['start_time'], metadata['end_time'])}
- **Status**: {metadata['status']}

## Configuration Summary
- **Rounds Completed**: {metadata['config']['num_rounds']}
- **Number of Clients**: {metadata['config']['num_clients']}
- **Local Learning Rate**: {metadata['config']['learning_rate_local']}
- **Local Epochs**: {metadata['config']['local_epochs']}
- **Batch Size**: {metadata['config']['batch_size_local']}
- **Random Seed**: {metadata['config']['seed']}
- **Model Configuration**: {metadata['config']['model_configuration']}

## Dataset Information
"""
        
        if metadata.get('dataset_info'):
            dataset = metadata['dataset_info']
            report_content += f"""- **Dataset**: {dataset['dataset_name']}
- **Task Type**: {dataset['task_type']}
- **Total Training Samples**: {dataset['total_train_samples']}
- **Total Validation Samples**: {dataset['total_validation_samples']}
- **Data Distribution**: {dataset['data_distribution']}
"""
        
        report_content += f"""
## Model Architecture Details
"""
        
        if metadata.get('model_info'):
            model_info = metadata['model_info']
            report_content += f"""- **Configuration**: {model_info['model_configuration']}
- **Heterogeneous**: {model_info['heterogeneous']}
- **Parameter Efficient**: {model_info['parameter_efficient']}
- **Fine-tuning Method**: {model_info['fine_tuning_method']}

### Models per Client:
"""
            for cid, model_data in model_info['models_per_client'].items():
                report_content += f"- **Client {cid}**: {model_data['model_name']} ({model_data['parameters']}, {model_data['architecture']})\n"
        
        report_content += f"""
## Performance Results

### Final Performance Metrics
"""
        
        if final_results:
            for metric, value in final_results.items():
                if isinstance(value, (int, float)):
                    report_content += f"- **{metric}**: {value:.4f}\n"
                else:
                    report_content += f"- **{metric}**: {value}\n"
        
        if stats:
            report_content += f"""
### Statistical Summary
- **Best Round Performance**: Round {stats.get('best_round', 'N/A')} ({stats.get('best_performance', 'N/A'):.4f})
- **Final Round Performance**: {stats.get('final_performance', 'N/A'):.4f}
- **Average Performance**: {stats.get('avg_performance', 'N/A'):.4f}
- **Performance Improvement**: {stats.get('total_improvement', 'N/A'):.4f}
- **Convergence**: {stats.get('converged', 'Unknown')}
"""
        
        report_content += f"""
## System Information
- **Python**: {metadata['system_info']['python_version']}
- **PyTorch**: {metadata['system_info']['torch_version']}
- **CUDA**: {metadata['system_info']['cuda_version']}
- **GPU(s)**: {', '.join(metadata['system_info']['gpu_info'])}
- **Memory**: {metadata['system_info']['memory_gb']:.1f} GB
- **Git Commit**: {metadata['system_info']['git_commit'][:8]}
- **Environment**: {metadata['system_info']['environment']}

## Reproducibility Information
All experiment parameters, random seeds, and system configurations have been logged for full reproducibility. 

### Key Files:
- `experiment_metadata.json`: Complete experimental setup
- `config.yaml`: Hyperparameter configuration
- `final_results.json`: Final performance metrics
- `round_results/`: Per-round detailed results
- `system_info.json`: System configuration for reproducibility

## Notes
This experiment implements the classical FedAvg algorithm with weighted parameter averaging. Results are suitable for comparison with other federated learning methods and for publication in academic venues.

## Author
**Inderjeet Singh**

---
Report generated on: {datetime.now().isoformat()}
"""
        
        # Save report
        with open(exp_dir / "final_report.md", "w") as f:
            f.write(report_content)
        
        # Also save to reports directory
        report_file = self.reports_dir / f"{experiment_id}_report.md"
        with open(report_file, "w") as f:
            f.write(report_content)
        
        logger.info(f"Generated final report: {report_file}")
    
    def _calculate_experiment_statistics(self, round_results: List[Dict], 
                                       final_results: Dict) -> Dict[str, Any]:
        """Calculate experiment statistics from round results"""
        if not round_results:
            return {}
        
        # Extract performance metrics
        performances = []
        for round_data in round_results:
            if 'avg_pass@1' in round_data:
                performances.append(round_data['avg_pass@1'])
            elif 'average_performance' in round_data:
                performances.append(round_data['average_performance'])
        
        if not performances:
            return {}
        
        stats = {
            'best_performance': max(performances),
            'best_round': performances.index(max(performances)),
            'final_performance': performances[-1],
            'avg_performance': sum(performances) / len(performances),
            'total_improvement': performances[-1] - performances[0],
            'converged': abs(performances[-1] - performances[-2]) < 0.001 if len(performances) > 1 else False
        }
        
        return stats
    
    def _calculate_duration(self, start_time: str, end_time: str) -> str:
        """Calculate experiment duration"""
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            return f"{duration.days}d {hours}h {minutes}m"
        except:
            return "Unknown"
    
    def _generate_experiment_summary(self, experiment_id: str):
        """Generate experiment summary for quick reference"""
        exp_dir = self.runs_dir / experiment_id
        
        try:
            with open(exp_dir / "experiment_metadata.json", "r") as f:
                metadata = json.load(f)
            
            with open(exp_dir / "final_results.json", "r") as f:
                final_results = json.load(f)
            
            summary = {
                "experiment_id": experiment_id,
                "algorithm": metadata['config']['algorithm'],
                "start_time": metadata['start_time'],
                "end_time": metadata.get('end_time'),
                "duration": self._calculate_duration(
                    metadata['start_time'], 
                    metadata.get('end_time', metadata['start_time'])
                ),
                "config": {
                    "rounds": metadata['config']['num_rounds'],
                    "clients": metadata['config']['num_clients'],
                    "lr_local": metadata['config']['learning_rate_local'],
                    "model_config": metadata['config']['model_configuration'],
                    "seed": metadata['config']['seed']
                },
                "final_results": final_results,
                "status": metadata['status']
            }
            
            summary_file = self.summary_dir / f"{experiment_id}_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Generated experiment summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary for {experiment_id}: {e}")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all FedAvg experiments"""
        experiments = []
        for exp_id, exp_data in self.registry["fedavg_experiments"].items():
            experiments.append({
                "id": exp_id,
                **exp_data
            })
        return experiments
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get specific experiment details"""
        return self.registry["fedavg_experiments"].get(experiment_id)

# Global instance for easy access
fedavg_experiment_manager = FedAvgExperimentManager()
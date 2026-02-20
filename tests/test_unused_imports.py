#!/usr/bin/env python3
"""
Import Validation Test for KNEXA-FL
Ensures no code imports from deleted files listed in Section A
"""

import ast
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_no_imports_from_deleted_files():
    """Ensure no code imports from files listed in Section A"""
    deleted_files = [
        'fl_benchmarking', 'fedavg_results_analyzer', 
        'session_orchestrator', 'global_pairing_coordinator',
        'p2p_service'
    ]
    
    src_files = Path('src').rglob('*.py')
    violations = []
    
    for file_path in src_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and any(deleted in node.module for deleted in deleted_files):
                        violations.append(f"File {file_path} imports from deleted module: {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(deleted in alias.name for deleted in deleted_files):
                            violations.append(f"File {file_path} imports deleted module: {alias.name}")
        except Exception as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    
    if violations:
        print("\n❌ Import violations detected:")
        for violation in violations:
            print(f"  - {violation}")
        raise AssertionError(f"{len(violations)} import violations found")
    else:
        print("✅ No imports from deleted files detected")


def test_required_imports_exist():
    """Ensure all required core modules can be imported"""
    required_imports = [
        ("src.main_p2p_real", "RealFederatedLearning"),
        ("src.client", "KnexaClient"),
        ("src.experiment_manager", "ExperimentManager"),
        ("src.structured_logging", "get_structured_logger"),
        ("src.utils.file_checksums", "sha256_file"),
        ("src.grpc_p2p.knowledge_distillation", "AdaptiveKnowledgeDistillation"),
        ("src.bandit", "LinUCB"),
    ]
    
    failed_imports = []
    
    for module_name, class_name in required_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if not hasattr(module, class_name):
                failed_imports.append(f"{module_name}.{class_name} not found")
            else:
                print(f"✅ Successfully imported {module_name}.{class_name}")
        except ImportError as e:
            failed_imports.append(f"Failed to import {module_name}: {e}")
    
    if failed_imports:
        print("\n❌ Import failures:")
        for failure in failed_imports:
            print(f"  - {failure}")
        raise AssertionError(f"{len(failed_imports)} import failures")
    else:
        print("\n✅ All required imports successful")


if __name__ == "__main__":
    print("="*60)
    print("KNEXA-FL Import Validation Test")
    print("="*60)
    
    print("\n1. Testing for imports from deleted files...")
    test_no_imports_from_deleted_files()
    
    print("\n2. Testing required imports...")
    test_required_imports_exist()
    
    print("\n" + "="*60)
    print("✅ All import tests passed!")
    print("="*60)
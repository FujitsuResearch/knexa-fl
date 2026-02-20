#!/usr/bin/env python3
"""
Academic Integrity Snapshot Test for KNEXA-FL
Runs mini-experiment and verifies metrics consistency
"""

import pytest
import subprocess
import json
import tempfile
from pathlib import Path
import sys
import os
import time


if os.environ.get("KNEXA_RUN_SNAPSHOT_TESTS") != "1":
    pytest.skip(
        "Snapshot integrity test disabled by default. Set KNEXA_RUN_SNAPSHOT_TESTS=1 to enable.",
        allow_module_level=True,
    )


@pytest.mark.snapshot
@pytest.mark.slow
def test_metrics_snapshot():
    """Run mini-experiment and verify metrics consistency"""
    print("Running KNEXA-FL mini-experiment for snapshot testing...")
    
    # Create temporary directory for experiment artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set environment variables to use temp directory
        env = os.environ.copy()
        env['KNEXA_ARTIFACTS_DIR'] = temp_dir
        
        # Run 2-round toy experiment with minimal configuration
        cmd = [
            sys.executable, 'src/main_p2p_real.py',
            '--rounds', '2',
            '--clients', '2', 
            '--seed', '42',
            '--local-pretrain-rounds', '1',
            '--batch-size-local', '4',
            '--learning-rate-local', '1e-4',
            '--experiment-name', 'snapshot_test'
        ]
        
        print(f"Command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env=env
            )
            
            elapsed_time = time.time() - start_time
            print(f"Experiment completed in {elapsed_time:.2f} seconds")
            
            if result.returncode != 0:
                print(f"\n❌ Experiment failed with return code: {result.returncode}")
                print(f"STDERR:\n{result.stderr}")
                print(f"STDOUT:\n{result.stdout[-1000:]}")  # Last 1000 chars
                raise AssertionError(f"Experiment failed: {result.stderr}")
            
            # Parse output for metrics
            output_lines = result.stdout.split('\n')
            metrics = parse_experiment_output(output_lines)
            
            # Validate metrics
            validate_metrics(metrics)
            
            # Check for synthetic data markers
            check_for_synthetic_data(output_lines)
            
            print("\n✅ Metrics snapshot test passed!")
            
        except subprocess.TimeoutExpired:
            print("❌ Experiment timed out after 10 minutes")
            raise AssertionError("Experiment timeout - possible infinite loop or hang")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise


def parse_experiment_output(output_lines):
    """Parse experiment output to extract key metrics"""
    metrics = {
        'rounds_completed': 0,
        'clients_trained': 0,
        'performance_values': [],
        'loss_values': [],
        'exchange_success_rate': None,
        'has_warnings': False,
        'has_errors': False
    }
    
    for line in output_lines:
        # Check for round completion
        if 'ROUND' in line and 'COMPLETED' in line:
            metrics['rounds_completed'] += 1
        
        # Extract performance values
        if 'pass@1' in line and ':' in line:
            try:
                # Extract numeric value after pass@1
                parts = line.split('pass@1')[-1].split()
                for part in parts:
                    try:
                        value = float(part.strip(':,'))
                        if 0 <= value <= 1:
                            metrics['performance_values'].append(value)
                            break
                    except ValueError:
                        continue
            except:
                pass
        
        # Extract loss values
        if 'loss' in line.lower() and ':' in line:
            try:
                parts = line.split(':')
                for i, part in enumerate(parts):
                    if 'loss' in part.lower() and i + 1 < len(parts):
                        try:
                            value = float(parts[i + 1].split()[0])
                            if 0 <= value <= 100:  # Reasonable loss range
                                metrics['loss_values'].append(value)
                        except ValueError:
                            pass
            except:
                pass
        
        # Check for warnings and errors
        if 'WARNING' in line or '⚠️' in line:
            metrics['has_warnings'] = True
        if 'ERROR' in line or '❌' in line:
            metrics['has_errors'] = True
    
    return metrics


def validate_metrics(metrics):
    """Validate that metrics are within expected ranges"""
    print("\nValidating experiment metrics...")
    
    # Check rounds completed
    if metrics['rounds_completed'] < 2:
        raise AssertionError(f"Expected 2 rounds, but only {metrics['rounds_completed']} completed")
    print(f"✅ Rounds completed: {metrics['rounds_completed']}")
    
    # Check performance values
    if not metrics['performance_values']:
        raise AssertionError("No performance values found in output")
    
    avg_performance = sum(metrics['performance_values']) / len(metrics['performance_values'])
    print(f"✅ Average performance: {avg_performance:.4f}")
    
    # Validate performance is not suspiciously high or low
    if avg_performance > 0.9:
        raise AssertionError(f"Suspiciously high average performance: {avg_performance}")
    if avg_performance < 0.001:
        raise AssertionError(f"Suspiciously low average performance: {avg_performance}")
    
    # Check loss values
    if metrics['loss_values']:
        avg_loss = sum(metrics['loss_values']) / len(metrics['loss_values'])
        print(f"✅ Average loss: {avg_loss:.4f}")
        
        # Validate loss is positive and reasonable
        if any(loss < 0 for loss in metrics['loss_values']):
            raise AssertionError("Negative loss values detected")
        if any(loss == 0.0 for loss in metrics['loss_values']):
            raise AssertionError("Exact zero loss values detected (suspicious)")
    
    # Check for critical errors
    if metrics['has_errors']:
        print("⚠️  Errors detected in output (may be expected for edge cases)")


def check_for_synthetic_data(output_lines):
    """Check for markers of synthetic/placeholder data"""
    print("\nChecking for synthetic data markers...")
    
    synthetic_markers = [
        'synthetic',
        'placeholder',
        'dummy',
        'fake',
        'mock',
        'fabricated',
        'generated baseline',
        'synthetic baseline'
    ]
    
    violations = []
    
    for line in output_lines:
        line_lower = line.lower()
        for marker in synthetic_markers:
            if marker in line_lower:
                violations.append(f"Synthetic marker '{marker}' found in: {line.strip()}")
    
    if violations:
        print("\n❌ Synthetic data markers detected:")
        for violation in violations[:5]:  # Show first 5
            print(f"  - {violation}")
        raise AssertionError(f"{len(violations)} synthetic data markers found")
    else:
        print("✅ No synthetic data markers detected")


if __name__ == "__main__":
    print("="*60)
    print("KNEXA-FL Academic Integrity Snapshot Test")
    print("="*60)
    
    try:
        test_metrics_snapshot()
        print("\n" + "="*60)
        print("✅ Academic integrity snapshot test passed!")
        print("="*60)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Academic integrity test failed!")
        print("="*60)
        raise

#!/usr/bin/env python3
"""
Proper pass@k evaluation with code execution verification
Fixes the critical evaluation methodology flaw identified in diagnostic analysis
"""
import ast
import random
import signal
import subprocess
import tempfile
import traceback
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import logging
import json
from datetime import datetime
from pathlib import Path
import hashlib
import threading
import functools

# Global lock for tokenizer thread safety
_tokenizer_lock = threading.Lock()

# CodeBLEU metric support
try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("CodeBLEU successfully imported")
except ImportError:
    CODEBLEU_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CodeBLEU not available. Install with: pip install codebleu")

# Simple BLEU as fallback
def simple_bleu_score(prediction: str, reference: str, n: int = 2) -> float:
    """
    Simple n-gram BLEU score as fallback when CodeBLEU is not available
    """
    if not prediction.strip() or not reference.strip():
        return 0.0
    
    # Tokenize by whitespace and common code tokens
    import re
    
    # Simple tokenization for code
    pred_tokens = re.findall(r'\w+|[^\w\s]', prediction.lower())
    ref_tokens = re.findall(r'\w+|[^\w\s]', reference.lower())
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Calculate n-gram overlap
    total_score = 0.0
    for n_gram in range(1, min(n + 1, len(pred_tokens) + 1)):
        pred_ngrams = set()
        ref_ngrams = set()
        
        for i in range(len(pred_tokens) - n_gram + 1):
            pred_ngrams.add(tuple(pred_tokens[i:i + n_gram]))
        
        for i in range(len(ref_tokens) - n_gram + 1):
            ref_ngrams.add(tuple(ref_tokens[i:i + n_gram]))
        
        if pred_ngrams:
            overlap = len(pred_ngrams & ref_ngrams)
            precision = overlap / len(pred_ngrams)
            total_score += precision / n  # Average across n-gram sizes
    
    return total_score

logger = logging.getLogger(__name__)

# Global cache for code execution results to avoid redundant evaluations
_CODE_EXECUTION_CACHE = {}

def _get_code_hash(code: str, test_code: str) -> str:
    """Generate a hash for code and test combination for caching"""
    combined = f"{code.strip()}\n===TEST===\n{test_code.strip()}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]

def _clear_code_cache():
    """Clear the code execution cache (for testing/debugging)"""
    global _CODE_EXECUTION_CACHE
    _CODE_EXECUTION_CACHE.clear()

def _get_cache_stats() -> Dict[str, int]:
    """Get cache statistics for monitoring"""
    return {
        "cache_size": len(_CODE_EXECUTION_CACHE),
        "cache_hits": getattr(_get_cache_stats, 'hits', 0),
        "cache_misses": getattr(_get_cache_stats, 'misses', 0)
    }


def validate_codebleu_environment() -> Dict[str, Any]:
    """
    Validate CodeBLEU environment for academic integrity
    Returns validation status to ensure no silent failures
    """
    validation = {
        "codebleu_available": CODEBLEU_AVAILABLE,
        "environment_ready": False,
        "test_calculation_works": False
    }
    
    if CODEBLEU_AVAILABLE:
        try:
            # Test with simple example to ensure tree-sitter works
            test_result = calc_codebleu(
                references=["def test(): return 1"],
                predictions=["def test(): return 1"],
                lang="python"
            )
            validation["test_calculation_works"] = True
            validation["environment_ready"] = True
            logger.info("CodeBLEU environment validation: READY")
        except Exception as e:
            validation["error"] = str(e)
            logger.warning(f"CodeBLEU environment validation: FAILED - {e}")
    else:
        logger.warning("CodeBLEU environment validation: NOT AVAILABLE")
    
    return validation


# Create a directory for code generation logs
# This will be updated dynamically if running within an experiment
CODE_GEN_LOG_DIR = Path("experimental_artifacts/knexa_fl/logs/code_generation")
CODE_GEN_LOG_DIR.mkdir(parents=True, exist_ok=True)

def set_code_gen_log_dir(new_dir: Path):
    """Update the code generation log directory (e.g., for experiments)"""
    global CODE_GEN_LOG_DIR
    CODE_GEN_LOG_DIR = Path(new_dir)
    CODE_GEN_LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_code_generation(prompt: str, generated_samples: List[str], extracted_codes: List[str], 
                       test_results: List[bool], problem_id: str = None, client_id: int = None,
                       codebleu_scores: List[float] = None):
    """Log code generation details for debugging and analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_data = {
        "timestamp": timestamp,
        "problem_id": problem_id,
        "client_id": client_id,
        "prompt": prompt,
        "num_samples": len(generated_samples),
        "samples": []
    }
    
    for i, (gen_text, code, passed) in enumerate(zip(generated_samples, extracted_codes, test_results)):
        sample_data = {
            "sample_index": i,
            "generated_text": gen_text,
            "extracted_code": code,
            "test_passed": passed,
            "code_length": len(code) if code else 0,
            "has_valid_code": bool(code and code.strip())
        }
        
        # Add CodeBLEU score if available
        if codebleu_scores and i < len(codebleu_scores):
            sample_data["codebleu_score"] = codebleu_scores[i]
        
        log_data["samples"].append(sample_data)
    
    # Calculate summary statistics
    log_data["summary"] = {
        "total_samples": len(generated_samples),
        "valid_code_samples": sum(1 for code in extracted_codes if code and code.strip()),
        "passed_samples": sum(test_results),
        "pass_rate": sum(test_results) / len(test_results) if test_results else 0
    }
    
    # Add CodeBLEU summary statistics
    if codebleu_scores:
        valid_codebleu_scores = [score for score in codebleu_scores if score is not None]
        if valid_codebleu_scores:
            log_data["summary"]["mean_codebleu"] = np.mean(valid_codebleu_scores)
            log_data["summary"]["std_codebleu"] = np.std(valid_codebleu_scores)
        
        # Track CodeBLEU evaluation success rate for academic transparency
        log_data["summary"]["codebleu_success_rate"] = len(valid_codebleu_scores) / len(codebleu_scores)
        log_data["summary"]["codebleu_failed_count"] = len(codebleu_scores) - len(valid_codebleu_scores)
    
    # Save to timestamped log file
    # Sanitize problem_id to prevent directory traversal issues
    safe_problem_id = problem_id.replace('/', '_').replace('\\', '_') if problem_id else 'unknown'
    log_filename = f"codegen_{client_id}_{safe_problem_id}_{timestamp}.json" if client_id is not None else f"codegen_{timestamp}.json"
    log_path = CODE_GEN_LOG_DIR / log_filename
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Also log summary to main logger
    logger.info(f"Code generation logged to {log_path}")
    logger.info(f"  Prompt: {prompt[:100]}...")
    logger.info(f"  Generated {len(generated_samples)} samples")
    logger.info(f"  Valid code: {log_data['summary']['valid_code_samples']}/{log_data['summary']['total_samples']}")
    logger.info(f"  Passed tests: {log_data['summary']['passed_samples']}/{log_data['summary']['total_samples']}")
    logger.info(f"  Pass rate: {log_data['summary']['pass_rate']:.2%}")
    
    # Log CodeBLEU summary if available
    if "mean_codebleu" in log_data["summary"]:
        logger.info(f"  Mean CodeBLEU: {log_data['summary']['mean_codebleu']:.4f} ± {log_data['summary']['std_codebleu']:.4f}")
    
    # Log CodeBLEU evaluation transparency
    if "codebleu_success_rate" in log_data["summary"]:
        logger.info(f"  CodeBLEU success rate: {log_data['summary']['codebleu_success_rate']:.1%} ({log_data['summary']['codebleu_failed_count']} failed)")


@contextmanager
def timeout(duration):
    """Context manager for timing out code execution"""
    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_python_code(generated_text: str) -> str:
    """Extract Python code from generated text, handling various formats and cleaning syntax"""
    # Remove the original prompt if present
    lines = generated_text.split('\n')
    
    # Find code blocks marked with ```python or ```
    in_code_block = False
    code_lines = []
    
    for line in lines:
        if line.strip().startswith('```python') or line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        
        if in_code_block:
            code_lines.append(line)
    
    if code_lines:
        return clean_python_code('\n'.join(code_lines))
    
    # If no code blocks found, try to extract the first complete function
    func_lines = []
    in_function = False
    indent_level = 0
    
    for line in lines:
        if line.strip().startswith('def '):
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            func_lines = [line]  # Start fresh with this function
        elif in_function:
            current_indent = len(line) - len(line.lstrip())
            # Continue if line is indented more than function def or is empty
            if (line.strip() == '' or current_indent > indent_level or 
                (current_indent == indent_level and line.strip().startswith(('elif', 'else', 'except', 'finally')))):
                func_lines.append(line)
            else:
                # Function is complete, stop here
                break
    
    if func_lines:
        return clean_python_code('\n'.join(func_lines))
    
    # Last resort: return cleaned entire generated text
    return clean_python_code(generated_text.strip())

def clean_python_code(code: str) -> str:
    """Clean common syntax issues in generated code"""
    if not code.strip():
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines that look like concatenated prompts or broken syntax
        if ('def ' in line and line.count('def ') > 1):
            # Multiple function definitions on one line - take only the first
            def_pos = line.find('def ')
            next_def = line.find('def ', def_pos + 4)
            if next_def != -1:
                line = line[:next_def].rstrip()
        
        # Remove common trailing syntax errors
        line = line.rstrip()
        if line.endswith(' 3.') or line.endswith(' 2.') or line.endswith(' 1.'):
            line = line[:-3].rstrip()
        
        # Fix common syntax concatenation issues - PRESERVE "from X import Y" syntax
        if ' import ' in line and not line.strip().startswith(('import ', 'from ')):
            # Only split if it's not a valid import statement
            if not line.strip().startswith('from ') or line.count(' import ') > 1:
                # Split on import if it's not at the beginning and not a valid "from" import
                parts = line.split(' import ')
                line = parts[0].rstrip()
                if line:
                    cleaned_lines.append(line)
                if len(parts) > 1:
                    cleaned_lines.append('import ' + parts[1])
                continue
        
        cleaned_lines.append(line)
    
    # Join lines and ensure proper function structure
    cleaned_code = '\n'.join(cleaned_lines)
    
    # Remove trailing incomplete content
    if '"""' in cleaned_code:
        # Ensure docstrings are properly closed
        parts = cleaned_code.split('"""')
        if len(parts) % 2 == 0:  # Odd number of """ means unclosed
            # Remove the last unclosed docstring
            cleaned_code = '"""'.join(parts[:-1])
    
    return cleaned_code.strip()


def calculate_codebleu_score(prediction: str, reference: str, lang: str = "python") -> Optional[Dict[str, float]]:
    """
    Calculate CodeBLEU score between prediction and reference code
    Returns None if CodeBLEU is not available or calculation fails
    """
    if not CODEBLEU_AVAILABLE:
        logger.debug("CodeBLEU not available - skipping score calculation")
        return None
    
    if not prediction.strip() or not reference.strip():
        logger.debug("Empty prediction or reference - skipping CodeBLEU")
        return None
    
    try:
        # CodeBLEU expects lists of predictions and references
        result = calc_codebleu(
            references=[reference],
            predictions=[prediction],
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25)  # Equal weights for all components
        )
        
        return {
            'codebleu': result.get('codebleu', 0.0),
            'ngram_match_score': result.get('ngram_match_score', 0.0),
            'weighted_ngram_match_score': result.get('weighted_ngram_match_score', 0.0),
            'syntax_match_score': result.get('syntax_match_score', 0.0),
            'dataflow_match_score': result.get('dataflow_match_score', 0.0)
        }
    except Exception as e:
        logger.warning(f"CodeBLEU calculation failed (tree-sitter compatibility issue): {e}")
        logger.info("CodeBLEU evaluation failed - no fallback substitution for academic integrity")
        
        # ACADEMIC INTEGRITY: Return None instead of fabricated scores
        # No BLEU substitution to maintain research integrity
        return None


def evaluate_codebleu_scores(predicted_codes: List[str], reference_solution: str, lang: str = "python") -> List[Optional[float]]:
    """
    Evaluate CodeBLEU scores for multiple predictions against a reference solution
    Returns list of CodeBLEU scores (or None for failed calculations)
    """
    if not CODEBLEU_AVAILABLE:
        logger.warning("CodeBLEU not available - skipping CodeBLEU evaluation")
        return [None] * len(predicted_codes)
    
    if not reference_solution.strip():
        logger.warning("No reference solution provided for CodeBLEU evaluation")
        return [None] * len(predicted_codes)
    
    scores = []
    for pred_code in predicted_codes:
        result = calculate_codebleu_score(pred_code, reference_solution, lang)
        if result:
            scores.append(result['codebleu'])
        else:
            scores.append(None)
    
    return scores


def safe_execute(code: str, test_code: str, timeout_seconds: int = 2) -> bool:
    """
    Safely execute code with test cases in an isolated environment
    Returns True if all tests pass, False otherwise
    """
    if not code.strip():
        return False
    
    # Check cache first
    code_hash = _get_code_hash(code, test_code)
    if code_hash in _CODE_EXECUTION_CACHE:
        # Update cache statistics
        if not hasattr(_get_cache_stats, 'hits'):
            _get_cache_stats.hits = 0
        _get_cache_stats.hits += 1
        return _CODE_EXECUTION_CACHE[code_hash]
    
    # Update cache statistics
    if not hasattr(_get_cache_stats, 'misses'):
        _get_cache_stats.misses = 0
    _get_cache_stats.misses += 1
    
    try:
        # Combine the generated code with test code
        full_code = f"{code}\n\n{test_code}"
        
        # Create a temporary file for execution
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        # Execute the code with timeout
        try:
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            # Check if execution was successful (no errors and exit code 0)
            success = result.returncode == 0 and not result.stderr.strip()
            
            if not success and result.stderr:
                logger.debug(f"Code execution failed: {result.stderr}")
            
            # Cache the result
            _CODE_EXECUTION_CACHE[code_hash] = success
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.debug("Code execution timed out")
            # Cache timeout as failure
            _CODE_EXECUTION_CACHE[code_hash] = False
            return False
        
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        logger.debug(f"Error in safe_execute: {e}")
        # Cache errors as failure
        _CODE_EXECUTION_CACHE[code_hash] = False
        return False


def safe_execute_batch(code_samples: List[str], test_code: str, timeout_seconds: int = 2) -> List[bool]:
    """
    Efficiently execute multiple code samples in parallel with batch processing and caching
    Returns list of boolean results indicating test success for each sample
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    import os
    
    if not code_samples:
        return []
    
    results = [False] * len(code_samples)
    uncached_samples = []
    
    # Check cache for all samples first
    for i, code in enumerate(code_samples):
        if not code.strip():
            results[i] = False
            continue
            
        code_hash = _get_code_hash(code, test_code)
        if code_hash in _CODE_EXECUTION_CACHE:
            # Cache hit
            if not hasattr(_get_cache_stats, 'hits'):
                _get_cache_stats.hits = 0
            _get_cache_stats.hits += 1
            results[i] = _CODE_EXECUTION_CACHE[code_hash]
        else:
            # Cache miss - needs execution
            uncached_samples.append((i, code, code_hash))
    
    if not uncached_samples:
        # All samples were cached
        return results
    
    # Update cache miss statistics
    if not hasattr(_get_cache_stats, 'misses'):
        _get_cache_stats.misses = 0
    _get_cache_stats.misses += len(uncached_samples)
    
    # Prepare batch execution for uncached samples
    temp_files = []
    
    try:
        # Create temporary files for uncached samples
        for i, code, code_hash in uncached_samples:
            full_code = f"{code}\n\n{test_code}"
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            temp_file.write(full_code)
            temp_file.close()
            temp_files.append((i, temp_file.name, code_hash))
        
        # Execute all samples in parallel using ThreadPoolExecutor
        def execute_single(file_info):
            idx, temp_file, code_hash = file_info
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                success = result.returncode == 0 and not result.stderr.strip()
                # Cache the result
                _CODE_EXECUTION_CACHE[code_hash] = success
                return idx, success
            except subprocess.TimeoutExpired:
                # Cache timeout as failure
                _CODE_EXECUTION_CACHE[code_hash] = False
                return idx, False
            except Exception:
                # Cache error as failure
                _CODE_EXECUTION_CACHE[code_hash] = False
                return idx, False
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(len(temp_files), 8)  # Limit concurrent processes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(execute_single, file_info) for file_info in temp_files]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, success = future.result()
                    results[idx] = success
                except Exception as e:
                    logger.debug(f"Error in batch execution: {e}")
    
    finally:
        # Clean up all temporary files
        for _, temp_file, _ in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    return results


def prepare_test_code(example: Dict[str, Any]) -> str:
    """
    Prepare test code for execution based on the dataset format
    """
    if 'test' in example:
        # HumanEval format
        return example['test']
    elif 'test_list' in example:
        # MBPP format - convert test list to executable code
        test_lines = []
        for test_case in example['test_list']:
            if test_case.strip():
                test_lines.append(f"assert {test_case}")
        return '\n'.join(test_lines)
    else:
        # Fallback: try to use any available test information
        logger.warning(f"No standard test format found for example: {example.keys()}")
        return ""


def extract_reference_solution(example: Dict[str, Any]) -> Optional[str]:
    """
    Extract reference solution from dataset example for CodeBLEU evaluation
    """
    # For HumanEval format, reconstruct complete solution
    if 'canonical_solution' in example and 'prompt' in example:
        canonical = example['canonical_solution'].strip()
        prompt = example['prompt']
        
        if canonical and prompt:
            # Extract imports and function signature from prompt
            prompt_lines = prompt.split('\n')
            imports = []
            func_signature = None
            
            for line in prompt_lines:
                line_stripped = line.strip()
                if line_stripped.startswith(('import ', 'from ')):
                    imports.append(line_stripped)
                elif line_stripped.startswith('def ') and ':' in line_stripped:
                    func_signature = line_stripped
            
            if func_signature:
                # Reconstruct complete solution
                complete_parts = []
                if imports:
                    complete_parts.extend(imports)
                    complete_parts.append('')  # Empty line after imports
                
                complete_parts.append(func_signature)
                
                # Ensure canonical solution is properly indented
                if canonical:
                    # Add canonical solution with proper indentation
                    canonical_lines = canonical.split('\n')
                    for line in canonical_lines:
                        if line.strip():  # Only add non-empty lines
                            # Ensure proper indentation (at least 4 spaces)
                            if not line.startswith('    '):
                                line = '    ' + line.lstrip()
                            complete_parts.append(line)
                        else:
                            complete_parts.append(line)  # Preserve empty lines
                
                complete_solution = '\n'.join(complete_parts)
                
                # Validate syntax before returning
                try:
                    import ast
                    ast.parse(complete_solution)
                    return complete_solution
                except SyntaxError as e:
                    logger.warning(f"Syntax error in reconstructed reference solution: {e}")
                    return None
    
    # Fallback: try other possible keys for reference solutions
    for key in ['solution', 'code', 'reference']:
        if key in example and example[key]:
            solution = example[key].strip()
            if solution:
                return clean_python_code(solution)
    
    logger.debug(f"No valid reference solution found for example with keys: {list(example.keys())}")
    return None


def generate_code_samples(model, tokenizer, prompt: str, num_samples: int, max_tokens: int = 256, temperature: float = 0.8) -> List[str]:
    """
    Generate multiple code samples for pass@k evaluation
    Uses proper sampling techniques for diversity
    Supports both decoder-only and encoder-decoder models
    """
    samples = []
    device = next(model.parameters()).device
    
    # Check if this is an encoder-decoder model (T5/mT5)
    is_encoder_decoder = hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder
    
    # Prepare input (thread-safe tokenizer access)
    with _tokenizer_lock:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    for _ in range(num_samples):
        try:
            with torch.no_grad():
                if is_encoder_decoder:
                    # For T5-style models, use encoder-decoder generation
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_length=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    # For T5, the entire output is the generated text (no input prefix)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # For decoder-only models (GPT-style)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        top_p=0.95,
                        top_k=50,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    # Decode only the generated part (exclude input prompt)
                    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                samples.append(generated_text)
            
        except Exception as e:
            logger.error(f"Failed to generate sample: {e}")
            raise RuntimeError(f"Code generation failed: {e}") from e
    
    return samples


def evaluate_pass_at_k(samples: List[str], test_code: str, k_values: List[int] = [1, 5, 10], 
                       prompt: str = None, problem_id: str = None, client_id: int = None,
                       reference_solution: str = None) -> Dict[str, float]:
    """
    Evaluate pass@k metrics and CodeBLEU scores for generated code samples
    """
    # Extract code from all samples first
    extracted_codes = [extract_python_code(sample) for sample in samples]
    
    # Use batch execution for significant performance improvement
    passed_samples = safe_execute_batch(extracted_codes, test_code)
    
    codebleu_scores = []
    
    # Calculate CodeBLEU scores if reference solution is provided
    if reference_solution:
        codebleu_scores = evaluate_codebleu_scores(extracted_codes, reference_solution)
    
    # Log the code generation details if prompt is provided
    if prompt is not None:
        log_code_generation(prompt, samples, extracted_codes, passed_samples, problem_id, client_id, codebleu_scores)
    
    # Calculate pass@k for each k
    results = {}
    total_samples = len(passed_samples)
    num_passed = sum(passed_samples)
    
    for k in k_values:
        # CORRECTED: Remove incorrect edge case handling
        # Always use the proper unbiased estimator formula
        
        if k > total_samples:
            # When k > n, use k = n for calculation (evaluate with all available samples)
            # This gives us the best estimate we can with limited samples
            effective_k = total_samples
        else:
            effective_k = k
            
        # Standard pass@k calculation using unbiased estimator
        # pass@k = 1 - C(n-c, k) / C(n, k) where n=total, c=passed
        if num_passed == 0:
            pass_at_k = 0.0
        elif total_samples - num_passed < effective_k:
            # If there aren't enough failing samples to fill k slots, 
            # we're guaranteed to get at least one passing sample
            pass_at_k = 1.0
        else:
            # Use the unbiased estimator
            from math import comb
            try:
                pass_at_k = 1.0 - comb(total_samples - num_passed, effective_k) / comb(total_samples, effective_k)
            except Exception as e:
                raise RuntimeError(f"Pass@k calculation failed: {e}") from e
        
        results[f'pass@{k}'] = pass_at_k
    
    # Add CodeBLEU metrics if available
    if codebleu_scores:
        valid_codebleu_scores = [score for score in codebleu_scores if score is not None]
        if valid_codebleu_scores:
            results['codebleu_mean'] = np.mean(valid_codebleu_scores)
            results['codebleu_std'] = np.std(valid_codebleu_scores)
            results['codebleu_max'] = np.max(valid_codebleu_scores)
            results['codebleu_min'] = np.min(valid_codebleu_scores)
            results['codebleu_count'] = len(valid_codebleu_scores)
            results['codebleu_success_rate'] = len(valid_codebleu_scores) / len(codebleu_scores)
        # ACADEMIC INTEGRITY: Don't inject zeros - simply omit metrics when unavailable
        # If no valid CodeBLEU scores, don't add CodeBLEU metrics to results
    
    return results


class ImprovedCodeEvaluator:
    """
    Improved code evaluator that replaces the inadequate string matching
    """
    
    def __init__(self, model, tokenizer, device, eval_samples: int = 50, max_k: int = 10, client_id: int = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eval_samples = min(eval_samples, 50)  # Scale up from 5 to 50
        # CORRECTED: Generate more samples than k for better statistical estimation
        # For pass@10, we should generate at least 15-20 samples
        self.max_k = max(max_k, 15)  # Ensure at least 15 samples for pass@10
        self.client_id = client_id  # For logging purposes
        
    def evaluate_on_dataset(self, dataset: List[Dict], progress_callback=None) -> Dict[str, float]:
        """
        Evaluate model on a dataset using proper pass@k metrics
        """
        # Sample evaluation problems (increased from 5 to 50+)
        eval_problems = random.sample(list(dataset), k=min(self.eval_samples, len(dataset)))
        
        all_results = {f'pass@{k}': [] for k in [1, 5, 10]}
        # Add CodeBLEU metrics to track
        all_results.update({
            'codebleu_mean': [],
            'codebleu_std': [],
            'codebleu_max': [],
            'codebleu_min': [],
            'codebleu_count': []
        })
        successful_evaluations = 0
        
        logger.info(f"Starting evaluation on {len(eval_problems)} problems with proper code execution")
        
        for i, example in enumerate(eval_problems):
            try:
                if progress_callback:
                    progress_callback(i, len(eval_problems))
                
                # Prepare prompt, test code, and reference solution
                prompt = example.get('prompt', '')
                test_code = prepare_test_code(example)
                reference_solution = extract_reference_solution(example)
                
                if not test_code:
                    logger.warning(f"No test code available for problem {i}, skipping")
                    continue
                
                # Generate code samples for pass@k evaluation
                samples = generate_code_samples(
                    self.model, 
                    self.tokenizer, 
                    prompt, 
                    num_samples=self.max_k,
                    max_tokens=256,
                    temperature=0.8
                )
                
                # Evaluate pass@k and CodeBLEU with logging
                problem_id = example.get('task_id', f'problem_{i}')
                problem_results = evaluate_pass_at_k(
                    samples, test_code, k_values=[1, 5, 10],
                    prompt=prompt, problem_id=problem_id, client_id=getattr(self, 'client_id', None),
                    reference_solution=reference_solution
                )
                
                # Accumulate results
                for metric, value in problem_results.items():
                    all_results[metric].append(value)
                
                successful_evaluations += 1
                
                logger.debug(f"Problem {i+1}/{len(eval_problems)}: {problem_results}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate problem {i}: {e}")
                raise RuntimeError(f"Evaluation failed for problem {i}: {e}") from e
        
        # Calculate final metrics
        final_results = {}
        for metric, values in all_results.items():
            if values:
                final_results[metric] = np.mean(values)
                final_results[f'{metric}_std'] = np.std(values)
            else:
                final_results[metric] = 0.0
                final_results[f'{metric}_std'] = 0.0
        
        final_results['successful_evaluations'] = successful_evaluations
        final_results['total_problems'] = len(eval_problems)
        
        logger.info(f"Evaluation complete: {successful_evaluations}/{len(eval_problems)} problems evaluated successfully")
        logger.info(f"Pass@k Results: pass@1={final_results.get('pass@1', 0):.3f}, "
                   f"pass@5={final_results.get('pass@5', 0):.3f}, "
                   f"pass@10={final_results.get('pass@10', 0):.3f}")
        
        # Log CodeBLEU results if available
        if final_results.get('codebleu_mean') is not None:
            logger.info(f"CodeBLEU Results: mean={final_results.get('codebleu_mean', 0):.3f}, "
                       f"std={final_results.get('codebleu_std', 0):.3f}, "
                       f"max={final_results.get('codebleu_max', 0):.3f}")
        
        return final_results
    
    def quick_eval_pass1(self, dataset: List[Dict]) -> float:
        """
        Quick evaluation method that maintains compatibility with existing code
        but uses proper code execution instead of string matching
        """
        # Use fewer samples for quick evaluation but still proper execution
        quick_samples = min(10, len(dataset), self.eval_samples // 5)
        eval_problems = random.sample(list(dataset), k=quick_samples)
        
        passed = 0
        total = 0
        
        for example in eval_problems:
            try:
                prompt = example.get('prompt', '')
                test_code = prepare_test_code(example)
                
                if not test_code:
                    continue
                
                # Generate single sample for pass@1 (use small temperature for deterministic but valid sampling)
                samples = generate_code_samples(self.model, self.tokenizer, prompt, num_samples=1, temperature=0.1)
                
                if samples and samples[0]:
                    code = extract_python_code(samples[0])
                    if code.strip():
                        passed += safe_execute(code, test_code)
                
                total += 1
                
            except Exception as e:
                logger.error(f"Quick eval failed for problem: {e}")
                raise RuntimeError(f"Quick evaluation failed: {e}") from e
        
        return passed / total if total > 0 else 0.0


def test_evaluation_system():
    """
    Test the evaluation system with sample code
    """
    # Test code extraction
    sample_generated = '''Here's the solution:

```python
def add_numbers(a, b):
    return a + b
```

This function adds two numbers.'''
    
    extracted = extract_python_code(sample_generated)
    print(f"Extracted code: {repr(extracted)}")
    
    # Test code execution
    test_code = '''
assert add_numbers(2, 3) == 5
assert add_numbers(0, 0) == 0
assert add_numbers(-1, 1) == 0
'''
    
    result = safe_execute(extracted, test_code)
    print(f"Execution result: {result}")
    
    # Test CodeBLEU calculation
    reference_solution = '''def add_numbers(a, b):
    """Add two numbers together."""
    return a + b'''
    
    if CODEBLEU_AVAILABLE:
        codebleu_result = calculate_codebleu_score(extracted, reference_solution)
        print(f"CodeBLEU result: {codebleu_result}")
    else:
        print("CodeBLEU not available - install with: pip install codebleu")
    
    # Test pass@k with CodeBLEU
    samples = [sample_generated]
    pass_k_results = evaluate_pass_at_k(
        samples, test_code, k_values=[1], 
        reference_solution=reference_solution
    )
    print(f"Pass@k with CodeBLEU results: {pass_k_results}")


if __name__ == "__main__":
    test_evaluation_system()
#!/usr/bin/env python3
"""
File Checksum Utilities for KNEXA-FL
Provides SHA256 hash calculation and verification for academic integrity
"""

import hashlib
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def sha256_file(filepath: str) -> str:
    """
    Calculate SHA256 hash of file
    
    Args:
        filepath: Path to file
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate SHA256 for {filepath}: {e}")
        raise


def verify_artifact(filepath: str, expected_hash: str) -> bool:
    """
    Verify artifact integrity by comparing SHA256 hash
    
    Args:
        filepath: Path to file to verify
        expected_hash: Expected SHA256 hash
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = sha256_file(filepath)
        return actual_hash == expected_hash
    except Exception as e:
        logger.error(f"Failed to verify artifact {filepath}: {e}")
        return False


def calculate_directory_checksums(directory: Path) -> dict:
    """
    Calculate SHA256 checksums for all files in a directory
    
    Args:
        directory: Path to directory
        
    Returns:
        Dictionary mapping relative file paths to SHA256 hashes
    """
    checksums = {}
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return checksums
        
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                relative_path = file_path.relative_to(directory)
                checksums[str(relative_path)] = sha256_file(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to checksum {file_path}: {e}")
                
    return checksums


def verify_directory_integrity(directory: Path, expected_checksums: dict) -> tuple:
    """
    Verify integrity of all files in a directory against expected checksums
    
    Args:
        directory: Path to directory
        expected_checksums: Dictionary of expected checksums
        
    Returns:
        Tuple of (all_valid: bool, failed_files: list)
    """
    failed_files = []
    
    for relative_path, expected_hash in expected_checksums.items():
        file_path = directory / relative_path
        
        if not file_path.exists():
            logger.error(f"Missing file: {relative_path}")
            failed_files.append(str(relative_path))
            continue
            
        if not verify_artifact(str(file_path), expected_hash):
            logger.error(f"Checksum mismatch: {relative_path}")
            failed_files.append(str(relative_path))
            
    return len(failed_files) == 0, failed_files
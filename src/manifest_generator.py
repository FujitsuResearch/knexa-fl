#!/usr/bin/env python3
"""
Manifest Generator for KNEXA-FL Experiments
Creates machine-readable artifact manifests with checksums for academic integrity
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ManifestGenerator:
    """
    Generates comprehensive manifests for experimental artifacts
    Ensures academic integrity through checksum verification
    """
    
    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.manifest_path = self.experiment_dir / "manifest.json"
        
    def generate_manifest(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive manifest for experiment artifacts"""
        
        logger.info("📋 Generating experiment manifest...")
        
        # Extract experiment metadata
        experiment_name = experiment_config.get("experiment_name", "unknown")
        timestamp = datetime.now().isoformat()
        
        manifest = {
            "experiment_id": experiment_name,
            "manifest_version": "1.0",
            "generated": timestamp,
            "experiment_config": experiment_config,
            "academic_integrity_statement": "All values in this manifest represent genuine experimental outputs. No synthetic or fabricated data has been introduced.",
            "directory_structure": self._map_directory_structure(),
            "artifacts": self._catalog_artifacts(),
            "checksums": self._generate_checksums(),
            "compression_info": self._analyze_compression(),
            "size_analysis": self._analyze_sizes()
        }
        
        # Save manifest
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
            
        logger.info(f"✅ Manifest generated: {self.manifest_path}")
        return manifest
        
    def _map_directory_structure(self) -> Dict[str, Any]:
        """Map the complete directory structure"""
        
        def scan_directory(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
            if current_depth >= max_depth:
                return {"truncated": True}
                
            structure = {}
            
            if not path.exists():
                return structure
                
            # Count files and subdirectories
            files = []
            subdirs = {}
            
            for item in path.iterdir():
                if item.is_file():
                    files.append({
                        "name": item.name,
                        "size_bytes": item.stat().st_size,
                        "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                    })
                elif item.is_dir():
                    subdirs[item.name] = scan_directory(item, max_depth, current_depth + 1)
            
            structure["files"] = files
            structure["subdirectories"] = subdirs
            structure["file_count"] = len(files)
            structure["subdir_count"] = len(subdirs)
            
            return structure
        
        return scan_directory(self.experiment_dir)
    
    def _catalog_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Catalog all important artifacts with metadata"""
        
        artifacts = {}
        
        # Key files to catalog
        key_files = [
            "run.log",
            "config.yaml", 
            "system_info.json",
            "final_results.json",
            "experiment_summary.json",
            "comprehensive_evaluation_report.json"
        ]
        
        # Catalog key files
        for filename in key_files:
            file_path = self.experiment_dir / filename
            if file_path.exists():
                artifacts[filename] = self._file_metadata(file_path)
        
        # Catalog directories
        important_dirs = [
            "raw_data", "checkpoints", "plots", "paper_materials", 
            "code_generation", "metrics", "round_results"
        ]
        
        for dirname in important_dirs:
            dir_path = self.experiment_dir / dirname
            if dir_path.exists():
                artifacts[f"{dirname}/"] = self._directory_metadata(dir_path)
                
        return artifacts
    
    def _file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Generate metadata for a single file"""
        stat = file_path.stat()
        
        return {
            "type": "file",
            "size_bytes": stat.st_size,
            "size_human": self._human_readable_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "md5": self._calculate_md5(file_path) if stat.st_size < 100 * 1024 * 1024 else "large_file_skipped"
        }
    
    def _directory_metadata(self, dir_path: Path) -> Dict[str, Any]:
        """Generate metadata for a directory"""
        
        total_size = 0
        file_count = 0
        
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            "type": "directory", 
            "total_size_bytes": total_size,
            "total_size_human": self._human_readable_size(total_size),
            "file_count": file_count,
            "subdirectory_count": len([d for d in dir_path.iterdir() if d.is_dir()])
        }
    
    def _generate_checksums(self) -> Dict[str, str]:
        """Generate checksums for critical files"""
        
        checksums = {}
        
        # Critical files for academic integrity
        critical_files = [
            "final_results.json",
            "config.yaml",
            "experiment_summary.json"
        ]
        
        for filename in critical_files:
            file_path = self.experiment_dir / filename
            if file_path.exists():
                checksums[filename] = self._calculate_md5(file_path)
                
        return checksums
    
    def _analyze_compression(self) -> Dict[str, Any]:
        """Analyze compression applied to artifacts"""
        
        compression_info = {
            "code_generation_logs": "Not compressed",
            "checkpoints": "Not compressed", 
            "estimated_savings": "0 MB"
        }
        
        # Check if code generation logs are compressed
        code_gen_dir = self.experiment_dir / "code_generation"
        if code_gen_dir.exists():
            archive_dir = code_gen_dir / "archive"
            summaries_dir = code_gen_dir / "summaries"
            
            if archive_dir.exists() and summaries_dir.exists():
                # Count original vs compressed
                original_files = len(list(archive_dir.glob("*.json")))
                summary_files = len(list(summaries_dir.glob("*.json")))
                
                if summary_files > 0:
                    compression_info["code_generation_logs"] = f"Compressed from {original_files} files to {summary_files} summaries"
        
        return compression_info
    
    def _analyze_sizes(self) -> Dict[str, Any]:
        """Analyze artifact sizes for optimization insights"""
        
        size_analysis = {}
        
        # Analyze major directories
        for dirname in ["checkpoints", "plots", "paper_materials", "code_generation", "raw_data"]:
            dir_path = self.experiment_dir / dirname
            if dir_path.exists():
                total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
                size_analysis[dirname] = {
                    "size_bytes": total_size,
                    "size_human": self._human_readable_size(total_size)
                }
        
        # Calculate total experiment size
        total_size = sum(f.stat().st_size for f in self.experiment_dir.rglob("*") if f.is_file())
        size_analysis["total_experiment"] = {
            "size_bytes": total_size,
            "size_human": self._human_readable_size(total_size)
        }
        
        return size_analysis
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum for a file"""
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate MD5 for {file_path}: {e}")
            return "calculation_failed"
    
    def _human_readable_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def verify_manifest(self) -> bool:
        """Verify manifest integrity by checking checksums"""
        
        if not self.manifest_path.exists():
            logger.error("Manifest file not found")
            return False
            
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
                
            checksums = manifest.get("checksums", {})
            
            for filename, expected_checksum in checksums.items():
                file_path = self.experiment_dir / filename
                if file_path.exists():
                    actual_checksum = self._calculate_md5(file_path)
                    if actual_checksum != expected_checksum:
                        logger.error(f"Checksum mismatch for {filename}: expected {expected_checksum}, got {actual_checksum}")
                        return False
                else:
                    logger.error(f"File missing: {filename}")
                    return False
                    
            logger.info("✅ Manifest verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Manifest verification failed: {e}")
            return False
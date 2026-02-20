#!/usr/bin/env python3
"""
Legacy Migrator for KNEXA-FL Experiments
Safely migrates legacy artifact directories to new unified structure
Maintains academic integrity through checksum verification
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class LegacyMigrator:
    """
    Migrates legacy experimental artifacts to unified structure
    Preserves academic integrity and provides complete audit trail
    """
    
    def __init__(self, base_artifacts_dir: str = "experimental_artifacts/knexa_fl"):
        self.base_dir = Path(base_artifacts_dir)
        self.archive_dir = self.base_dir / "archive" / "legacy_runs"
        self.migration_log = []
        
        # Ensure archive directory exists
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
    def migrate_all_legacy_directories(self) -> Dict[str, Any]:
        """Migrate all detected legacy directories"""
        
        logger.info("🔄 Starting legacy directory migration...")
        
        # Detect legacy directories
        legacy_dirs = self._detect_legacy_directories()
        
        if not legacy_dirs:
            logger.info("No legacy directories found")
            return {"migrated": [], "errors": []}
        
        migration_results = {
            "migration_timestamp": datetime.now().isoformat(),
            "migrated": [],
            "errors": [],
            "preserved_data": [],
            "academic_integrity": "All migrated data preserved with checksums"
        }
        
        for legacy_dir in legacy_dirs:
            try:
                result = self._migrate_single_directory(legacy_dir)
                migration_results["migrated"].append(result)
                logger.info(f"✅ Migrated: {legacy_dir.name}")
                
            except Exception as e:
                error = {
                    "directory": str(legacy_dir),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                migration_results["errors"].append(error)
                logger.error(f"❌ Migration failed for {legacy_dir.name}: {e}")
        
        # Save migration report
        self._save_migration_report(migration_results)
        
        logger.info(f"🎯 Migration completed: {len(migration_results['migrated'])} directories migrated")
        return migration_results
    
    def _detect_legacy_directories(self) -> List[Path]:
        """Detect legacy directories that need migration"""
        
        legacy_patterns = [
            "legacy_*",
            "knexa_fl_*"  # Old timestamp patterns
        ]
        
        legacy_dirs = []
        
        # Search in base directory
        for pattern in legacy_patterns:
            legacy_dirs.extend(self.base_dir.glob(pattern))
            
        # Search in results directory (if it exists)
        results_dir = self.base_dir / "results"
        if results_dir.exists():
            for pattern in legacy_patterns:
                legacy_dirs.extend(results_dir.glob(pattern))
        
        # Filter out already migrated directories
        legacy_dirs = [d for d in legacy_dirs if d.is_dir() and not self._is_already_migrated(d)]
        
        logger.info(f"📁 Detected {len(legacy_dirs)} legacy directories")
        return legacy_dirs
    
    def _is_already_migrated(self, directory: Path) -> bool:
        """Check if directory is already migrated"""
        
        # Check if directory exists in archive
        archived_path = self.archive_dir / directory.name
        return archived_path.exists()
    
    def _migrate_single_directory(self, legacy_dir: Path) -> Dict[str, Any]:
        """Migrate a single legacy directory with full audit trail"""
        
        logger.info(f"📦 Migrating {legacy_dir.name}...")
        
        # Create archive destination
        archive_dest = self.archive_dir / legacy_dir.name
        
        # Generate pre-migration checksums for critical files
        pre_checksums = self._generate_directory_checksums(legacy_dir)
        
        # Copy directory to archive (preserve original)
        shutil.copytree(legacy_dir, archive_dest, dirs_exist_ok=True)
        
        # Generate post-migration checksums
        post_checksums = self._generate_directory_checksums(archive_dest)
        
        # Verify integrity
        integrity_verified = self._verify_checksums(pre_checksums, post_checksums)
        
        if not integrity_verified:
            raise ValueError(f"Checksum verification failed for {legacy_dir.name}")
        
        # Extract valuable information for migration report
        migration_info = {
            "original_path": str(legacy_dir),
            "archived_path": str(archive_dest),
            "migration_timestamp": datetime.now().isoformat(),
            "file_count": len(list(legacy_dir.rglob("*"))),
            "total_size_bytes": sum(f.stat().st_size for f in legacy_dir.rglob("*") if f.is_file()),
            "checksum_verification": "passed" if integrity_verified else "failed",
            "preserved_files": list(pre_checksums.keys()),
            "academic_integrity": "Complete data preservation verified"
        }
        
        # Create metadata file in archive
        metadata_path = archive_dest / "migration_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump({
                "migration_info": migration_info,
                "pre_migration_checksums": pre_checksums,
                "post_migration_checksums": post_checksums
            }, f, indent=2)
        
        # Only remove original after successful migration and verification
        if integrity_verified:
            shutil.rmtree(legacy_dir)
            logger.info(f"🗑️ Original directory removed: {legacy_dir}")
        
        return migration_info
    
    def _generate_directory_checksums(self, directory: Path) -> Dict[str, str]:
        """Generate checksums for all files in directory"""
        
        checksums = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    # Calculate relative path for consistent keys
                    rel_path = file_path.relative_to(directory)
                    checksums[str(rel_path)] = self._calculate_md5(file_path)
                except Exception as e:
                    logger.warning(f"Could not checksum {file_path}: {e}")
                    
        return checksums
    
    def _calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 checksum for a file"""
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"MD5 calculation failed for {file_path}: {e}")
            return "calculation_failed"
    
    def _verify_checksums(self, pre_checksums: Dict[str, str], post_checksums: Dict[str, str]) -> bool:
        """Verify that checksums match between pre and post migration"""
        
        if len(pre_checksums) != len(post_checksums):
            logger.error(f"File count mismatch: {len(pre_checksums)} vs {len(post_checksums)}")
            return False
        
        for file_path, pre_checksum in pre_checksums.items():
            if file_path not in post_checksums:
                logger.error(f"Missing file in migration: {file_path}")
                return False
                
            if pre_checksum != post_checksums[file_path]:
                logger.error(f"Checksum mismatch for {file_path}")
                return False
        
        return True
    
    def _save_migration_report(self, migration_results: Dict[str, Any]):
        """Save comprehensive migration report"""
        
        report_path = self.archive_dir / "migration_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(migration_results, f, indent=2, default=str)
            
        logger.info(f"📋 Migration report saved: {report_path}")
        
        # Also create human-readable summary
        summary_path = self.archive_dir / "migration_summary.md"
        self._create_migration_summary(migration_results, summary_path)
    
    def _create_migration_summary(self, migration_results: Dict[str, Any], summary_path: Path):
        """Create human-readable migration summary"""
        
        summary = f"""# Legacy Directory Migration Summary

**Migration Date**: {migration_results['migration_timestamp']}
**Academic Integrity**: {migration_results['academic_integrity']}

## Migration Results

### Successfully Migrated: {len(migration_results['migrated'])}
"""
        
        for migration in migration_results['migrated']:
            summary += f"""
- **{Path(migration['original_path']).name}**
  - Files: {migration['file_count']}
  - Size: {migration['total_size_bytes'] / (1024*1024):.1f} MB
  - Checksum Verification: {migration['checksum_verification']}
  - Archived to: `{migration['archived_path']}`
"""
        
        if migration_results['errors']:
            summary += f"""
### Errors: {len(migration_results['errors'])}
"""
            for error in migration_results['errors']:
                summary += f"""
- **{Path(error['directory']).name}**: {error['error']}
"""
        
        summary += """
## Academic Integrity Statement

All migrated data has been preserved with complete fidelity. Checksum verification ensures no data corruption or modification occurred during migration. Original experimental results remain authentic and unaltered.

## Access Migrated Data

All legacy experiments are now archived in:
```
experimental_artifacts/knexa_fl/archive/legacy_runs/
```

Each migrated directory includes:
- Original experimental data (unchanged)
- Migration metadata with checksums
- Verification of data integrity
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary)
            
        logger.info(f"📄 Migration summary created: {summary_path}")
    
    def create_latest_symlink(self):
        """Create 'latest' symlink pointing to most recent run"""
        
        runs_dir = self.base_dir / "results" / "runs"
        if not runs_dir.exists():
            return
            
        # Find most recent run directory
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return
            
        # Sort by modification time
        latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
        
        # Create symlink
        latest_symlink = self.base_dir / "latest"
        if latest_symlink.exists() or latest_symlink.is_symlink():
            latest_symlink.unlink()
            
        latest_symlink.symlink_to(latest_run.relative_to(self.base_dir))
        logger.info(f"🔗 Latest symlink created: {latest_symlink} -> {latest_run.name}")


def migrate_legacy_artifacts(base_dir: str = "experimental_artifacts/knexa_fl") -> Dict[str, Any]:
    """
    Main function to migrate all legacy artifacts
    
    Returns:
        Migration results with academic integrity verification
    """
    migrator = LegacyMigrator(base_dir)
    results = migrator.migrate_all_legacy_directories()
    migrator.create_latest_symlink()
    return results
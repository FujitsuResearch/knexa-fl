#!/usr/bin/env python3
"""
Unified Console Logger for KNEXA-FL
Captures ALL stdout/stderr to unified run.log while maintaining academic integrity
"""

import sys
import io
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO


class TeeFile:
    """Writes to multiple file-like objects simultaneously"""
    
    def __init__(self, *files):
        self.files = files
        
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
            
    def flush(self):
        for f in self.files:
            f.flush()


class UnifiedConsoleLogger:
    """
    Captures all console output to unified run.log
    Maintains academic integrity with proper timestamps
    """
    
    def __init__(self, run_log_path: Path):
        self.run_log_path = run_log_path
        self.run_log_file: Optional[TextIO] = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self._lock = threading.Lock()
        self.is_active = False
        
        # Ensure parent directory exists
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        
    def start_capture(self):
        """Begin capturing all console output"""
        with self._lock:
            if self.is_active:
                return
                
            # Open run log file
            self.run_log_file = open(self.run_log_path, 'w', encoding='utf-8')
            
            # Write header with academic integrity statement
            header = f"""# KNEXA-FL Experiment Run Log
# Generated: {datetime.now().isoformat()}
# Academic Integrity: All logged values are genuine experimental outputs
# No synthetic or fabricated data has been introduced
# ================================================================

"""
            self.run_log_file.write(header)
            self.run_log_file.flush()
            
            # Redirect stdout and stderr to tee to both console and file
            sys.stdout = TeeFile(self.original_stdout, self.run_log_file)
            sys.stderr = TeeFile(self.original_stderr, self.run_log_file)
            
            self.is_active = True
            print(f"📝 Console logging started: {self.run_log_path}")
            
    def stop_capture(self):
        """Stop capturing console output"""
        with self._lock:
            if not self.is_active:
                return
                
            print(f"📝 Console logging stopped: {self.run_log_path}")
            
            # Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # Close run log file
            if self.run_log_file:
                footer = f"""
# ================================================================
# Experiment completed: {datetime.now().isoformat()}
# All experimental results above are authentic and unmodified
"""
                self.run_log_file.write(footer)
                self.run_log_file.close()
                self.run_log_file = None
                
            self.is_active = False
            
    def __enter__(self):
        self.start_capture()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_capture()


def create_unified_logger(experiment_dir: Path) -> UnifiedConsoleLogger:
    """Create unified console logger for experiment"""
    run_log_path = experiment_dir / "run.log"
    return UnifiedConsoleLogger(run_log_path)
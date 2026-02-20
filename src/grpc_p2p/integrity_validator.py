#!/usr/bin/env python3
"""
Research Integrity Validator for KNEXA-FL
Ensures all reported results are from actual computation, not synthetic data
"""

import logging
import time
from typing import Dict, List, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class IntegrityValidator:
    """
    Validates research integrity by checking for synthetic data and implementation issues
    """
    
    def __init__(self):
        self.validation_history = []
        self.integrity_violations = []
        
    def validate_p2p_exchange(self, exchange_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a P2P exchange result for integrity issues
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Performance gain should not be exactly 0 if exchange succeeded
        if exchange_result.get('success', False):
            perf_gain = exchange_result.get('performance_gain', 0.0)
            if perf_gain == 0.0:
                issues.append("Successful exchange reported but zero performance gain")
            
            # Check 2: Performance gain should be realistic (not too high)
            if perf_gain > 0.2:  # 20% gain in one exchange is suspicious
                issues.append(f"Unrealistic performance gain: {perf_gain:.4f}")
        
        # Check 3: Bytes transferred should match exchange type
        bytes_transferred = exchange_result.get('bytes_transferred', 0)
        exchange_type = exchange_result.get('exchange_type', 'unknown')
        
        if exchange_result.get('success', False) and bytes_transferred == 0:
            if exchange_result.get('role') == 'teacher':
                issues.append("Teacher role but no bytes transferred")
        
        # Check 4: Check for synthetic data markers
        if 'error' in exchange_result and 'not_implemented' in exchange_result['error']:
            issues.append("Implementation incomplete - cannot use synthetic data")
        
        # Check 5: Timing sanity check
        transfer_time = exchange_result.get('transfer_time', 0)
        if transfer_time == 0 and exchange_result.get('success', False):
            issues.append("Zero transfer time for successful exchange")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_round_results(self, round_id: int, results: List[Tuple[Any, Any]], 
                             failures: List[Any]) -> Dict[str, Any]:
        """
        Validate an entire round's results
        """
        validation_report = {
            'round_id': round_id,
            'timestamp': time.time(),
            'total_clients': len(results) + len(failures),
            'explicit_failures': len(failures),
            'integrity_issues': [],
            'synthetic_data_detected': False,
            'implementation_gaps': [],
            'is_valid': True
        }
        
        # Check each client's results
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            client_id = metrics.get('client_id', 'unknown')
            
            # Check for synthetic performance values
            local_perf = metrics.get('local_performance', 0.0)
            global_perf = metrics.get('global_performance', 0.0)
            
            # Suspicious patterns that indicate synthetic data
            if local_perf == 0.3 or local_perf == 0.5:  # Common fallback values
                validation_report['integrity_issues'].append(
                    f"Client {client_id}: Suspicious default performance value {local_perf}"
                )
                validation_report['synthetic_data_detected'] = True
            
            # Check P2P exchange integrity
            p2p_exchanges = metrics.get('p2p_exchanges', 0)
            p2p_gain = metrics.get('p2p_performance_gain', 0.0)
            
            if p2p_exchanges > 0 and p2p_gain == 0.0:
                validation_report['integrity_issues'].append(
                    f"Client {client_id}: {p2p_exchanges} exchanges but zero gain"
                )
            
            # Check for implementation gaps
            if 'not_implemented' in str(metrics):
                validation_report['implementation_gaps'].append(
                    f"Client {client_id}: Missing implementation detected"
                )
        
        # Final validation
        validation_report['is_valid'] = (
            len(validation_report['integrity_issues']) == 0 and
            len(validation_report['implementation_gaps']) == 0 and
            not validation_report['synthetic_data_detected']
        )
        
        # Log results
        if not validation_report['is_valid']:
            logger.error(f"❌ INTEGRITY VALIDATION FAILED for round {round_id}")
            logger.error(f"   Issues: {len(validation_report['integrity_issues'])}")
            logger.error(f"   Implementation gaps: {len(validation_report['implementation_gaps'])}")
            logger.error(f"   Synthetic data: {validation_report['synthetic_data_detected']}")
            
            # Log specific issues
            for issue in validation_report['integrity_issues']:
                logger.error(f"   - {issue}")
        else:
            logger.info(f"✅ Integrity validation passed for round {round_id}")
        
        self.validation_history.append(validation_report)
        return validation_report
    
    def get_integrity_summary(self) -> Dict[str, Any]:
        """
        Get summary of all integrity checks
        """
        if not self.validation_history:
            return {'status': 'no_validations_performed'}
        
        total_rounds = len(self.validation_history)
        valid_rounds = sum(1 for v in self.validation_history if v['is_valid'])
        
        all_issues = []
        for validation in self.validation_history:
            all_issues.extend(validation['integrity_issues'])
        
        summary = {
            'total_rounds_validated': total_rounds,
            'valid_rounds': valid_rounds,
            'invalid_rounds': total_rounds - valid_rounds,
            'integrity_rate': valid_rounds / total_rounds if total_rounds > 0 else 0,
            'total_issues': len(all_issues),
            'synthetic_data_rounds': sum(1 for v in self.validation_history if v['synthetic_data_detected']),
            'implementation_gap_rounds': sum(1 for v in self.validation_history if v['implementation_gaps'])
        }
        
        # Generate report
        logger.info("=" * 80)
        logger.info("RESEARCH INTEGRITY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total rounds validated: {total_rounds}")
        logger.info(f"Valid rounds: {valid_rounds} ({summary['integrity_rate']*100:.1f}%)")
        logger.info(f"Invalid rounds: {summary['invalid_rounds']}")
        logger.info(f"Total integrity issues: {summary['total_issues']}")
        logger.info(f"Rounds with synthetic data: {summary['synthetic_data_rounds']}")
        logger.info(f"Rounds with implementation gaps: {summary['implementation_gap_rounds']}")
        logger.info("=" * 80)
        
        if summary['integrity_rate'] < 1.0:
            logger.error("⚠️  WARNING: Research integrity compromised - results may not be valid for publication")
        
        return summary
# KNEXA-FL Test Suite

This directory contains guard-rail tests to ensure academic integrity and code quality for the KNEXA-FL implementation.

## Test Files

### test_unused_imports.py
- **Purpose**: Validates that no code imports from deleted/archived files
- **Coverage**: 
  - Checks all Python files in src/ for imports from deleted modules
  - Verifies all required core modules can be imported successfully
- **Run**: `python tests/test_unused_imports.py` or `pytest tests/test_unused_imports.py`

### test_metrics_snapshot.py
- **Purpose**: Runs mini-experiment to verify metrics consistency and academic integrity
- **Coverage**:
  - Executes 2-round experiment with 2 clients
  - Validates performance metrics are in reasonable ranges
  - Checks for synthetic data markers
  - Ensures no placeholder values
- **Run**: `python tests/test_metrics_snapshot.py` (requires ~10 minutes)

## CI/CD Integration

The `.github/workflows/knexa_validation.yml` workflow runs these tests automatically on:
- Every push to the repository
- Every pull request

The workflow includes:
1. **Security scanning** with bandit
2. **Import validation** tests
3. **Academic integrity** snapshot tests
4. **Code quality** checks (formatting, linting)

## Running All Tests

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all tests
pytest tests/ -v

# Run with timeout for long tests
pytest tests/ -v --timeout=600
```

## Academic Integrity Standards

These tests ensure:
- ✅ No synthetic or fabricated data
- ✅ All metrics from real model inference
- ✅ Complete dependency integrity
- ✅ Reproducible experimental results
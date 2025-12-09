"""
Test runner for Hanoi calibration tests
"""

import sys
import os
import pytest

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_calibration_tests():
    """Run all calibration-related tests"""
    print("Running Hanoi calibration tests...")
    
    # Run the calibration tests
    exit_code = pytest.main([
        'test_hanoi_calibration.py',
        '-v',
        '--tb=short',
        '--color=yes'
    ])
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_calibration_tests()
    sys.exit(exit_code)

"""
Test runner for MDAP harness
Runs all unit and integration tests
"""

import subprocess
import sys
import os

def run_tests():
    """Run all tests and return success status"""
    print("🧪 Running MDAP Test Suite")
    print("=" * 50)
    
    # Test files to run
    test_files = [
        "mdap/test_mdap_harness.py",
        "mdap/test_hanoi_integration.py"
    ]
    
    all_passed = True
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            all_passed = False
            continue
        
        print(f"\n📋 Running {test_file}...")
        print("-" * 30)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v",
                "--tb=short"
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            if result.returncode != 0:
                print(f"❌ {test_file} failed")
                all_passed = False
            else:
                print(f"✅ {test_file} passed")
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())

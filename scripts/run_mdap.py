#!/usr/bin/env python3
"""
MDAP Harness Run Script
Provides convenient commands to run MDAP examples and tests
"""

import os
import sys
import subprocess
import asyncio
import logging
from datetime import datetime
from typing import Optional

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VENV_PATH = os.path.expanduser("~/virtualenvs/mdap_harness_venv")

def print_status(message: str):
    print(f"{BLUE}[MDAP]{NC} {message}")

def print_success(message: str):
    print(f"{GREEN}[SUCCESS]{NC} {message}")

def print_warning(message: str):
    print(f"{YELLOW}[WARNING]{NC} {message}")

def print_error(message: str):
    print(f"{RED}[ERROR]{NC} {message}")

def check_venv():
    """Check if virtual environment exists"""
    if not os.path.exists(VENV_PATH):
        print_error("Virtual environment not found!")
        print("Please run './setup_mdap.sh' first to set up the environment.")
        sys.exit(1)

def activate_venv():
    """Activate virtual environment"""
    print_status("Activating virtual environment...")
    activate_script = os.path.join(VENV_PATH, "bin", "activate")
    if os.path.exists(activate_script):
        # In Python, we can't actually source the script, so we'll modify PATH
        venv_bin = os.path.join(VENV_PATH, "bin")
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        os.environ["VIRTUAL_ENV"] = VENV_PATH
    else:
        print_error(f"Virtual environment activation script not found at {activate_script}")
        sys.exit(1)

def check_env():
    """Check if .env file exists"""
    env_file = os.path.join(PROJECT_DIR, ".env")
    if not os.path.exists(env_file):
        print_warning(".env file not found!")
        print("Please create a .env file with your API keys.")
        print("You can copy it from .env.example:")
        print("  cp .env.example .env")
        print("Then edit .env with your API keys.")
        sys.exit(1)

def run_command(cmd: list, cwd: Optional[str] = None):
    """Run a command and return its result"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_DIR,
            check=True,
            capture_output=False
        )
        return result
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        sys.exit(e.returncode)

def run_unit_tests():
    """Run unit tests"""
    print_status("Running MDAP unit tests...")
    run_command([sys.executable, "-m", "pytest", "mdap/test_mdap_harness.py", "-v"])

def run_integration_tests():
    """Run integration tests"""
    print_status("Running MDAP integration tests...")
    run_command([sys.executable, "-m", "pytest", "mdap/test_hanoi_integration.py", "-v"])

def run_all_tests():
    """Run all tests"""
    print_status("Running all MDAP tests...")
    run_command([sys.executable, "mdap/test_runner.py"])

def run_example(disks: int = 2):
    """Run Hanoi example"""
    print_status(f"Running Hanoi example with {disks} disks...")
    run_command([sys.executable, "mdap/example_hanoi.py"])

def run_tests(disks: int = 3):
    """Run Hanoi test suite"""
    print_status("Running MDAP test suite...")
    run_command([sys.executable, "test_hanoi.py"])

def run_benchmark(disks: int = 3):
    """Run performance benchmark"""
    print_status(f"Running benchmark with {disks} disks...")
    benchmark_code = f"""
import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def benchmark():
    print(f'Benchmark: Testing different K margins')
    k_values = [1, 2, 3, 4]
    
    for k in k_values:
        print(f'Testing K={{k}}:')
        config = MDAPConfig(
            model='cerebras/zai-glm-4.6',
            k_margin=k,
            max_candidates=10,
            temperature=0.1
        )
        
        solver = HanoiMDAP(config)
        
        try:
            trace = await solver.solve_hanoi({disks})
            moves = trace[-1].move_count
            print(f'  ✅ K={{k}}: {{moves}} moves')
        except Exception as e:
            print(f'  ❌ K={{k}}: Failed - {{e}}')

asyncio.run(benchmark())
"""
    run_command([sys.executable, "-c", benchmark_code])

def solve_hanoi(disks: int = 3):
    """Solve Hanoi with specified disk count"""
    print_status(f"Solving Towers of Hanoi with {disks} disks...")
    
    # Setup logging
    logs_dir = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"solve_hanoi_{timestamp}.log")
    
    solve_code = f"""
import asyncio
import logging
import os
from datetime import datetime
from mdap.hanoi_solver import HanoiMDAP, MDAPConfig

# Setup logging
LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOGS_DIR, f'solve_hanoi_{{timestamp}}.log')

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def solve():
    logger.info('Starting Hanoi solver')
    logger.info('🏗️  MDAP Hanoi Solver')
    logger.info('=' * 40)
    
    config = MDAPConfig(
        model='cerebras/zai-glm-4.6',
        k_margin=3,
        max_candidates=10,
        temperature=0.1
    )
    
    logger.info(f'Created solver with config: model={{config.model}}, k_margin={{config.k_margin}}')
    solver = HanoiMDAP(config)
    
    try:
        logger.info(f'Attempting to solve {disks}-disk Towers of Hanoi')
        trace = await solver.solve_hanoi({disks})
        
        final_state = trace[-1]
        logger.info(f'Solution completed in {{final_state.move_count}} moves')
        logger.info(f'Solution trace: {{len(trace)}} states')
        logger.info(f'✅ Solved in {{final_state.move_count}} moves!')
        logger.info(f'Optimal solution: {{2**{disks} - 1}} moves')
        
        if final_state.move_count == 2**{disks} - 1:
            logger.info('🎯 Found optimal solution!')
        else:
            extra = final_state.move_count - (2**{disks} - 1)
            logger.info(f'Used {{extra}} extra moves')
        
        logger.info(f'Initial state: {{trace[0].pegs}}')
        logger.info(f'Final state:   {{trace[-1].pegs}}')
        
    except Exception as e:
        logger.error(f'Failed to solve Hanoi: {{e}}')
        logger.error('Possible causes: API key issue, model unavailable, network problems, rate limiting')
        raise

try:
    asyncio.run(solve())
except Exception as e:
    print(f'ERROR: {{e}}')
    import traceback
    traceback.print_exc()
"""
    run_command([sys.executable, "-c", solve_code])

def start_interactive():
    """Start interactive Python session"""
    print_status("Starting interactive Python session...")
    print("MDAP modules are imported. Available:")
    print("  - MDAPHarness, MDAPConfig")
    print("  - HanoiMDAP, HanoiState")
    print("  - RedFlagParser")
    print("")
    
    interactive_code = """
from mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser
from hanoi_solver import HanoiMDAP, HanoiState
print('✅ MDAP modules loaded successfully!')
print('Type help() for available functions or exit() to quit.')
import code
code.interact(local=dict(globals(), **locals()))
"""
    run_command([sys.executable, "-c", interactive_code])

def show_help():
    """Show help message"""
    print("MDAP Harness Run Script")
    print("")
    print("Usage: python run_mdap.py [COMMAND] [OPTIONS]")
    print("")
    print("Commands:")
    print("  example [disks]    Run Hanoi example (default: 3 disks)")
    print("  test [disks]       Run Hanoi test suite (default: 3,4 disks)")
    print("  unit               Run unit tests only")
    print("  integration        Run integration tests only")
    print("  test-all           Run all unit and integration tests")
    print("  benchmark [disks]  Run performance benchmark (default: 3 disks)")
    print("  solve [disks]      Solve Hanoi with specified disk count")
    print("  interactive        Start interactive Python session with MDAP loaded")
    print("  help               Show this help message")
    print("")
    print("Examples:")
    print("  python run_mdap.py example 3       # Run 3-disk example")
    print("  python run_mdap.py test            # Run all tests")
    print("  python run_mdap.py unit            # Run unit tests only")
    print("  python run_mdap.py integration     # Run integration tests only")
    print("  python run_mdap.py test-all        # Run comprehensive test suite")
    print("  python run_mdap.py solve 5         # Solve 5-disk Hanoi")
    print("  python run_mdap.py benchmark       # Run benchmark")

def main():
    """Main entry point"""
    # Check prerequisites
    check_venv()
    activate_venv()
    check_env()
    
    # Parse command
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    
    if command == "example":
        disks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        run_example(disks)
    elif command == "test":
        disks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_tests(disks)
    elif command == "unit":
        run_unit_tests()
    elif command == "integration":
        run_integration_tests()
    elif command == "test-all":
        run_all_tests()
    elif command == "benchmark":
        disks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        run_benchmark(disks)
    elif command == "solve":
        disks = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        solve_hanoi(disks)
    elif command == "interactive":
        start_interactive()
    elif command in ["help", "-h", "--help"]:
        show_help()
    else:
        print_error(f"Unknown command: {command}")
        print("")
        show_help()
        sys.exit(1)
    
    print_success("Done!")

if __name__ == "__main__":
    main()

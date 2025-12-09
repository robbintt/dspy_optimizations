#!/bin/bash

# MDAP Harness Run Script
# Provides convenient commands to run MDAP examples and tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$HOME/virtualenvs/mdap_harness_venv"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[MDAP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if virtual environment exists
check_venv() {
    if [ ! -d "$VENV_PATH" ]; then
        print_error "Virtual environment not found!"
        echo "Please run './setup_mdap.sh' first to set up the environment."
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
}

# Function to check if .env file exists
check_env() {
    if [ ! -f ".env" ]; then
        print_warning ".env file not found!"
        echo "Please create a .env file with your API keys."
        echo "You can copy it from .env.example:"
        echo "  cp .env.example .env"
        echo "Then edit .env with your API keys."
        exit 1
    fi
}

# Function to run unit tests
run_unit_tests() {
    print_status "Running MDAP unit tests..."
    python -m pytest test_mdap_harness.py -v
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running MDAP integration tests..."
    python -m pytest test_hanoi_integration.py -v
}

# Function to run all tests
run_all_tests() {
    print_status "Running all MDAP tests..."
    python test_runner.py
}

# Function to show help
show_help() {
    echo "MDAP Harness Run Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  example [disks]    Run Hanoi example (default: 3 disks)"
    echo "  test [disks]       Run Hanoi test suite (default: 3,4 disks)"
    echo "  unit               Run unit tests only"
    echo "  integration        Run integration tests only"
    echo "  test-all           Run all unit and integration tests"
    echo "  benchmark [disks]  Run performance benchmark (default: 3 disks)"
    echo "  solve [disks]      Solve Hanoi with specified disk count"
    echo "  interactive        Start interactive Python session with MDAP loaded"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 example 3       # Run 3-disk example"
    echo "  $0 test            # Run all tests"
    echo "  $0 unit            # Run unit tests only"
    echo "  $0 integration     # Run integration tests only"
    echo "  $0 test-all        # Run comprehensive test suite"
    echo "  $0 solve 5         # Solve 5-disk Hanoi"
    echo "  $0 benchmark       # Run benchmark"
}

# Function to run example
run_example() {
    local disks=${1:-2}
    print_status "Running Hanoi example with $disks disks..."
    python example_hanoi.py
}

# Function to run tests
run_tests() {
    print_status "Running MDAP test suite..."
    python test_hanoi.py
}

# Function to run benchmark
run_benchmark() {
    local disks=${1:-3}
    print_status "Running benchmark with $disks disks..."
    python -c "
import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def benchmark():
    print(f'Benchmark: Testing different K margins')
    k_values = [1, 2, 3, 4]
    
    for k in k_values:
        print(f'Testing K={k}:')
        config = MDAPConfig(
            model='cerebras/zai-glm-4.6',
            k_margin=k,
            max_candidates=10,
            temperature=0.1
        )
        
        solver = HanoiMDAP(config)
        
        try:
            trace = await solver.solve_hanoi($disks)
            moves = trace[-1].move_count
            print(f'  ✅ K={k}: {moves} moves')
        except Exception as e:
            print(f'  ❌ K={k}: Failed - {e}')

asyncio.run(benchmark())
"
}

# Function to solve specific disk count
solve_hanoi() {
    local disks=${1:-3}
    print_status "Solving Towers of Hanoi with $disks disks..."
    python -c "
import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def solve():
    config = MDAPConfig(
        model='cerebras/zai-glm-4.6',
        k_margin=3,
        max_candidates=10,
        temperature=0.1
    )
    
    solver = HanoiMDAP(config)
    trace = await solver.solve_hanoi($disks)
    
    final_state = trace[-1]
    print(f'✅ Solved $disks-disk Hanoi in {final_state.move_count} moves!')
    print(f'Optimal solution: {2**$disks - 1} moves')
    
    if final_state.move_count == 2**$disks - 1:
        print('🎯 Found optimal solution!')
    else:
        extra = final_state.move_count - (2**$disks - 1)
        print(f'Used {extra} extra moves')

asyncio.run(solve())
"
}

# Function to start interactive session
start_interactive() {
    print_status "Starting interactive Python session..."
    print "MDAP modules are imported. Available:"
    print "  - MDAPHarness, MDAPConfig"
    print "  - HanoiMDAP, HanoiState"
    print "  - RedFlagParser"
    echo ""
    python -c "
from mdap_harness import MDAPHarness, MDAPConfig, RedFlagParser
from hanoi_solver import HanoiMDAP, HanoiState
print('✅ MDAP modules loaded successfully!')
print('Type help() for available functions or exit() to quit.')
import code
code.interact(local=dict(globals(), **locals()))
"
}

# Main script logic
main() {
    # Check prerequisites
    check_venv
    activate_venv
    check_env
    
    # Parse command
    case "${1:-help}" in
        "example")
            run_example "$2"
            ;;
        "test")
            run_tests "$2"
            ;;
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "test-all")
            run_all_tests
            ;;
        "benchmark")
            run_benchmark "$2"
            ;;
        "solve")
            solve_hanoi "$2"
            ;;
        "interactive")
            start_interactive
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
    
    print_success "Done!"
}

# Run main function with all arguments
main "$@"

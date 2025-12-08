#!/bin/bash

# MDAP Test Runner Script
# Provides convenient commands to run MDAP tests

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
    echo -e "${BLUE}[TEST]${NC} $1"
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

# Function to show help
show_help() {
    echo "MDAP Test Runner Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  unit               Run unit tests only"
    echo "  integration        Run integration tests only"
    echo "  all                Run all unit and integration tests"
    echo "  coverage           Run tests with coverage report"
    echo "  specific [test]    Run specific test file or test"
    echo "  watch              Run tests in watch mode"
    echo "  verbose            Run tests with verbose output"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 unit            # Run unit tests only"
    echo "  $0 integration     # Run integration tests only"
    echo "  $0 all             # Run all tests"
    echo "  $0 specific test_mdap_harness.py::TestRedFlagParser"
    echo "  $0 coverage        # Run with coverage report"
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

# Function to run tests with coverage
run_coverage_tests() {
    print_status "Running tests with coverage report..."
    
    # Check if coverage is installed
    if ! python -c "import coverage" 2>/dev/null; then
        print_status "Installing coverage..."
        pip install coverage
    fi
    
    # Run coverage
    coverage run -m pytest test_mdap_harness.py test_hanoi_integration.py -v
    coverage report -m
    coverage html
    
    print_success "Coverage report generated in htmlcov/"
}

# Function to run specific test
run_specific_test() {
    local test_spec="$1"
    
    if [ -z "$test_spec" ]; then
        print_error "Please specify a test file or test"
        echo "Example: $0 specific test_mdap_harness.py"
        echo "Example: $0 specific test_mdap_harness.py::TestRedFlagParser::test_valid_move_response"
        exit 1
    fi
    
    print_status "Running specific test: $test_spec"
    python -m pytest "$test_spec" -v
}

# Function to run tests in watch mode
run_watch_tests() {
    print_status "Running tests in watch mode..."
    
    # Check if pytest-watch is installed
    if ! python -c "import pytest_watch" 2>/dev/null; then
        print_status "Installing pytest-watch..."
        pip install pytest-watch
    fi
    
    ptw test_mdap_harness.py test_hanoi_integration.py --runner "python -m pytest -v"
}

# Function to run tests with verbose output
run_verbose_tests() {
    print_status "Running tests with verbose output..."
    python -m pytest test_mdap_harness.py test_hanoi_integration.py -v -s --tb=long
}

# Function to run tests with specific markers
run_marked_tests() {
    local marker="$1"
    
    if [ -z "$marker" ]; then
        print_error "Please specify a marker"
        echo "Available markers: unit, integration, slow"
        exit 1
    fi
    
    print_status "Running tests with marker: $marker"
    python -m pytest -m "$marker" -v
}

# Function to check test dependencies
check_dependencies() {
    print_status "Checking test dependencies..."
    
    # Check if pytest is installed
    if ! python -c "import pytest" 2>/dev/null; then
        print_error "pytest is not installed. Please run: pip install pytest pytest-asyncio"
        exit 1
    fi
    
    # Check if pytest-asyncio is installed
    if ! python -c "import pytest_asyncio" 2>/dev/null; then
        print_error "pytest-asyncio is not installed. Please run: pip install pytest pytest-asyncio"
        exit 1
    fi
    
    print_success "All test dependencies are available"
}

# Function to run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    # Run with different configurations
    echo "Testing different K margins..."
    for k in 1 2 3 4; do
        echo "Testing K=$k..."
        python -c "
import asyncio
from hanoi_solver import HanoiMDAP, MDAPConfig

async def test_k(k):
    config = MDAPConfig(k_margin=k, max_candidates=5)
    solver = HanoiMDAP(config)
    try:
        trace = await solver.solve_hanoi(3)
        print(f'K={k}: {trace[-1].move_count} moves')
    except Exception as e:
        print(f'K={k}: Failed - {e}')

asyncio.run(test_k($k))
"
    done
}

# Main script logic
main() {
    # Check prerequisites
    check_venv
    activate_venv
    check_dependencies
    
    # Parse command
    case "${1:-help}" in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "all")
            run_all_tests
            ;;
        "coverage")
            run_coverage_tests
            ;;
        "specific")
            run_specific_test "$2"
            ;;
        "watch")
            run_watch_tests
            ;;
        "verbose")
            run_verbose_tests
            ;;
        "marked")
            run_marked_tests "$2"
            ;;
        "performance")
            run_performance_tests
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
    
    print_success "Test execution completed!"
}

# Run main function with all arguments
main "$@"

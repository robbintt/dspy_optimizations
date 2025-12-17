#!/bin/bash

# A reusable shell script to run the GEPA optimization.
# Ensure you are in the correct directory (project root) and your Python
# environment is activated before running this script.

# --- Configuration & Defaults ---
# The absolute or relative path to the Python runner script
RUNNER_SCRIPT="gepa_self_optimizer/run_gepa_optimization.py"

# Default GEPA profile
DEFAULT_PROFILE="development"
# Default data file
DEFAULT_DATA_FILE="golden_set.json"
# Default output file
DEFAULT_OUTPUT_FILE="optimized_program.json"
# Default metric function
DEFAULT_METRIC="gepa_config.refinement_gepa_metric"

# --- Argument Parsing ---
# Initialize variables with defaults
PROFILE="$DEFAULT_PROFILE"
OVERRIDE_AUTO=""
STUDENT_MODULE=""
METRIC="$DEFAULT_METRIC"
DATA_FILE="$DEFAULT_DATA_FILE"
OUTPUT_FILE="$DEFAULT_OUTPUT_FILE"

usage() {
    echo "Usage: $0 [-p profile] [-a auto_override] -s student_module [-m metric] -d data_file -o output_file"
    echo "  -p: GEPA profile to use (e.g., development, medium, heavy). Default: '$DEFAULT_PROFILE'."
    echo "  -a: Override budget setting (light, medium, heavy)."
    echo "  -s: (Required) Import path to the dspy.Module to optimize (e.g., 'gepa_system.GlmSelfReflect')."
    echo "  -m: Import path to the metric function. Default: '$DEFAULT_METRIC'."
    echo "  -d: (Required) Path to the training data JSON file. Default: '$DEFAULT_DATA_FILE'."
    echo "  -o: (Required) Path to save the final optimized program. Default: '$DEFAULT_OUTPUT_FILE'."
    echo "  -h: Display this help message."
    exit 1
}

# Parse options using getopts
while getopts ":p:a:s:m:d:o:h" opt; do
    case ${opt} in
        p )
            PROFILE=$OPTARG
            ;;
        a )
            OVERRIDE_AUTO=$OPTARG
            ;;
        s )
            STUDENT_MODULE=$OPTARG
            ;;
        m )
            METRIC=$OPTARG
            ;;
        d )
            DATA_FILE=$OPTARG
            ;;
        o )
            OUTPUT_FILE=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            echo "Invalid option: -$OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done

# Check for mandatory arguments
if [ -z "$STUDENT_MODULE" ] || [ -z "$DATA_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Error: Missing required arguments."
    usage
fi

# --- Build and Execute Python Command ---
echo "--- GEPA Run Configuration ---"
echo "Profile:           $PROFILE"
echo "Auto Override:     ${OVERRIDE_AUTO:-'(none)'}"
echo "Student Module:    $STUDENT_MODULE"
echo "Metric Function:   $METRIC"
echo "Data File:         $DATA_FILE"
echo "Output File:       $OUTPUT_FILE"
echo "------------------------------"

PYTHON_CMD="python $RUNNER_SCRIPT --profile $PROFILE --student_module $STUDENT_MODULE --metric $METRIC --data_file $DATA_FILE --output_file $OUTPUT_FILE"

if [ -n "$OVERRIDE_AUTO" ]; then
    PYTHON_CMD="$PYTHON_CMD --override_auto $OVERRIDE_AUTO"
fi

echo "Executing command:"
echo $PYTHON_CMD
echo "------------------------------"

# Execute the command
exec $PYTHON_CMD

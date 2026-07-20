#!/bin/bash

# ==============================================================================
# Script: run_batch_frequentist_dark_portal.sh
# Usage: ./run_batch_frequentist_dark_portal.sh <input_target>
#
# Arguments:
#   <input_target>: Can be one of three things:
#       1. A single JSON config file (e.g., config.json).
#       2. A directory containing multiple JSON config files.
#       3. A text file containing a list of JSON file paths (one per line).
#
# Note: This script runs in serial. It extracts the 'output_dir' from the JSON.
# ==============================================================================

# Updated to target your main orchestrator script
SCRIPT_TO_RUN="src/main.py"

# 1. Validate Arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_target>"
    echo "Examples:"
    echo "  $0 configs/run1.json        (Runs single analysis)"
    echo "  $0 ./configs                (Scans folder for .json files)"
    echo "  $0 analysis_list.txt        (Runs paths listed in text file)"
    exit 1
fi

INPUT_PATH="$1"

# 2. Check if Python script exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "Error: Python script '$SCRIPT_TO_RUN' not found in current directory."
    exit 1
fi

# 3. Check if input path exists
if [ ! -e "$INPUT_PATH" ]; then
    echo "Error: Input target '$INPUT_PATH' does not exist."
    exit 1
fi

# Function to execute the job
run_job() {
    local config_file="$1"
    
    # Validation: Must be a valid file
    if [ ! -f "$config_file" ]; then
        echo "  [SKIP] '$config_file' not found or is not a file."
        return
    fi

    # Extract output_dir from the JSON file using sed
    # Looks for "output_dir": "VALUE"
    local output_dir=$(sed -n 's/.*"output_dir": *"\([^"]*\)".*/\1/p' "$config_file")

    if [ -z "$output_dir" ]; then
        echo "  [SKIP] Could not parse 'output_dir' from $config_file."
        return
    fi

    echo "=================================================================="
    echo "  Starting Frequentist Analysis (Dark Portal)"
    echo "  Config: $config_file"
    echo "  Output: $output_dir"
    echo "=================================================================="
    
    # Run the command with both required arguments
    # python -u -m "$SCRIPT_TO_RUN" --config_file="$config_file" --output_dir="$output_dir"
    python -u -m src.main --config_file="$config_file" --output_dir="$output_dir"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "  [SUCCESS] Analysis finished."
    else
        echo "  [FAILURE] Error occurred during analysis."
    fi
    echo ""
}

# 4. Main Logic Switch
if [ -d "$INPUT_PATH" ]; then
    # --- CASE A: DIRECTORY OF CONFIGS ---
    echo "Input identified as a directory. Scanning for .json files..."
    
    # Use nullglob to handle empty directories safely
    shopt -s nullglob
    count=0
    
    # Iterate over all .json files in the directory
    for config in "$INPUT_PATH"/*.json; do
        run_job "$config"
        ((count++))
    done
    shopt -u nullglob
    
    if [ "$count" -eq 0 ]; then
        echo "No .json files found in $INPUT_PATH"
    fi

elif [ -f "$INPUT_PATH" ]; then
    # Check extension to decide between Single Run or List File
    filename=$(basename -- "$INPUT_PATH")
    extension="${filename##*.}"

    if [ "$extension" == "json" ]; then
        # --- CASE B: SINGLE CONFIG FILE ---
        echo "Input identified as a single JSON config file."
        run_job "$INPUT_PATH"
    else
        # --- CASE C: LIST FILE ---
        echo "Input identified as a list of paths."
        
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments (starting with #)
            [[ -z "$line" ]] && continue
            [[ "$line" =~ ^#.*$ ]] && continue
            
            # Trim whitespace (if any)
            clean_path=$(echo "$line" | xargs)
            
            run_job "$clean_path"
        done < "$INPUT_PATH"
    fi

else
    echo "Error: Input type not recognized."
    exit 1
fi

echo "Batch processing complete."
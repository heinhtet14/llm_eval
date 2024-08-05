#!/bin/bash

# Directory containing JSON files
#JSON_DIR="/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/data/llama3/"
JSON_DIR="/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/data/phi3/llm_response/phase2"

# Python script name
PYTHON_SCRIPT="/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/src/benchmark_v1.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Check if directory exists
if [ ! -d "$JSON_DIR" ]; then
    echo "Error: Directory '$JSON_DIR' not found!"
    exit 1
fi

# Process each JSON file in the directory
for json_file in "$JSON_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        echo "Processing $json_file..."
        
        # Get the directory of the current JSON file
        file_dir=$(dirname "$json_file")
        
        # Get the base name of the JSON file (without extension)
        base_name=$(basename "$json_file" .json)
        
        # Run the Python script with the current JSON file
        python "$PYTHON_SCRIPT" "$json_file"
        
        # Move the result file to the same directory as the input JSON file
        mv evaluation_results_*.json "${file_dir}/${base_name}_evaluation_results.json"
        
        echo "Finished processing $json_file"
        echo "Results saved in ${file_dir}/${base_name}_evaluation_results.json"
        echo "----------------------------------------"
    fi
done

echo "All files processed."
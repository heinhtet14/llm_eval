#!/bin/bash

# Directory containing JSON files
json_dir="/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/data/llama3/"

# Python script name
python_script="/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/pyrogue/rogue.py"

# Check if Python script exists
if [ ! -f "$python_script" ]; then
    echo "Error: Python script '$python_script' not found!"
    exit 1
fi

# Check if directory exists
if [ ! -d "$json_dir" ]; then
    echo "Error: Directory '$json_dir' not found!"
    exit 1
fi

# Process each JSON file in the directory
for json_file in "$json_dir"*.json; do
    if [ -f "$json_file" ]; then
        echo "Processing $json_file..."
        python "$python_script" "$json_file"
        echo "Finished processing $json_file"
        echo "----------------------------------------"
    else
        echo "Warning: No JSON files found in $json_dir"
        exit 1
    fi
done

echo "All files processed."
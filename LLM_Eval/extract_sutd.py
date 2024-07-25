import json

def extract_interview_data(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract id and interview_data from the segmented_data
    extracted_data = []
    for item in data:
        if 'segmented_data' in item:
            for segment in item['segmented_data']:
                extracted_data.append({
                    "id": segment["id"],
                    "interview_data": segment["interview_data"]
                })

    # Write the extracted data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(extracted_data, f, indent=2)

    print(f"Extracted data has been saved to {output_file}")

# Example usage
input_file = "QA_SUTD.json"
output_file = "extracted_interview_data.json"
extract_interview_data(input_file, output_file)
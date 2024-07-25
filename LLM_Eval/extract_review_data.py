import json

def extract_review_data(input_file, output_file):
    neg_reviews = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:  # Skip empty lines
                neg_reviews.append({
                    "id": i,
                    "review": line
                })
   
    output_data = {
        "pos_reviews": neg_reviews
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Extracted data has been saved to {output_file}")

input_file = "TrainingDataPositive.txt"
output_file = "TrainingDataPositive.json"
extract_review_data(input_file, output_file)
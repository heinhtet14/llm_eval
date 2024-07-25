import json
from rouge import Rouge

# Read the JSON file
with open('response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

# Extract Version1
version1 = data['versions']['Version3']

print("Version3:", version1)


reference = "Comprehensive guide to business basics"
# Calculate ROUGE scores
rouge = Rouge()
scores = rouge.get_scores(version1, reference)

# Print the scores
print("\nROUGE Scores with 2nd Coder:")
for metric, score in scores[0].items():
    print(f"{metric}: {score}")
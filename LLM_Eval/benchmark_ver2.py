import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import numpy as np

def evaluate_short_summary(summary, reference):
    # Initialize SentenceBERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode sentences
    summary_embedding = model.encode([summary])
    reference_embedding = model.encode([reference])

    # Calculate semantic similarity
    semantic_similarity = cosine_similarity(summary_embedding, reference_embedding)[0][0]

    # Calculate BERTScore
    P, R, F1 = score([summary], [reference], lang='en', verbose=False)

    # Simple concept coverage (based on word overlap)
    summary_words = set(summary.lower().split())
    reference_words = set(reference.lower().split())
    concept_coverage = len(summary_words.intersection(reference_words)) / len(summary_words)

    # Information density (ratio of unique words to total words)
    info_density = len(set(summary.split())) / len(summary.split())

    return {
        "Semantic_Similarity": semantic_similarity,
        "BERTScore_F1": F1.item(),
        "Concept_Coverage": concept_coverage,
        "Information_Density": info_density
    }

# Read the JSON file
with open('data\\response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

# Evaluate each version against the review
results = {}
for version_key in ['Version1', 'Version2', 'Version3']:
    results[version_key] = evaluate_short_summary(data[version_key], data['review'])

# Print results
print(json.dumps(results, indent=2))

# Optionally, save results to a file
with open('short_summary_evaluation_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)

# Print summaries for reference
print("\nSummaries:")
for version_key in ['Version1', 'Version2', 'Version3']:
    print(f"{version_key}: {data[version_key]}")

print("\nOriginal Review:")
print(data['review'])
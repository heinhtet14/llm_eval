import json
import nltk
from nltk.translate import meteor_score
from nltk import word_tokenize

nltk.download('wordnet')
with open('response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

version1 = data['versions']['Version1']

reference = """Comprehensive guide to business basics"""

reference_tokens = word_tokenize(reference)
summary_tokens = word_tokenize(version1)

# Calculate METEOR score
meteor = meteor_score.meteor_score([reference_tokens], summary_tokens)

print(f"Version1: {version1}")
print(f"Reference: {reference}")
print(f"METEOR score: {meteor}")
import json
import nltk
from blanc import BlancHelp, BlancTune

nltk.download('punkt')

with open('response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

# Extract Version1
version1 = data['versions']['Version1']
print(version1)

reference = """Comprehensive guide to business basics"""

# Initialize BLANC
blanc_help = BlancHelp()
blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)

# Calculate BLANC scores
blanc_help_score = blanc_help.eval_once(reference, version1)
blanc_tune_score = blanc_tune.eval_once(reference, version1)

# Results
print(f"Version1: {version1}")
print(f"Reference: {reference}")
print(f"BLANC-help score: {blanc_help_score}")
print(f"BLANC-tune score: {blanc_tune_score}")
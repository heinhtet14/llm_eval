import json
import nltk
from blanc import BlancHelp, BlancTune
from nltk.translate import meteor_score
from nltk import word_tokenize
from rouge import Rouge

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

def evaluate_summary(summary, reference):
    # Initialize metrics
    blanc_help = BlancHelp()
    blanc_tune = BlancTune(finetune_mask_evenly=False, show_progress_bar=False)
    rouge = Rouge()

    # Calculate BLANC scores
    blanc_help_score = blanc_help.eval_once(reference, summary)
    blanc_tune_score = blanc_tune.eval_once(reference, summary)

    # Calculate METEOR score
    reference_tokens = word_tokenize(reference)
    summary_tokens = word_tokenize(summary)
    meteor = meteor_score.meteor_score([reference_tokens], summary_tokens)

    # Calculate ROUGE scores
    rouge_scores = rouge.get_scores(summary, reference)[0]

    return {
        "BLANC-help": blanc_help_score,
        "BLANC-tune": blanc_tune_score,
        "METEOR": meteor,
        "ROUGE-1": rouge_scores['rouge-1'],
        "ROUGE-2": rouge_scores['rouge-2'],
        "ROUGE-L": rouge_scores['rouge-l']
    }

# Read the JSON file
with open('/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/data/response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

# Evaluate each version against the review
results = {}
for version_key in ['Version1', 'Version2', 'Version3']:
    results[version_key] = evaluate_summary(data[version_key], data['review'])

# Print results
print(json.dumps(results, indent=2))

# Optionally, save results to a file
with open('evaluation_results.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)

# Print summaries for reference
print("\nSummaries:")
for version_key in ['Version1', 'Version2', 'Version3']:
    print(f"{version_key}: {data[version_key]}")

print("\nOriginal Review:")
print(data['review'])
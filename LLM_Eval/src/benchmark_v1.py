import json
import sys
from datetime import datetime
import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
from blanc import BlancHelp, BlancTune
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from bert_score import BERTScorer
import numpy as np

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def calculate_rouge(summary, reference):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)[0]
    return {
        "ROUGE-1": float(scores['rouge-1']['f']),
        "ROUGE-2": float(scores['rouge-2']['f']),
        "ROUGE-L": float(scores['rouge-l']['f'])
    }

def calculate_meteor(summary, reference):
    reference_tokens = word_tokenize(reference)
    summary_tokens = word_tokenize(summary)
    return {"METEOR": float(meteor_score.meteor_score([reference_tokens], summary_tokens))}

def calculate_blanc(summary, reference):
    blanc_help = BlancHelp()
    blanc_tune = BlancTune(finetune_mask_evenly=False)
    return {
        "BLANC-help": float(blanc_help.eval_once(reference, summary)),
        "BLANC-tune": float(blanc_tune.eval_once(reference, summary))
    }

def calculate_bertscore(summary, reference):
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score([summary], [reference])
    return {"BERTScore": float(F1.item())}

def semantic_similarity(summary, reference):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    summary_embedding = model.encode([summary])
    reference_embedding = model.encode([reference])
    return float(cosine_similarity(summary_embedding, reference_embedding)[0][0])

def sentiment_match(summary, reference):
    nlp = spacy.load("en_core_web_sm")
    doc_summary = nlp(summary)
    doc_reference = nlp(reference)
    return float(1 - abs(doc_summary.sentiment - doc_reference.sentiment))

def key_concept_coverage(summary, reference):
    nlp = spacy.load("en_core_web_sm")
    doc_reference = nlp(reference)
    key_concepts = set([token.lemma_ for token in doc_reference if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
    doc_summary = nlp(summary)
    summary_concepts = set([token.lemma_ for token in doc_summary if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
    
    coverage = len(summary_concepts.intersection(key_concepts)) / len(key_concepts) if key_concepts else 0
    return float(coverage)

def evaluate_summary(summary, reference):
    metrics = {}
    metrics.update(calculate_rouge(summary, reference))
    metrics.update(calculate_meteor(summary, reference))
    metrics.update(calculate_blanc(summary, reference))
    metrics.update(calculate_bertscore(summary, reference))
    metrics.update({
        "Semantic Similarity": semantic_similarity(summary, reference),
        "Sentiment Match": sentiment_match(summary, reference),
        "Key Concept Coverage": key_concept_coverage(summary, reference)
    })
    return metrics

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def main():
    # Load the JSON file
    if len(sys.argv) != 2:
        print("Usage: python summary_evaluation.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
        
    with open(input_file, 'r') as file:
        data = json.load(file)

    # List to store results for all reviews
    all_results = []
    highest_scores = []

    # Process each review and its summaries
    for i, item in enumerate(data):
        review_result = {
            "review_id": i + 1,
            "review_text": item['review'],
            "summaries": {}
        }
        
        # Dictionary to store the highest scores for this review
        review_highest_scores = {}

        for version in ['Version1', 'Version2', 'Version3']:
            metrics = evaluate_summary(item[version], item['review'])
            review_result["summaries"][version] = {
                "text": item[version],
                "metrics": metrics
            }
            
            for metric, score in metrics.items():
                # Update the highest score for this metric
                if metric not in review_highest_scores or score > review_highest_scores[metric]:
                    review_highest_scores[metric] = score

        review_result["highest_scores"] = review_highest_scores
        all_results.append(review_result)
        highest_scores.append(review_highest_scores)

    # Calculate average of highest scores across all reviews
    avg_highest_scores = {}
    num_reviews = len(data)

    for metric in highest_scores[0].keys():
        total = sum(review[metric] for review in highest_scores)
        avg_highest_scores[metric] = total / num_reviews

    # Prepare final results dictionary
    final_results = {
        "detailed_results": all_results,
        "average_highest_scores": avg_highest_scores
    }

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'summary_evaluation_results_{timestamp}.json'
    with open(output_filename, 'w') as outfile:
        json.dump(final_results, outfile, indent=2, cls=NumpyEncoder)

    print(f"Results have been saved to {output_filename}")

    # Optionally, print average highest scores to console
    print("\n--- Average of Highest Scores Across All Reviews ---")
    for metric, avg_score in avg_highest_scores.items():
        print(f"{metric}: {avg_score:.4f}")

if __name__ == "__main__":
    main()
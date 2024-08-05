import json
import nltk
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy
from datetime import datetime

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def relevance_score(code, full_text):
    code_embedding = sentence_model.encode([code])[0]
    text_embedding = sentence_model.encode([full_text])[0]
    return 1 - cosine(code_embedding, text_embedding)

def informativeness_score(code, full_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([full_text, code])
    feature_names = vectorizer.get_feature_names_out()
    full_text_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    code_scores = dict(zip(feature_names, tfidf_matrix.toarray()[1]))
    return sum(code_scores.values()) / sum(full_text_scores.values())

def sentiment_accuracy(code, full_text):
    sia = SentimentIntensityAnalyzer()
    code_sentiment = sia.polarity_scores(code)['compound']
    text_sentiment = sia.polarity_scores(full_text)['compound']
    return 1 - abs(code_sentiment - text_sentiment)

def specificity_score(code, full_text):
    stop_words = set(stopwords.words('english'))
    code_words = [w for w in word_tokenize(code.lower()) if w not in stop_words]
    text_words = [w for w in word_tokenize(full_text.lower()) if w not in stop_words]
    common_words = set(code_words).intersection(set(text_words))
    return len(common_words) / len(code_words) if code_words else 0

def clarity_score(code):
    doc = nlp(code)
    word_count = len([token for token in doc if not token.is_punct])
    if word_count == 0:
        return 0
    avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / word_count
    return 1 / (1 + avg_word_length)  # Higher for shorter average word length

def evaluate_short_code_automated(code, full_text):
    return {
        "Relevance": relevance_score(code, full_text),
        "Informativeness": informativeness_score(code, full_text),
        "Sentiment Accuracy": sentiment_accuracy(code, full_text),
        "Specificity": specificity_score(code, full_text),
        "Clarity": clarity_score(code)
    }

def interpret_scores(scores):
    interpretations = {
        "Relevance": ["Not Relevant", "Slightly Relevant", "Moderately Relevant", "Highly Relevant"],
        "Informativeness": ["Not Informative", "Minimally Informative", "Somewhat Informative", "Very Informative"],
        "Sentiment Accuracy": ["Completely Opposite", "Mismatches", "Neutral", "Mostly Matches", "Matches Perfectly"],
        "Specificity": ["Too General", "Moderately General", "Appropriately Specific", "Too Specific"],
        "Clarity": ["Very Unclear", "Somewhat Unclear", "Mostly Clear", "Very Clear"]
    }
    
    results = {}
    for metric, score in scores.items():
        index = min(int(score * len(interpretations[metric])), len(interpretations[metric]) - 1)
        results[metric] = interpretations[metric][index]
    return results

def calculate_average_highest_scores(results):
    total_highest_scores = {
        "Relevance": 0,
        "Informativeness": 0,
        "Sentiment Accuracy": 0,
        "Specificity": 0,
        "Clarity": 0
    }
    
    review_count = len(results)
    
    for review in results:
        for metric in total_highest_scores.keys():
            highest_score = max(
                review["versions"]["Version1"]["scores"][metric],
                review["versions"]["Version2"]["scores"][metric],
                review["versions"]["Version3"]["scores"][metric]
            )
            total_highest_scores[metric] += highest_score
    
    average_highest_scores = {
        metric: score / review_count
        for metric, score in total_highest_scores.items()
    }
    
    return average_highest_scores

def process_reviews(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    results = []
    for i, review in enumerate(data):
        review_result = {
            "review_id": i + 1,
            "review_text": review["review"],
            "versions": {}
        }
        for version in ["Version1", "Version2", "Version3"]:
            scores = evaluate_short_code_automated(review[version], review["review"])
            interpretations = interpret_scores(scores)
            review_result["versions"][version] = {
                "text": review[version],
                "scores": scores,
                "interpretations": interpretations
            }
        results.append(review_result)
    
    average_highest_scores = calculate_average_highest_scores(results)
    
    return results, average_highest_scores

def save_results(results, average_highest_scores, output_file):
    output_data = {
        "individual_reviews": results,
        "average_highest_scores": average_highest_scores
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_reviews.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.json"
    
    results, average_highest_scores = process_reviews(input_file)
    save_results(results, average_highest_scores, output_file)
    print(f"Results saved to {output_file}")
    
    print("\nAverage Highest Scores:")
    for metric, score in average_highest_scores.items():
        print(f"  {metric}: {score:.4f}")
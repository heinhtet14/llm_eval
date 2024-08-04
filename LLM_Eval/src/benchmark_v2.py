import nltk
import json
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import spacy

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
    relevance = relevance_score(code, full_text)
    informativeness = informativeness_score(code, full_text)
    sentiment_acc = sentiment_accuracy(code, full_text)
    specificity = specificity_score(code, full_text)
    clarity = clarity_score(code)

    return {
        "Relevance": relevance,
        "Informativeness": informativeness,
        "Sentiment Accuracy": sentiment_acc,
        "Specificity": specificity,
        "Clarity": clarity
    }
    
# Function to interpret scores
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

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python summary_evaluation.py <input_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
        
    with open(input_file, 'r') as file:
        data = json.load(file)
    # Example usage
    full_text = "This restaurant offers a wide variety of delicious dishes. The service was excellent and the atmosphere was cozy. Highly recommended for a nice dinner out."
    code = "Excellent dining experience"

    results = evaluate_short_code_automated(code, full_text)
    print("\nAutomated Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    interpreted_results = interpret_scores(results)
    print("\nInterpreted Results:")
    for metric, interpretation in interpreted_results.items():
        print(f"{metric}: {interpretation}")
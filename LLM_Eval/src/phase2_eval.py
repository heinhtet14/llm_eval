import ijson
import ollama
import json
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_reviews(file_path, num_reviews=100):
    reviews = []
    try:
        with open(file_path, 'rb') as file:
            parser = ijson.parse(file)
            current_item = {}
            for prefix, event, value in parser:
                if prefix.endswith('.id'):
                    if current_item:
                        reviews.append(current_item)
                        if len(reviews) == num_reviews:
                            break
                    current_item = {'id': value}
                elif prefix.endswith('.interview_data'):
                    current_item['interview_data'] = value
                elif prefix.endswith('.code1'):
                    current_item['code1'] = value
                elif prefix.endswith('.code2'):
                    current_item['code2'] = value
            if current_item:
                reviews.append(current_item)
    except Exception as e:
        logging.error(f"Error extracting reviews: {e}")
    return reviews

def analyze_review(review, code1, code2):
    role = """
    You are a helpful qualitative
    analysis assistant, aiding researchers in
    developing final codes that can be utilized
    in subsequent stages, including final coding
    processes.
    """

    prompt = f"""
    Please create three concise,
    non-repetitive, and general six-word code
    combinations for the text below using code1
    and code2:

    text is {review}

    code1 is {code1}

    code2 is {code2}

    Requirements:
    1. 6 words or fewer;
    2. No duplicate words;
    3. Be general;
    4. Three distinct versions

    You should answer like this:

    Here is the format of results:
    Version1:
    Version2:
    Version3:
    """

    try:
        response = ollama.chat(model='gemma:7b', messages=[
            {
                'role': 'system',
                'content': role
            },
            {
                'role': 'user',
                'content': prompt
            },
        ])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error in analyze_review: {e}")
        return None

def parse_analysis_result(result):
    versions = {'Version1': '', 'Version2': '', 'Version3': ''}
    if result is None:
        return versions
    lines = result.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('Version'):
            try:
                key, value = line.split(':', 1)
                versions[key.strip()] = value.strip()
            except ValueError:
                logging.error(f"Error parsing line: {line}")
    logging.debug(f"Parsed versions: {versions}")
    return versions

def save_results(results, output_file):
    try:
        # Get the directory of the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create a 'results' folder in the same directory as the script
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create the full path for the output file
        full_path = os.path.join(results_dir, output_file)
        
        # Check if we have write permissions
        if os.access(results_dir, os.W_OK):
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Results successfully saved to {full_path}")
        else:
            logging.error(f"No write permission for {results_dir}")
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise  # Re-raise the exception to stop execution

# Main execution
file_path = r'D:\Internship_SUTD\llm_eval\LLM_Eval\data\llm_data\phase2.json'
reviews = extract_reviews(file_path)

all_results = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'analysis_results_{timestamp}.json'

try:
    for i, review_data in enumerate(reviews, 1):
        print(f"\nAnalyzing Review {review_data['id']}:")
        review = review_data['interview_data']
        print(review[:100] + "..." if len(review) > 100 else review)
        result = analyze_review(review, review_data['code1'], review_data['code2'])
        if result:
            print("\nAnalysis Result:")
            print(result)
            logging.debug(f"Raw result from AI: {result}")
            
            parsed_result = parse_analysis_result(result)
            
            review_result = {
                "id": review_data['id'],
                "review": review,
                "code1": review_data['code1'],
                "code2": review_data['code2'],
                "Version1": parsed_result['Version1'],
                "Version2": parsed_result['Version2'],
                "Version3": parsed_result['Version3']
            }
            logging.debug(f"Review result: {review_result}")
            
            all_results.append(review_result)
            
            # Save results after each analysis
            save_results(all_results, output_file)
            print(f"Results saved to {output_file}")
        else:
            print("Failed to analyze this review. Moving to the next one.")
        print("-" * 50)
        
        if i < len(reviews):
            continue_analysis = input("Press Enter to continue to the next review, or type 'q' to quit: ")
            if continue_analysis.lower() == 'q':
                break

    print(f"\nAll results have been saved to {output_file}")
except Exception as e:
    logging.error(f"An error occurred during execution: {e}")
    print("An error occurred. Please check the log for details.")
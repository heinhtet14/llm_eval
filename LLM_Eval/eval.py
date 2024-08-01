import ijson
import ollama
import json
from datetime import datetime

def extract_reviews(file_path, num_reviews=100):
    reviews = []
    with open(file_path, 'rb') as file:
        parser = ijson.parse(file)
        for prefix, event, value in parser:
            if prefix.endswith('.interview_data'):
                reviews.append(value)
                if len(reviews) == num_reviews:
                    break
    return reviews

def analyze_review(review):
    role = """
    You are a helpful qualitative
    analysis assistant, aiding researchers in
    developing codes that can be utilized in
    subsequent stages, including discussions
    for creating codebooks and final coding
    processes
    """

    prompt = f"""
    Please create three general
    summaries for {review} (within six-word);

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

def parse_analysis_result(result):
    lines = result.strip().split('\n')
    versions = {}
    for line in lines:
        if line.startswith('Version'):
            key, value = line.split(':', 1)
            versions[key.strip()] = value.strip()
    return versions

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# Main execution
file_path = 'D:\Internship_SUTD\llm_eval\LLM_Eval\data\extracted_interview_data.json'  
reviews = extract_reviews(file_path)


all_results = []


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f'analysis_results_{timestamp}.json'

for i, review in enumerate(reviews, 1):
    print(f"\nAnalyzing Review {i}:")
    print(review[:100] + "..." if len(review) > 100 else review)
    result = analyze_review(review)
    print("\nAnalysis Result:")
    print(result)
    
    parsed_result = parse_analysis_result(result)
    
    
    all_results.append({
        "review": review,
        "Version1": parsed_result.get('Version1', ''),
        "Version2": parsed_result.get('Version2', ''),
        "Version3": parsed_result.get('Version3', '')
    })
    
    # Save results after each analysis
    save_results(all_results, output_file)
    print(f"Results saved to {output_file}")
    print("-" * 50)

    
    if i < len(reviews):
        continue_analysis = input("Press Enter to continue to the next review, or type 'q' to quit: ")
        if continue_analysis.lower() == 'q':
            break

print(f"\nAll results have been saved to {output_file}")
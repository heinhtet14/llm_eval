import json
import nltk
from blanc import BlancHelp, BlancTune

nltk.download('punkt')

with open('/mnt/d/Internship_SUTD/llm_eval/LLM_Eval/data/response_llama3.json', 'r') as json_file:
    data = json.load(json_file)

# Extract Version1
version1 = data['versions']['Version1']
print(version1)

reference = """How A Business Works was an excellent book to read as I began my first semester as a college
               student. Although my goal is to major in Business,
               I started my semester off with no idea of even the
               basic guidelines a Business undergrad should know.
               This book describes in detail every aspect dealing
               with business relations, and I enjoyed reading it.
               It felt great going to my additional business classes
               prepared and knowledgeable on the subject they
               were describing. Very well written, Professor
               Haeberle! I recommend this book to anyone and
               everyone who would like additional knowledge
               pertaining to business matters."""

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
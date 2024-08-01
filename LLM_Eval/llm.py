import ollama
import json

role = """
You are a helpful qualitative
analysis assistant, aiding researchers in
developing final codes that can be utilized
in subsequent stages, including final coding
processes.
"""
text = """How A Business Works was an excellent book to read as I began my first semester as a college
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
pertaining to business matters.  """

code1 = "Detailed introduction to business relations"
code2 = "Comprehensive guide to business basics"

prompt = f"""
Please create three concise,
non-repetitive, and general six-word code
combinations for the text below using code1
and code2:

text is {text}

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


response = ollama.chat(model='phi3:medium', messages=[
    {
        'role': 'system',
        'content': role
    },
    {
        'role': 'user',
        'content': prompt
    },
])
print("Response with Phi3")
print(response['message']['content'])
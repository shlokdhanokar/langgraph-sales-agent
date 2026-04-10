import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def test_model(m):
    try:
        res = genai.GenerativeModel(m).generate_content('hi')
        print(f'{m}: SUCCESS (usable!)')
    except Exception as e:
        print(f'{m}: FAILED ({e})')

models_to_test = [
    'gemini-flash-latest',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemma-3-1b-it'
]

for model in models_to_test:
    test_model(model)

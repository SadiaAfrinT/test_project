import os
import google.generativeai as genai

def ask(question: str) -> str:
    """
    This function takes a question as a string and returns a string as an answer.
    """
    # Get your API key from https://aistudio.google.com/app/apikey
    # Store it in an environment variable called GOOGLE_API_KEY
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

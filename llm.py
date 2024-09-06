import requests
import os
from dotenv import load_dotenv
import PyPDF2
import time

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in the environment variables")

# Function to extract text from a PDF using PyPDF2
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
    return text

# Query ChatGPT using OpenAI API with retry logic
def query_chatgpt(prompt, model="gpt-3.5-turbo"):
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    max_retries = 5
    delay = 5  # Start with 5 seconds delay
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            if response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"An error occurred while querying ChatGPT: {e}")
                return "Error: Unable to get a response from ChatGPT."
    return "Error: Max retries exceeded."

# Main function to extract text and query ChatGPT
def main(pdf_path):
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("No text was extracted from the PDF.")
        return

    # Step 2: Start user interaction
    print("PDF content extracted. Type 'q' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'q':
            print("Exiting...")
            break
        try:
            # Combine the query with the extracted text
            prompt = f"Based on the following document: {pdf_text}\n\nAnswer the following question: {query}"
            result = query_chatgpt(prompt)
            print(f"ChatGPT: {result}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    pdf_path = r"knowledgeBase/self.pdf" 
    main(pdf_path)

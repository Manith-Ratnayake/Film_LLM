from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Get the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Please set it in your .env file.")

# Load the PDF
pdf_path = "customData/the_film_business_handbook.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Convert the text to embeddings for querying
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(documents, embeddings)

# Set up the Conversational Retrieval Chain
llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

# Ask questions
def main():
    print("Type 'q' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == 'q':
            print("Exiting...")
            break
        try:
            result = qa_chain.run(query)
            print(result)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

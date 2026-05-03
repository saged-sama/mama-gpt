
"""
This code is designed to implement a Retrieval-Augmented Generation system using Large Language Models.
It uses the Gemini model from Google Generative AI and LangChain for processing and generating responses based on the Nvidia 10-K report.
"""

# Importing necessary libraries
import textwrap
import google as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter, PythonCodeTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain import RetrievalQA

# Constants
GOOGLE_API_KEY = 'your_google_api_key'  # Replace with your actual Google API key
PDF_PATH = "/content/Nvidia10K2024Q3.pdf"
GEMINI_MODEL_NAME = "gemini-pro"
EMBEDDING_MODEL_NAME = "models/embedding-001"
TEMPERATURE = 0.2
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

# Function to convert text to Markdown format
def to_markdown(text, bullet_point='*', blockquote_symbol='> '):
    """
    Converts a given text to Markdown format.
    Args:
    text (str): The text to be converted.
    bullet_point (str, optional): The bullet point symbol for lists. Defaults to '*'.
    blockquote_symbol (str, optional): The symbol for blockquotes. Defaults to '> '.
    Returns:
    Markdown: The text converted to Markdown format.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    # Replace bullet points and handle new lines for blockquotes
    wrapped_text = text.replace('•', f'  {bullet_point}')
    wrapped_text = textwrap.indent(wrapped_text, blockquote_symbol, lambda line: True)

    return Markdown(wrapped_text)

# Function to load and split the PDF document
def load_and_split_pdf(pdf_path):
    """
    Loads a PDF document and splits it into pages.
    Args:
    pdf_path (str): Path to the PDF file.
    Returns:
    list: List of pages from the PDF document.
    """
    pdf_loader = PyPDFLoader(pdf_path)
    return pdf_loader.load_and_split()

# Function to set up the Gemini model
def setup_gemini_model(model_name, api_key, temperature):
    """
    Sets up the Gemini model for text generation.
    Args:
    model_name (str): Name of the Gemini model.
    api_key (str): API key for accessing Google Generative AI.
    temperature (float): Temperature setting for the model.
    Returns:
    ChatGoogleGenerativeAI: Configured Gemini model.
    """
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True)

# Function to create embeddings and vector index
def create_embeddings_and_index(texts, model_name, api_key):
    """
    Creates embeddings for texts and builds a vector index for retrieval.
    Args:
    texts (list): List of texts to create embeddings for.
    model_name (str): Name of the embedding model.
    api_key (str): API key for accessing Google Generative AI.
    Returns:
    Retriever: Vector index for text retrieval.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})
    return vector_index

# Function to create RAG QA Chain
def create_rag_qa_chain(model, vector_index):
    """
    Creates a Retrieval-Augmented Generation QA chain.
    Args:
    model (ChatGoogleGenerativeAI): Configured Gemini model.
    vector_index (Retriever): Vector index for retrieval.
    Returns:
    RetrievalQA: Configured RAG QA Chain.
    """
    return RetrievalQA.from_chain_type(model, retriever=vector_index, return_source_documents=True)

# Main Execution Flow
def main():
    """
    Main function to execute the RAG workflow.
    """
    pages = load_and_split_pdf(PDF_PATH)
    
    # Splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)

    # Setting up the Gemini model and embeddings
    gemini_model = setup_gemini_model(GEMINI_MODEL_NAME, GOOGLE_API_KEY, TEMPERATURE)
    vector_index = create_embeddings_and_index(texts, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY)
    
    # Creating RAG QA Chain
    qa_chain = create_rag_qa_chain(gemini_model, vector_index)

    # Example Usage
    question = "What basis was used for preparing Nvidia's unaudited condensed consolidated financial statements?"
    result = qa_chain({"query": question})
    print("Answer:", result["result"])

if __name__ == "__main__":
    main()
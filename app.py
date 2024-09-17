import streamlit as st
import requests
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
import torch
import re
import transformers

# Function to list all files in a GitHub folder using the GitHub API
def list_files_in_github_folder(repo_owner, repo_name, folder_path):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to list files in the folder: {response.status_code}")
        return []

# Function to download PDF from GitHub
def download_pdf_from_github(github_raw_url, save_as):
    response = requests.get(github_raw_url)
    if response.status_code == 200:
        with open(save_as, 'wb') as f:
            f.write(response.content)
        return save_as
    else:
        st.error("Failed to download the file.")
        return None

# GitHub repository details
repo_owner = "krishna18122001"
repo_name = "chatbot_teams"
folder_path = "pdf"  # Path to the folder in the GitHub repository containing PDF files

# List files in the GitHub folder
files_in_github_folder = list_files_in_github_folder(repo_owner, repo_name, folder_path)

# Download each PDF file from the GitHub folder
downloaded_files = []
for file_info in files_in_github_folder:
    if file_info['name'].endswith(".pdf"):  # Only process PDF files
        file_url = file_info['download_url']
        downloaded_file = download_pdf_from_github(file_url, file_info['name'])
        if downloaded_file:
            downloaded_files.append(downloaded_file)

# Load the downloaded PDFs into LangChain
from langchain_community.document_loaders import PyPDFLoader  # Import PDF Loader for LangChain

documents = []
for file_path in downloaded_files:
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())

# Initialize embeddings and ChromaDB
model_name = "sentence-transformers/all-mpnet-base-v2"
device = "cpu"
model_kwargs = {"device": device}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(documents)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="pdf_db")
books_db = Chroma(persist_directory="./pdf_db", embedding_function=embeddings)

books_db_client = books_db.as_retriever()

# Initialize the model and tokenizer
model_name = "stabilityai/stablelm-zephyr-3b"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(model_name, max_new_tokens=1024)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

query_pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    torch_dtype=torch.float16,
    device_map=device,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=query_pipeline)

books_db_client_retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=books_db_client,
    verbose=True
)

st.title("RAG System with ChromaDB")

# Initialize session state for tracking previous questions and answers
if "history" not in st.session_state:
    st.session_state.history = []

# Function to retrieve answer using the RAG system
def test_rag(qa, query):
    return qa.run(query)

query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        # Get the answer from RAG
        books_retriever = test_rag(books_db_client_retriever, query)

        # Extracting the relevant answer using regex
        corrected_text_match = re.search(r"Helpful Answer:(.*)", books_retriever, re.DOTALL)
        
        if corrected_text_match:
            corrected_text_books = corrected_text_match.group(1).strip()
        else:
            corrected_text_books = "No helpful answer found."

        # Store the query and answer in session state
        st.session_state.history.append({"question": query, "answer": corrected_text_books})

# Display previous questions and answers
if st.session_state.history:
    for idx, item in enumerate(st.session_state.history):
        st.write(f"**Question:** {item['question']}")
        st.write(f"**Answer:** {item['answer']}")
        st.write("---")

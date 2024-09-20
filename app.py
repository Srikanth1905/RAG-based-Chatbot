import streamlit as st
import os
import time
import warnings
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import hashlib
from datetime import datetime
import pandas as pd

load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set page config
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Constants and Environment Variables
DATA_DIR = os.getenv('DATA_PATH')
T5_MODEL_PATH = os.getenv('T5_MODEL_PATH')
SENTENCE_TRANSFORMER_PATH = os.getenv('SENTENCE_TRANSFORMER_PATH')
VECTOR_STORE_PATH = "faiss_index.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize session state variables
if 't5_model' not in st.session_state:
    st.session_state.t5_model = None
if 't5_tokenizer' not in st.session_state:
    st.session_state.t5_tokenizer = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'document_texts' not in st.session_state:
    st.session_state.document_texts = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'query_cache' not in st.session_state:
    st.session_state.query_cache = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'model_loading_time': 0,
        'faiss_index_status': 'Not created',
        'embeddings_status': 'Not created',
        'queries': []
    }

@st.cache_resource
def load_models():
    start_time = time.time()
    try:
        t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
        t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(DEVICE)
        embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)
        end_time = time.time()
        st.session_state.metrics['model_loading_time'] = end_time - start_time
        return t5_tokenizer, t5_model, embedding_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None, None, None

def read_file(file_path: str) -> str:
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                pdf = PdfReader(f)
                return "\n".join(page.extract_text() for page in pdf.pages)
        elif file_path.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
    return ""

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 3000) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def create_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    st.session_state.metrics['embeddings_status'] = 'Creating'
    embeddings = model.encode(texts, show_progress_bar=True)
    st.session_state.metrics['embeddings_status'] = 'Created'
    return embeddings

def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    st.session_state.metrics['faiss_index_status'] = 'Creating'
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, VECTOR_STORE_PATH)
    st.session_state.metrics['faiss_index_status'] = 'Created'
    return index

def load_documents(data_dir: str) -> List[Dict[str, str]]:
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx', '.txt')):
                file_path = os.path.join(root, file)
                content = read_file(file_path)
                if content:
                    documents.append({"content": content, "source": file_path})
    return documents

def create_vector_store():
    documents = load_documents(DATA_DIR)
    if not documents:
        st.error("No documents found in the specified directory.")
        return None, [], []

    with st.spinner("Processing documents..."):
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(doc["content"])
            all_chunks.extend([(chunk, doc["source"]) for chunk in chunks])

        chunk_texts, chunk_sources = zip(*all_chunks)
        embeddings = create_embeddings(chunk_texts, st.session_state.embedding_model)
        faiss_index = create_faiss_index(embeddings)
        faiss.write_index(faiss_index, VECTOR_STORE_PATH)

    return faiss_index, chunk_texts, chunk_sources

def load_existing_index():
    if os.path.exists(VECTOR_STORE_PATH):
        st.session_state.metrics['faiss_index_status'] = 'Loading'
        index = faiss.read_index(VECTOR_STORE_PATH)
        st.session_state.metrics['faiss_index_status'] = 'Loaded'
        return index
    return None

def retrieve_relevant_chunks(query: str, k: int = 5) -> Tuple[List[str], List[float]]:
    if st.session_state.faiss_index is None:
        raise ValueError("FAISS index is not initialized.")

    query_embedding = create_embeddings([query], st.session_state.embedding_model)
    distances, indices = st.session_state.faiss_index.search(query_embedding, k)
    relevant_chunks = [st.session_state.document_texts[i] for i in indices[0] if i < len(st.session_state.document_texts)]
    relevance_scores = [1 / (1 + d) for d in distances[0][:len(relevant_chunks)]]
    return relevant_chunks, relevance_scores

def create_rag_prompt(query: str, relevant_chunks: List[str], chat_history: List[Dict[str, str]]) -> str:
    context = "\n".join(relevant_chunks)
    recent_history = chat_history[-5:]  # Get last 5 exchanges
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    prompt_template = f"""Recent conversation:
{history_str}

User's current question: {query}

Relevant information from documents:
{context}

Based on the conversation history and the information above, provide a detailed and accurate response to the user's current question. Ensure the response is clear, addresses the query comprehensively, and maintains continuity with the conversation history. If the provided information is insufficient, indicate what additional details are needed."""

    return prompt_template

def generate_response(prompt: str) -> str:
    inputs = st.session_state.t5_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    with torch.no_grad():
        output = st.session_state.t5_model.generate(**inputs, max_length=512, num_return_sequences=1, temperature=0.7)
    return st.session_state.t5_tokenizer.decode(output[0], skip_special_tokens=True)

def process_query(user_query: str) -> Tuple[str, List[str]]:
    start_time = time.time()
    
    cache_key = hashlib.md5(user_query.encode()).hexdigest()
    if cache_key in st.session_state.query_cache:
        return st.session_state.query_cache[cache_key]

    try:
        relevant_chunks, _ = retrieve_relevant_chunks(user_query)
        if not relevant_chunks:
            return "I couldn't find any relevant information. Could you please rephrase or ask something else?", []

        rag_prompt = create_rag_prompt(user_query, relevant_chunks, st.session_state.chat_history)
        answer = generate_response(rag_prompt)
        
        result = (answer.strip(), relevant_chunks[:5])
        st.session_state.query_cache[cache_key] = result
        
        end_time = time.time()
        response_time = end_time - start_time
        st.session_state.metrics['queries'].append({
            'query': user_query,
            'response_time': response_time,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I encountered an error while processing your query. Please try again.", []

def summarize_document(content: str) -> str:
    max_length = 5000
    inputs = st.session_state.t5_tokenizer.encode("summarize: " + content[:max_length], return_tensors="pt", max_length=max_length, truncation=True).to(DEVICE)
    summary_ids = st.session_state.t5_model.generate(inputs, max_length=750, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = st.session_state.t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def stream_response(response: str):
    placeholder = st.empty()
    for i in range(len(response.split())):
        placeholder.markdown(" ".join(response.split()[:i+1]))
        time.sleep(0.05)

def display_metrics():
    st.title("Performance Metrics")

    st.metric("Model Loading Time", f"{st.session_state.metrics['model_loading_time']:.2f} s")
    st.metric("FAISS Index Status", st.session_state.metrics['faiss_index_status'])
    st.metric("Embeddings Status", st.session_state.metrics['embeddings_status'])

    if st.session_state.metrics['queries']:
        df = pd.DataFrame(st.session_state.metrics['queries'])
        st.subheader("Query Response Times")
        st.line_chart(df.set_index('timestamp')['response_time'])

        st.subheader("Recent Queries")
        st.dataframe(df[['query', 'response_time', 'timestamp']].tail())

    else:
        st.info("No queries processed yet.")

def main():
    initialize_resources()

    st.title("Chatbot")

    # Sidebar for document upload, summarization, and metrics
    # st.sidebar.subheader("Document Management")
    # uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    # if uploaded_files:
    #     with st.spinner("Processing uploaded files..."):
    #         new_documents = []
    #         for uploaded_file in uploaded_files:
    #             file_path = os.path.join(DATA_DIR, uploaded_file.name)
    #             with open(file_path, "wb") as f:
    #                 f.write(uploaded_file.getbuffer())
                
    #             content = read_file(file_path)
    #             new_documents.append({"content": content, "source": file_path})
                
                # Generate and display summary
                # summary = summarize_document(content)
                # st.sidebar.subheader(f"Summary of {uploaded_file.name}")
                # st.sidebar.write(summary)
            
                # st.session_state.faiss_index, st.session_state.document_texts, _ = create_vector_store()
                # st.success("Vector store updated with new documents.")
                # st.session_state.query_cache.clear()  # Clear cache when new documents are added

    # Add a button to view metrics
    if st.sidebar.button("View Metrics"):
        display_metrics()

    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your question?"):
        st.session_state.chat_history.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response, relevant_chunks = process_query(prompt)
            stream_response(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

def initialize_resources():
    if (st.session_state.t5_model is None or st.session_state.embedding_model is None):
        with st.spinner("Loading models..."):
            (st.session_state.t5_tokenizer,
             st.session_state.t5_model,
             st.session_state.embedding_model) = load_models()

    if st.session_state.faiss_index is None:
        existing_index = load_existing_index()
        if existing_index:
            st.session_state.faiss_index = existing_index
            st.success("Loaded existing FAISS index.")
        else:
            st.session_state.faiss_index, st.session_state.document_texts, _ = create_vector_store()
            st.success("Created new FAISS index.")

if __name__ == "__main__":
    main()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
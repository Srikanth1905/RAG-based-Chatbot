import os
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()
T5_MODEL_PATH = os.getenv('T5_MODEL_PATH')
SENTENCE_TRANSFORMER_PATH = os.getenv('SENTENCE_TRANSFORMER_PATH')
DATA_PATH = os.getenv('N_DATA_PATH')

# Set Streamlit page configuration
st.set_page_config(page_title=" CSE Chatbot", page_icon="💻")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

@st.cache_resource
def load_models():
    # Load Flan-T5 Large model
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH, local_files_only=True)
    text_generation = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens = 1024)
    llm = HuggingFacePipeline(pipeline=text_generation)

    # Load Sentence Transformer embeddings
    embeddings = HuggingFaceEmbeddings(model_name=SENTENCE_TRANSFORMER_PATH, 
                                       model_kwargs={'device': 'cpu'})

    return llm, embeddings

# Load models
try:
    llm, embeddings = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def load_documents():
    loaders = [
        DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'}, use_multithreading=True),
        DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader, use_multithreading=True),
        DirectoryLoader(DATA_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, use_multithreading=True),
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

def create_chain():
    prompt = ChatPromptTemplate.from_template("""
    You are a Computer Science Engineering expert. Answer the following question based on the given context.

    Context: {context}

    Question: {input}

    Answer: Provide a detailed and structured response related to Computer Science Engineering.
    If the question is not related to CSE, politely inform the user and suggest asking about CSE topics.
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_store.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

# Streamlit UI
st.title(" Chatbot 💻")

# Load documents and create vector store if not already done
if st.session_state.vector_store is None:
    with st.spinner("Loading documents and creating vector store..."):
        try:
            st.session_state.vector_store = load_documents()
            st.success("Documents loaded successfully!")
        except Exception as e:
            st.error(f"Error loading documents: {e}")
            st.stop()

# Create the chain
chain = create_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_input = st.chat_input("Ask me ...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"input": user_input})
            answer = response.get('answer', "I'm sorry, I couldn't generate a response.")
        except Exception as e:
            answer = f"An error occurred: {str(e)}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
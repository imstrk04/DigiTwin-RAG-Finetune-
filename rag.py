import streamlit as st
import os
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import numpy as np

# Initialize the Ollama model
llm = OllamaLLM(base_url="http://localhost:11434", model="llama3.1")

# Ensure 'pdfs' directory exists
if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

class CustomEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        """Embed search docs."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        """Embed query text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def __call__(self, text):
        """Make the class callable."""
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or list of strings")

# Initialize session state variables
if "embeddings_model" not in st.session_state:
    # Using SentenceTransformers for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.embeddings_model = CustomEmbeddings(model)

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "loader" not in st.session_state:
    st.session_state.loader = None

if "docs" not in st.session_state:
    st.session_state.docs = None

if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = None

if "final_documents" not in st.session_state:
    st.session_state.final_documents = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def vector_embedding(uploaded_file=None):
    if uploaded_file:
        with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.loader = PyPDFDirectoryLoader("pdfs")
    else:
        st.session_state.loader = PyPDFDirectoryLoader("pdfs")
    
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
    
    # Create FAISS index using the embeddings
    st.session_state.vectors = FAISS.from_documents(
        st.session_state.final_documents,
        st.session_state.embeddings_model
    )

def summarize_document():
    if "final_documents" not in st.session_state or not st.session_state.final_documents:
        return "Please upload a document first."
    
    try:
        full_text = " ".join([doc.page_content for doc in st.session_state.final_documents])
        
        summary_prompt = ChatPromptTemplate.from_template(
            """
            Please provide a concise summary of the following document:
            {text}
            
            Summary:
            """
        )
        
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        summary = summary_chain.run(text=full_text[:4000])
        
        return summary
    except ValueError as e:
        st.error(f"Error during summarization: {str(e)}")
        return "An error occurred while summarizing the document. Please try again."
    except Exception as e:
        st.error(f"Unexpected error during summarization: {str(e)}")
        return "An unexpected error occurred. Please try again or contact support."

st.title("Chat with PDF")

st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file and st.sidebar.button("Upload & Embed"):
    vector_embedding(uploaded_file)
    st.sidebar.success("Document uploaded and embedded!")

if st.sidebar.button("Summarize Document"):
    with st.spinner("Generating summary..."):
        summary = summarize_document()
        st.write("### Document Summary")
        st.write(summary)

prompt1 = st.text_input("Ask a question from the document")

if st.button("Ask"):
    if prompt1:
        if "vectors" not in st.session_state or not st.session_state.vectors:
            vector_embedding()

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question:
            <context>
            {context}
            </context>
            Questions: {input}
            """
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        response = retrieval_chain.invoke({'input': prompt1})
        answer = response['answer']

        st.session_state.chat_history.append({"question": prompt1, "answer": answer})

        st.write("### Answer")
        st.write(answer)

        with st.expander("Chat History"):
            for i, chat in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}:** {chat['question']}")
                st.write(f"**A{i+1}:** {chat['answer']}")
                st.write("-----------------------------------")
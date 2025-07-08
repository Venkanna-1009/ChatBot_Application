import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM

import streamlit as st
# --- Load and chunk data (run once, or load from existing vector store) ---
def load_chunks(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    sections = []
    current_section = ""
    for line in text.splitlines():
        if line.strip().startswith("#"):
            if current_section:
                sections.append(current_section.strip())
                current_section = ""
        current_section += line + "\n"
    if current_section:
        sections.append(current_section.strip())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    for section in sections:
        chunks = text_splitter.split_text(section)
        all_chunks.extend(chunks)
    return all_chunks

# --- Build or load vector store ---
def get_vector_store(persist_directory="chroma_store_1"):
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    if not (os.path.exists(persist_directory) and os.listdir(persist_directory)):
        chunks = load_chunks("extracted_structured_data.txt")
        Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=persist_directory)
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

# --- LLM setup ---
def get_llm():
    return OllamaLLM(model="llama3.2")

def get_prompt_template(context, question):
    return f"""You are an expert assistant. Use the following context to answer the question.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

# --- Streamlit UI ---
st.title("RAG Chatbot (Ollama + Llama 3.2)")
user_query = st.text_input("Ask a question:")

if user_query:
    if st.button("Enter"):
        vectordb = get_vector_store()
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(user_query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = get_prompt_template(context, user_query)
        llm = get_llm()
        answer = llm.invoke(prompt)
        st.markdown(f" $$ Bot: {answer}")

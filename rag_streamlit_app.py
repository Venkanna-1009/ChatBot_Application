import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM


@st.cache_data(show_spinner=False)
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
@st.cache_resource(show_spinner=False)
def get_vector_store(chunks=None, persist_directory="chroma_store_1"):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Vector store already exists at '{persist_directory}', loading existing store.")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model='nomic-embed-text'))
    else:
        print(f"Vector store is not found at '{persist_directory}', creating a new store.")
        embedding_model = OllamaEmbeddings(model='nomic-embed-text')
        vectordb = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=persist_directory)
        vectordb.persist()
    return vectordb

# --- LLM setup ---
@st.cache_resource(show_spinner=False)
def get_llm(temperature=0):
    return OllamaLLM(model="llama3.2",temperature=temperature)

def get_prompt_template(context, question):
    return f"""You are an expert assistant. Use the following context to answer the question.
    Context:
    {context}
    
    Question: {question} 
    Answer:"""

# Streamlit UI 
st.title("RAG Chatbot (Ollama + Llama 3.2)")
user_query = st.text_input("Ask a question:")

if user_query:
    if st.button("Enter"):
        chunks = load_chunks("extracted_structured_data.txt")
        st.write(f"Loaded {len(chunks)} chunks.")
        vectordb = get_vector_store(chunks)
        st.write("Vector store built and persisted.")
        llm = get_llm()
        st.write("llm loaded.")
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(user_query)
        if not docs:
            st.warning("Please ask queries related to jaaji technologies.")
        else:
            st.write(f"Number of retrieved docs: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                st.write(f"\n  Retrieved Chunk {i} ")
                st.write(doc.page_content[:250])
            context = "\n".join([doc.page_content for doc in docs])
            prompt = get_prompt_template(context, user_query)
            answer = llm.invoke(prompt)
            st.markdown(f"$$ Bot: {answer}")

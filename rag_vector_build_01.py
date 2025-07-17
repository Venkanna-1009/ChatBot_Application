import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM


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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = []
    for section in sections:
        chunks = text_splitter.split_text(section)
        all_chunks.extend(chunks)
    return all_chunks


def build_vector_store(chunks, persist_directory="chroma_store"):
    # Check if the vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Vector store already exists at '{persist_directory}', loading existing store.")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
    else:
        print(f"Vector store not found at '{persist_directory}', creating new store.")
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        vectordb = Chroma.from_texts(chunks, embedding=embedding_model, persist_directory=persist_directory)
        vectordb.persist()
    return vectordb

def get_llm(temperature=0):
    return OllamaLLM(model="llama3.2",temperature=temperature)

def get_prompt_template(context, question):
    return f"""You are an expert assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer:"""

if __name__ == "__main__":
    # Load and chunk data
    chunks = load_chunks("extracted_structured_data.txt")
    print(f"Loaded {len(chunks)} chunks.")

    # Only build vector store, do not check for existing one
    vectordb = build_vector_store(chunks)
    print("Vector store built and persisted.")

    # Loading llm
    llm = get_llm()
    print("LLM loaded.")

    # Retrieval + Generation loop
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print("RAG pipeline ready. Ask questions!")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        docs = retriever.invoke(query)

        if not docs:
            print("please ask queries related to jaaji technologies.")
        else:
            print(f"number of retrieved docs: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                print(f"\n  retrieved Chunk {i} ")
                print(doc.page_content[:250])
            context = "\n".join([doc.page_content for doc in docs])
            prompt = get_prompt_template(context, query)
            answer = llm.invoke(prompt)
            print("Bot:", answer)
import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
import faiss
from ctransformers import AutoModelForCausalLM
import asyncio

# Load CSV/JSON and convert to documents
def load_and_process_dataset(file_path, file_type='csv'):
    if file_type == 'csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_json(file_path)

    # Combine row content into a single text string
    text_data = []
    for i, row in df.iterrows():
        row_text = " ".join([str(val) for val in row.values if pd.notnull(val)])
        text_data.append(row_text)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.create_documents(text_data)

    # Embedding setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=uuids)

    return vector_store

# Retrieve relevant documents
def retrieve_docs(vector_store, query):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    source_docs = retriever.invoke(query)

    content = ''
    for doc in source_docs:
        content += doc.page_content + "\n"
    return content

# Generate a response using local model
def generate_response(content, question):
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=0
    )

    prompt = f"""
    Answer the user question based on the context given and not prior knowledge.
    ------------------
    context: {content}
    ------------------
    question: {question}
    """

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = model(prompt, max_new_tokens=200)
    return response

# Streamlit UI
def main():
    st.title("RAG Chatbot (CSV/JSON Dataset Based)")

    file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

    if file:
        file_type = file.name.split(".")[-1]
        with open(f"uploaded_data.{file_type}", "wb") as f:
            f.write(file.getbuffer())

        st.success("Dataset uploaded successfully.")

        # Process dataset
        vector_store = load_and_process_dataset(f"uploaded_data.{file_type}", file_type)

        st.info("Data indexed. You can now ask questions about your dataset.")

        question = st.text_input("Enter your question:")

        if question:
            st.info("Generating answer...")

            content = retrieve_docs(vector_store, question)
            response = generate_response(content, question)

            st.subheader("Answer:")
            st.write(response)

if __name__ == "__main__":
    main()

# RAG Chatbot using LangChain & Mistral

This project is a Retrieval-Augmented Generation (RAG) Chatbot built using Python, LangChain, FAISS, and the Mistral 7B language model. The chatbot is capable of answering user questions based on a custom dataset provided in `.csv` format.

---

## Objective

The goal of this assignment is to:
- Load a custom dataset.
- Set up a RAG pipeline using LangChain.
- Build a chatbot interface using Streamlit.
- Retrieve accurate and context-aware answers based on the dataset.

## Steps I Followed

1.Dataset Loading
   - Chose a CSV file as the external knowledge base.
   - Loaded it using pandas and prepared it for embedding.

2. Text Processing & Embedding
   - Split dataset text into chunks using RecursiveCharacterTextSplitter.
   - Converted chunks into vector embeddings using HuggingFace model: all-MiniLM-L6-v2.

3.Vector Storage & Retrieval
   - Created a vector store using FAISS.
   - Configured a retriever to return the top-k relevant chunks for any query.

4. LLM Integration
   - Used the ctransformers library to load Mistral-7B Instruct model.
   - Provided context + question as a prompt and generated the answer.

5.Chatbot Interface
   - Built using Streamlit with support for:
     - Dataset upload
     - Question input
     - Display of answer in real time

6. Sample Q&A Output
   - Collected chatbot responses for 7 example questions.
   - Saved in sample_responses.xlsx as required.

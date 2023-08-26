from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import requests
import re
import PyPDF2
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Your code to create 'embeddings', 'db', and other necessary components
def extract_text_from_pdf(uploaded_pdf):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_pdf)
    num_pages = len(pdf_reader.pages)

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [token for token in tokens if len(token) > 1 and not token.isdigit()]
    return " ".join(cleaned_tokens)
def split_docs(documents,chunk_size=400,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
# Streamlit app
def main():
    st.title("Document Based Question Answer Using Model GPT3")
    
    uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded:
        pdf_text = extract_text_from_pdf(uploaded)
        cleaned_text = clean_text(pdf_text)
        output_filename = "document.txt"
        with open(output_filename, "w") as output_file:
            output_file.write(cleaned_text)
        
        st.text_area("Processed Text", value=cleaned_text, height=300)
        
        st.success("PDF processed and text saved.")
    
    query = st.text_input("Enter your query:")
    
    if st.button("Search and Generate"):
        # Load the documents
        documents = TextLoader(r"C:\Users\AMAN ARYAN\OneDrive\Desktop\doc_based\document.txt").load()  # Update with your actual implementation
        
        # split the documents in smaller chunks for better searching and reduce the token size
        document_chunk = split_docs(documents)  # Update with your actual implementation
        
        # importing embedding model to create embeddings of chunks
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")  # Update with your actual implementation
        
        # chroma is an open source vector database to store vector index
        db = Chroma.from_documents(document_chunk, embeddings)  # Update with your actual implementation
        url = "https://tg-api-zehr.onrender.com/api/v1/llmodels/gpt"
        embedding_vector = embeddings.embed_query(query)
        docs = db.similarity_search_by_vector(embedding_vector)
        prompt = f"{query} give answer on the basis of Docs {docs}"
        st.subheader("Generated docs:")
        st.write(docs)
        payload = {"prompt": prompt}

        response = requests.post(url, json=payload)
        response_data = response.json()
        content = response_data.get('content', '')
        
        st.subheader("Generated Content:")
        st.write(content)

if __name__ == "__main__":
    main()

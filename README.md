# doc_based
This code appears to be a Python script for a Streamlit web application that performs document-based question-answering (QA) using the GPT-3 model. Let's break down its main components and functionality:

1. **Importing Libraries:**
   - The script imports various libraries and modules required for different tasks, including document processing, text cleaning, PDF extraction, and interaction with the GPT-3 model. These libraries include langchain, Streamlit, requests, re (regular expressions), PyPDF2 (for PDF processing), nltk (Natural Language Toolkit), and more.

2. **PDF Text Extraction:**
   - The `extract_text_from_pdf` function is defined to extract text content from a PDF file uploaded by the user. It uses the PyPDF2 library to read the PDF and extract text from its pages.

3. **Text Cleaning:**
   - The `clean_text` function is defined to clean the extracted text. It removes non-alphanumeric characters, tokenizes the text, and filters out single-character tokens and digits.

4. **Document Splitting:**
   - The `split_docs` function splits the cleaned text into smaller chunks for better searching and to reduce token size. It uses the `RecursiveCharacterTextSplitter` from the langchain library.

5. **Streamlit App:**
   - The `main` function sets up a Streamlit web application.
   - Users can upload a PDF file using the "Upload a PDF file" button. The uploaded PDF is processed to extract and clean its text content.
   - The processed text is displayed in a text area widget.
   - Users can enter a query/question in the "Enter your query" text input field.
   - When the "Search and Generate" button is clicked, the following steps are performed:
     - The documents are loaded using the `TextLoader` from langchain (the actual implementation is expected to be provided).
     - The documents are split into smaller chunks for better searching.
     - Sentence embeddings of the query and document chunks are generated using the SentenceTransformer model.
     - A vector database (Chroma) is used to store and index the document vectors.
     - Similarity search is performed to find documents that are relevant to the query.
     - A prompt is generated using the query and the selected documents.
     - The prompt is sent to an external API (`https://tg-api-zehr.onrender.com/api/v1/llmodels/gpt`) that presumably uses the GPT-3 model to generate an answer based on the provided prompt.
     - The generated content (answer) is displayed in the Streamlit app.

6. **Running the Streamlit App:**
   - The script includes the conditional statement `if __name__ == "__main__":` to run the `main` function when the script is executed.

Please note that some parts of the code are marked with comments like "Update with your actual implementation." These are placeholders where you should replace them with the actual code or implementation details specific to your project or use case.

This script essentially creates a user interface for users to upload a PDF, enter a query, and get answers based on the documents and the GPT-3 model. It leverages various libraries and services for document processing, embeddings, and text generation.

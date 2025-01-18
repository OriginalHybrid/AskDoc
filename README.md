
# Project AskDoc: Interactive PDF Query Tool leveraging RAGs using Langchain

AskDoc is a Streamlit-based application designed to enable users to upload PDF files and interactively query their content. It leverages creating a RAG using Langchain with local LLM 'Llama-3.2-1B-Instruct' and FAISS vector storage to provide accurate answers to user queries, making document analysis intuitive and efficient.

## Features

- **PDF Upload and Parsing**: Upload PDF files and extract their textual content for processing.
- **Text Chunking**: Split the extracted text into manageable chunks for better processing and embedding generation.
- **Hugging Face Embeddings**: Utilize Hugging Face `sentence-transformers` for embedding generation.
- **Vector Search with FAISS**: Perform similarity searches on embedded chunks to retrieve relevant sections of the document.
- **Local Hugging Face Model Integration**: Load and use locally hosted Hugging Face models for query answering.
- **Interactive Q&A**: Input a question, and the app retrieves and generates a detailed answer from the PDF content.
- **Chat History**: Maintains a complete chat history of user queries and generated responses.
- **Reference Questions**: Extract additional reference questions and answers from the query results.
- **Streamlit Interface**: A user-friendly web-based interface for seamless interaction.

## How It Works

1. **PDF Upload**:
   - Users upload their desired PDF document.
   - The text content is extracted using `PyPDF2`.

2. **Text Processing**:
   - The text is split into smaller chunks using the `CharacterTextSplitter` from LangChain for embedding generation.

3. **Embedding and Search**:
   - Embeddings are generated using Hugging Face models (`sentence-transformers/all-MiniLM-L6-v2`).
   - FAISS is used for efficient similarity search on these embeddings.

4. **Model Query**:
   - User queries are processed through a locally hosted Hugging Face model (e.g., `meta-llama/Llama-3.2-1B-Instruct`).
   - The model generates answers by utilizing relevant text chunks retrieved from the vector search.

5. **Output**:
   - The app displays the main question and its answer.
   - Provides reference questions and their answers, if available.
   - Maintains and displays the full chat history for easy reference.

## Authentication

To use Hugging Face models, an authentication token (`HF_TOKEN`) is required. The token should be stored in an `.env` file in the following format:

## Screenshots

### 1. Application Home
![Application Home](images/1%20(1).png)

![PDF Upload](images/1%20(2).png)

### 2. Query Input
![Query Input](images/1%20(3).png)

![Query Input 2](images/1%20(4).png)

### 5. Reference Questions
![Reference Questions](images/1%20(5).png)

![Reference Questions 2](images/1%20(6).png)


## Instructions to Run

```
pip install -r requirements.txt
```

Add Huggingface Token to the .env file

```
streamlit run chat.py
```

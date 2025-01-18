from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
import os

from dotenv import load_dotenv
load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from huggingface_hub import login


# Load environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Setup authentication
def setup_authentication():
    """Setup Hugging Face authentication"""
    try:
        login(token=HF_TOKEN)
        logger.info("Successfully authenticated with Hugging Face")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {str(e)}")
        raise

# Global variable to store chat history
CHAT_HISTORY = []

def update_chat_history(response):
    """
    Updates the global chat history with a new chat response.

    Args:
        response (dict): A dictionary containing 'Question' and 'Helpful Answer'.
    """
    global CHAT_HISTORY  # Declare the global variable
    if "Question" in response and "Helpful Answer" in response:
        # Append the new chat response as a dictionary to the global chat history
        CHAT_HISTORY.append({
            "Question": response["Question"],
            "Helpful Answer": response["Helpful Answer"]
        })
    else:
        print("Invalid response format. Must contain 'Question' and 'Helpful Answer'.")

def extract_question_and_answer(text):
    """
    Extracts the 'Question' and 'Helpful Answer' from the given string.
    The 'Helpful Answer' will include all lines until the next 'Question:' substring appears.

    Args:
        text (str): The input string containing the question and answer.

    Returns:
        dict: A dictionary with keys 'Question' and 'Helpful Answer'.
    """
    # Split the text into lines
    lines = text.split("\n")

    # Initialize the result dictionary
    result = {"Question": None, "Helpful Answer": ""}
    is_collecting_answer = False  # Flag to track if we are collecting the answer

    # Iterate through the lines to find the question and answer
    for line in lines:
        if line.startswith("Question:"):
            # If a new question is found, stop collecting the previous answer
            if result["Question"] is not None:
                break  # Stop processing as we only want the first question-answer pair
            result["Question"] = line.replace("Question:", "").strip()
            is_collecting_answer = True  # Start collecting the answer
        elif line.startswith("Helpful Answer:"):
            # Start collecting the answer from this line
            result["Helpful Answer"] = line.replace("Helpful Answer:", "").strip()
        elif is_collecting_answer:
            # Append lines to the answer until the next "Question:" appears
            result["Helpful Answer"] += " " + line.strip()

    # Clean up any extra whitespace in the answer
    result["Helpful Answer"] = result["Helpful Answer"].strip()

    return result

def get_reference_qna(response, query_result):
    """
    Extracts reference questions and answers from the response text, excluding the main question-answer pair.

    Args:
        response (str): The input response text containing the main question-answer pair
                       and optionally additional reference questions and answers.
        query_result (dict): A dictionary containing the main 'Question' and 'Helpful Answer'.

    Returns:
        list: A list of dictionaries, where each dictionary contains a 'Question' and 'Helpful Answer',
              excluding the main question-answer pair. Returns an empty list if no reference questions are found.
    """
    # Split the response into lines
    lines = response.split("\n")

    # Initialize variables
    reference_questions = []
    current_question = None
    current_answer = None
    is_reference_section = False  # Flag to track if we are in the reference section

    # Iterate through the lines to extract reference questions and answers
    for line in lines:
        if line.startswith("Question:"):
            # If we encounter a new question, save the previous question-answer pair (if any)
            if current_question and current_answer:
                # Check if the current question-answer pair matches the main query_result
                if current_question != query_result["Question"] or current_answer != query_result["Helpful Answer"]:
                    reference_questions.append({
                        "Question": current_question,
                        "Helpful Answer": current_answer
                    })
                current_question = None
                current_answer = None

            # Set the current question and mark as reference section
            current_question = line.replace("Question:", "").strip()
            is_reference_section = True  # Reference questions start after the first main question-answer pair
        elif line.startswith("Helpful Answer:") and is_reference_section:
            # Set the current answer
            current_answer = line.replace("Helpful Answer:", "").strip()
        elif is_reference_section and current_answer:
            # Append additional lines to the current answer if it spans multiple lines
            current_answer += " " + line.strip()

    # Add the last question-answer pair if it exists and doesn't match the main query_result
    if current_question and current_answer:
        if current_question != query_result["Question"] or current_answer != query_result["Helpful Answer"]:
            reference_questions.append({
                "Question": current_question,
                "Helpful Answer": current_answer
            })

    return reference_questions

import streamlit as st

def display_reference_question(reference_question_list):
    """
    Displays the list of reference questions and answers on the Streamlit app.

    Args:
        reference_question_list (list): A list of dictionaries, where each dictionary contains
                                        'Question' and 'Helpful Answer'.
    """
    if not reference_question_list:
        st.write("No reference questions available.")
    else:
        st.write("### Reference Questions and Answers:")
        for idx, reference in enumerate(reference_question_list, start=1):
            st.write(f"**Reference {idx}:**")
            st.write("Question: ", reference['Question'])
            st.write("Answer:   ")
            st.markdown(reference['Helpful Answer'])
            st.write("---")  # Add a separator for better readability

def update_chat_history(query_result):
    """
    Updates the chat history stored in Streamlit's session state.

    Args:
        query_result (dict): A dictionary containing 'Question' and 'Helpful Answer'.
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history if not present

    if "Question" in query_result and "Helpful Answer" in query_result:
        # Append the new query result to the chat history
        st.session_state.chat_history.append(query_result)
    else:
        print("Invalid query result format. Must contain 'Question' and 'Helpful Answer'.")

def get_embeddings(chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage

# Load local Hugging Face model
def load_local_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    # max_length = model.config.n_positions  # Get max sequence length
    max_length = model.config.max_position_embeddings
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=max_length, truncation=True, device=0)
    return HuggingFacePipeline(pipeline=hf_pipeline)


def get_pdf_content(pdf):
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks


def main():
    # load_dotenv()
    st.set_page_config(page_title="AskDoc: Ask your PDF")
    st.header("AskDoc")
    global CHAT_HISTORY
    print("Initial Global Chat History", CHAT_HISTORY)
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf:
    
        pdf_text = get_pdf_content(pdf)

        # Get Pdf Chunks    
        chunks =get_chunks(pdf_text)
        
        # create embeddings
        knowledge_base = get_embeddings(chunks)
        # knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
        
            # llm = OpenAI()
            model_name = "meta-llama/Llama-3.2-1B-Instruct"
            # Initialize the local model
            llm = load_local_model(model_name=model_name)

            ##############################################################
            chain = load_qa_chain(llm, chain_type="stuff")
            # print("Chain coutput from load qa...", chain)
            # print("Type coutput from load qa...", type(chain))

            response = chain.run(input_documents=docs, question=user_question)
            print("response from load_qa_chain...\n", response)
            # print("response type", type(response))
            query_result = extract_question_and_answer(response)
            update_chat_history(query_result)
            st.write("Message from load_qa_chain...")
            st.write("Question: ", query_result['Question'])
            st.write("Answer:   ")
            st.markdown(query_result['Helpful Answer'])

            # Get Sample Reference questions from response if available 
            reference_qna_list = get_reference_qna(response, query_result)

            if len(reference_qna_list) != 0:
                display_reference_question(reference_qna_list)

            # Display the full chat history
            st.write("Message History: \n", st.session_state.chat_history)

            # Debugging: Print the chat history
            print("st.session_state.chat_history", str(st.session_state.chat_history))
            print("TYPE: st.session_state.chat_history", type(st.session_state.chat_history))
    

if __name__ == '__main__':
    if setup_authentication():
        # Authentication succeeded, proceed with tasks
        logger.info("Authentication Successful...\nProceeding with Hugging Face operations...")
        main()
    else:
        # Authentication failed, handle the error
        st.set_page_config(page_title="AskDoc: Ask your PDF")
        st.header("AskDoc")
        logger.warning("Authentication failed. Please check your HF_TOKEN.")
        st.warning("Unable to authenticate with Hugging Face. Please provide a valid token.")
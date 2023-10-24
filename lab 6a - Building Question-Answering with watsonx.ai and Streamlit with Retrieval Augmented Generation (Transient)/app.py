import logging
import os
import pickle
import tempfile

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PIL import Image

from langChainInterface import LangChainInterface

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

handler = StdOutCallbackHandler()

api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)

if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }

GEN_API_KEY = os.getenv("GENAI_KEY", None)

# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=300)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=2)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)

@st.cache_data
def read_pdf(uploaded_files,chunk_size =250,chunk_overlap=20):
    for uploaded_file in uploaded_files:
      bytes_data = uploaded_file.read()
      with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
      # Write content to the temporary file
          temp_file.write(bytes_data)
          filepath = temp_file.name
          with st.spinner('Waiting for the file to upload'):
             loader = PyPDFLoader(filepath)
             data = loader.load()
             text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunk_size, chunk_overlap=chunk_overlap)
             docs = text_splitter.split_documents(data)
             return docs

@st.cache_data
def read_push_embeddings():
    embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("db.pickle"):
        with open("db.pickle",'rb') as file_name:
            db = pickle.load(file_name)
    else:     
        db = FAISS.from_documents(docs, embeddings)
        with open('db.pickle','wb') as file_name  :
             pickle.dump(db,file_name)
        st.write("\n")
    return db

# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):
    docs = read_pdf(uploaded_files)
    db = read_push_embeddings()
    docs = db.similarity_search(user_question)
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.0,
        # GenParams.TOP_K: 100,
        # GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1
    }
    model_llm = LangChainInterface(model=ModelTypes.LLAMA_2_70B_CHAT.value, credentials=creds, params=params, project_id=project_id)
    chain = load_qa_chain(model_llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=user_question)

    st.text_area(label="Model Response", value=response, height=100)
    st.write()

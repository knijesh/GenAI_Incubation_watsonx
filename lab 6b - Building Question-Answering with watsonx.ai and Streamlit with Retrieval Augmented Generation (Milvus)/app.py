import logging
import os
import pickle
import tempfile

import streamlit as st
from dotenv import load_dotenv
#from genai.schemas import GenerateParams
from ibm_watson_machine_learning.metanames import \
    GenTextParamsMetaNames as GenParams
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from pymilvus import (
    connections,
    Collection
)
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
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

@st.cache_data
def milvussearch(query):
    model_name = 'all-MiniLM-L6-v2'

    collection_name = "travel_leave"
    connections.connect("default", host="128.168.140.66", port="19530")
    # print(fmt.format("Start loading"))
    collection = Collection(collection_name)
    collection.load()

    print(collection)
    search_params = {
        "metric_type": "L2",
        "params": {"ef": 10},
    }

    #query = "How do I take sick leave?"

    model = SentenceTransformer(model_name)
    vectors_to_search = model.encode([query]).tolist()

    result = collection.search(vectors_to_search, "vector", search_params,
                                limit=3,
                                output_fields=["text", "vector"],
                                )

    hits = result[0]
    def text2doc(t):
        return Document(page_content = t)

    docs = [text2doc(h.entity.get('text')) for h in hits]
    return docs


# show user input
if user_question := st.text_input(
    "Ask a question about your Document:"
):
    #docs = read_pdf(uploaded_files)
    #db = read_push_embeddings()
    docs = milvussearch(user_question)
    print ("PRINTING DOCS")
    print (docs)
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.0,
        # GenParams.TOP_K: 100,
        # GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1
    }
    model_llm = LangChainInterface(model='ibm/granite-13b-instruct-v1', credentials=creds, params=params, project_id=project_id)
    chain = load_qa_chain(model_llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=user_question)

    st.text_area(label="Model Response", value=response, height=100)
    st.write()

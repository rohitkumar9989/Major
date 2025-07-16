from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
#Embedders
embed=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
#Doc_load and vector_stores
docs=CSVLoader("Train_data.csv")
docs=docs.load()
ret=FAISS.from_documents(docs, embed)

ret.save_local("Abhi_index")
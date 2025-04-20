#
# 1. Loader - TextLoader
# 2. Splitter - CharacterTextSplitter
# 3. Embedding - OpenAIEmbeddings
# 4. Vector Store - PineCOneVectorStore

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == '__main__':
    print("Hello, World!")

    # Load the text file using TextLoader
    print("loading...")
    loader = TextLoader("C:/Users/ramki/Documents/Ram/Technical/My_Learnings/Git_Repos/ice_breaker-vector_dbs/mediumblog1.txt",  encoding = 'UTF-8')
    documents = loader.load()

    # Split the documents into smaller chunks using CharacterTextSplitter
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)

    # Create embeddings using OpenAIEmbeddings
    print("embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    # Intialize Pinecone Vector Store
    print("ingesting...")
    PineconeVectorStore.from_documents(split_docs, embeddings, index_name = os.environ.get("INDEX_NAME"))
    print("Finish...")





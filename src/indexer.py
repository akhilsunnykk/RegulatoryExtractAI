import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def index_regulatory_files(directory_path, persist_directory="./chroma_db"):
    print("Loading documents...")
    docs = []
    for file in os.listdir(directory_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory_path, file))
            docs.extend(loader.load())

    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(docs)

    print("Embedding and indexing into ChromaDB...")
    # Swapped to Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"Successfully indexed {len(splits)} chunks.")
    return vectorstore

if __name__ == "__main__":
    index_regulatory_files("../data/regulatory_docs")
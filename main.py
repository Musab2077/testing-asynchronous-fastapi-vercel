from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

@app.get('/')
async def testing():
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings=HuggingFaceEmbeddings(model_name= embedding_model_name)
    return "hello"
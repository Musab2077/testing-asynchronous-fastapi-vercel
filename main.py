from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(title="Hugging Face Embeddings API")

@app.on_event("startup")
async def load_model():
    global embeddings
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings=HuggingFaceEmbeddings(model_name= embedding_model_name)

@app.get('/testing')
async def testing():
    try:
        return "hello"
    except Exception as e:
        return e
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests
import os

load_dotenv()

app = FastAPI(title="Hugging Face Embeddings API")

embeddings=HuggingFaceEmbeddings(model_name= "sentence-transformers/paraphrase-MiniLM-L3-v2")
# embeddings = HuggingFaceInferenceAPIEmbed 

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

index_name = "midster-bot"

documents = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = documents.as_retriever(search_type = 'similarity', search_kwargs = {'k':3})
llm = ChatGroq(model_name = 'llama-3.3-70b-versatile',groq_api_key = GROQ_API_KEY)
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. keep the "
    "answer detailed."
    "Give the answers in markdown format"
    "\n\n"
    "{context}"
)

system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, system_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.get('/test/{text}')
async def test_embedding(text: str):
    response = rag_chain.invoke({"input" : text})
    return response
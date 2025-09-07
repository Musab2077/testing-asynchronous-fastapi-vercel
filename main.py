from fastapi import FastAPI
import asyncio

app = FastAPI()

@app.get('/')
async def testing():
    await asyncio.sleep(1)
    return "hello"
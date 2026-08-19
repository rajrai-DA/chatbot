from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat, evaluate, health

app = FastAPI(title="Wells Fargo RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    # Vite picks the next free port (5173, 5174, ...) if the default is taken by
    # another project, so match any localhost port rather than hardcoding one.
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(evaluate.router)

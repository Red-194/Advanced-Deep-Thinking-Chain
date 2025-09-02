from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import torch, gc
from engines import AdvancedDeepThinkingChain, EngineWrapper
from models import ModelInit

app = FastAPI()

models = {
    "1": ("microsoft/phi-2", "2.7B parameters - Balanced"),
    "2": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B parameters - Fast"),
    "3": ("google/gemma-2b-it", "2B parameters - Good quality"),
    "4": ("Qwen/Qwen1.5-0.5B-Chat", "0.5B parameters - Very fast"),
    "5": ("custom", "Enter your own model")
}

@app.get('/')
async def main():
    return RedirectResponse(url='http://127.0.0.1:8000/docs')

engine = EngineWrapper()

@app.post('/initialize-model')
async def initialize_model(init: ModelInit):
    pass
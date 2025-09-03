from typing import List, Dict
from fastapi import FastAPI, HTTPException
import torch, gc
from engines import AdvancedDeepThinkingChain, Config
from models import ModelInit, ThoughtProcess, ThinkRequest

app = FastAPI()

engine: Config | None = None
thinker: AdvancedDeepThinkingChain | None = None
history: List[Dict] = []

models = {
    "1": ("microsoft/phi-2", "2.7B parameters - Balanced"),
    "2": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B parameters - Fast"),
    "3": ("google/gemma-2b-it", "2B parameters - Good quality"),
    "4": ("Qwen/Qwen1.5-0.5B-Chat", "0.5B parameters - Very fast"),
    "5": ("custom", "Enter your own model")
}

@app.get('/')
async def main():
    return {'message' : 'Success'}

@app.get('/models')
async def get_models():
    return models

@app.post("/initialize")
async def initialize_model(request: ModelInit):
    global thinker, config

    model_info = models.get(str(request.model_number), models["1"])
    model_name = request.custom_model.strip() if model_info[0] == "custom" else model_info[0]

    await clear_model()

    config = Config(
        model_name=model_name,
        temperature=request.temperature,
        streaming=request.streaming
    )
    thinker = AdvancedDeepThinkingChain(config)

    return {
        "status": "initialized",
        "model_name": model_name,
        "description": model_info[1],
        "config": config.config
    }
    
@app.post("/clear")
async def clear_model():
    global engine, thinker

    if engine is None and thinker is None:
        return {"status": "nothing to clear"}

    if thinker is not None:
        del thinker
        thinker = None

    if engine is not None:
        engine.unload_model()
        engine = None

    gc.collect()
    torch.cuda.empty_cache()

    return {"status": "cleared"}

@app.post("/think/deep", response_model=ThoughtProcess)
async def think_deep(request: ThinkRequest):
    if thinker is None:
        raise HTTPException(status_code=400, detail="Model not initialized")
    return await thinker.think_deeply(request.prompt)

    
@app.post("/think/quick")
async def think_quick(request: ThinkRequest):
    global thinker, history

    if thinker is None:
        raise HTTPException(status_code=400, detail="Model not initialized")

    try:
        response = await thinker.quick_think(request.prompt)
        history.append({
            "mode": "quick",
            "prompt": request.prompt,
            "response": response
        })
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/history")
async def get_history():
    return {"history": history}

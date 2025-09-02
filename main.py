from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import torch, gc
from engines import AdvancedDeepThinkingChain, EngineWrapper
from models import ModelInit

app = FastAPI()

engine: EngineWrapper | None = None
thinker: AdvancedDeepThinkingChain | None = None

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

engine = EngineWrapper()

@app.post("/initialize-model")
async def initialize_model(request: ModelInit):
    global engine, thinker
    
    await clear_model()
    
    model_info = models.get(str(request.model_number), models["1"])
    if model_info[0] == "custom":
        if not request.custom_model or not request.custom_model.strip():
            raise HTTPException(status_code=400, detail="custom_model is required for model_number 5")
        model_name = request.custom_model.strip()
    else:
        model_name = model_info[0]

    if engine is None:
        engine = EngineWrapper()

    engine.config.update({
        "model_name": model_name,
        "temperature": request.temperature,
        "streaming": request.streaming
    })

    engine.load_model(model_name)
    thinker = AdvancedDeepThinkingChain(engine)

    return {
        "status": "initialized",
        "model_name": model_name,
        "description": model_info[1],
        "config": engine.config
    }
    
@app.post("/clear-model")
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

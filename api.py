from fastapi import FastAPI, HTTPException
import uvicorn
from model import inference as model_inference, mock_inference
from model import ModelError
import asyncio
import os
from typing import List
PORT = os.getenv("PORT", 8080)
RELOAD = os.getenv("RELOAD", True)
app = FastAPI()

@app.get("/", status_code=200, tags=["root"])
async def root():
    message = "".join(["Welcome to the inference API.\n", 
                       "Please visit /docs for the API documentation"])
    return {"message": message}

@app.post("/inference", status_code=200,
#                        response_model=List,
                        tags=["inference"],
                        summary="Inference",
                        description=model_inference.__doc__)
async def inference(csv_path: str = None):
    try:        
        labels = await asyncio.to_thread(model_inference, csv_path)
        return {"prediction": labels}
    except ModelError as e:
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

@app.get("/mock_inference", status_code=200,
#                        response_model=List,
                        tags=["mock_inference"],
                        summary="Mock Inference",
                        description=mock_inference.__doc__)
async def mock_inference_route():
    try:        
        labels = await asyncio.to_thread(mock_inference)
        return {"prediction": labels}
    except ModelError as e:
        raise HTTPException(status_code=e.code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=PORT, reload=RELOAD)
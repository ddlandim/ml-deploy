from fastapi import FastAPI, File, UploadFile
import uvicorn
from model import inference as model_inference
import asyncio

app = FastAPI()

@app.post("/inference")
async def inference(csv_path: str = None):
    try:        
        labels = await asyncio.to_thread(model_inference, csv_path)
        return {"prediction": labels}
    except Exception as e:
        return {"error": f"Inference error: {e}"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
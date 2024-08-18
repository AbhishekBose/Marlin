from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import time

app = FastAPI()

# Load the model once, when the app starts
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    query: str

@app.post("/embeddings/")
async def create_embeddings(request: EmbeddingRequest):
    try:
        # Generate embeddings
        start_time = time.time()
            embeddings = model.encode([request.query])
        print(f"Time taken for encoding: {round((time.time()-start_time)*1000,2)} ms")
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health_check")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

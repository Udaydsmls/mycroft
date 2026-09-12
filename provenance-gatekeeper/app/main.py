from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions

app = FastAPI(title="Provenance Gatekeeper Webhook")

# Initialize ChromaDB connection (must match ingest.py configuration)
client = chromadb.PersistentClient(path="./chroma_db")
huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Define the expected JSON payload schema from n8n
class VerificationRequest(BaseModel):
    claim_id: str
    generated_claim: str

@app.post("/verify")
async def verify_claim(payload: VerificationRequest):
    try:
        # 1. Connect to the populated collection
        collection = client.get_collection(
            name="financial_source_docs",
            embedding_function=huggingface_ef
        )
        
        # 2. Perform a semantic similarity search against the generated claim
        results = collection.query(
            query_texts=[payload.generated_claim],
            n_results=1
        )
        
        # 3. Extract the highest matching context
        if results["documents"] and results["documents"][0]:
            retrieved_context = results["documents"][0][0]
        else:
            retrieved_context = "No relevant source context found in the database."
        
        # 4. Return the payload containing both the claim and the retrieved ground-truth
        return {
            "status": "success",
            "claim_id": payload.claim_id,
            "original_claim": payload.generated_claim,
            "retrieved_context": retrieved_context,
            "action": "ready_for_evaluation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
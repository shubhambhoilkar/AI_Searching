import os
import json
import math
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI

# ENV & CONFIG

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

if not all([OPENAI_API_KEY, MONGO_URI, MONGO_DB, MONGO_COLLECTION]):
    raise RuntimeError("Missing required environment variables")

# CLIENT INITIALIZATION

openai_client = OpenAI(api_key=OPENAI_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# FASTAPI APP

app = FastAPI(
    title="AI Tender Search (Vector Similarity)",
    version="2.0.0"
)

# Pydantic MODELS

class SearchRequest(BaseModel):
    query: str


# EMBEDDING & SIMILARITY UTILS

def generate_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# API ENDPOINT (VECTOR SEARCH)

PROJECTION = {
    "_id": 0,
    "bid_number": 1,
    "region": 1,
    "state": 1,
    "start_date": 1,
    "end_date": 1,
    "sentence_frame": 1,
    "embedding": 1
}

SIMILARITY_THRESHOLD = 0.60  # 60%

@app.post("/ai-search")
def ai_search(request: SearchRequest):
    """
    Semantic tender search using vector similarity
    """

    # 1. Generate embedding for user query
    user_embedding = generate_embedding(request.query)

    # 2. Fetch candidate tenders (only those with embeddings)
    candidates = list(
        collection.find(
            {"embedding": {"$exists": True}},
            PROJECTION
        ).limit(200)
    )

    if not candidates:
        return {
            "user_query": request.query,
            "matched_results": 0,
            "results": []
        }

    # 3. Compute similarity
    matched_results = []

    for doc in candidates:
        tender_embedding = doc.get("embedding")
        if not tender_embedding:
            continue

        similarity = cosine_similarity(user_embedding, tender_embedding)

        if similarity >= SIMILARITY_THRESHOLD:
            matched_results.append({
                "bid_number": doc.get("bid_number"),
                "region": doc.get("region"),
                "state": doc.get("state"),
                "start_date": doc.get("start_date"),
                "end_date": doc.get("end_date"),
                "similarity_score": round(similarity * 100, 2)
            })

    # 4. Sort by similarity (descending)
    matched_results.sort(
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    return {
        "user_query": request.query,
        "similarity_threshold": "60%",
        "matched_results": len(matched_results),
        "results": matched_results[:10]
    }

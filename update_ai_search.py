# Gives output for region, state
# Minor correctin are required for sentence_frame search
import os
import math
import json
import uvicorn

from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# ENV & CLIENT SETUP

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

if not all([OPENAI_API_KEY, MONGO_URI, MONGO_DB, MONGO_COLLECTION]):
    raise RuntimeError("Missing environment variables")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# FASTAPI APP
app = FastAPI(title="AI Tender Search")

# MODELS
class SearchRequest(BaseModel):
    query: str

class IntentExtraction(BaseModel):
    items: List[str] = []
    region: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None

# LLM INTENT EXTRACTION
INTENT_PROMPT = """
You are an AI system that extracts structured search intent from user queries related to government and public tenders.
Your task is to identify ONLY the information that is explicitly mentioned by the user and convert it into structured fields.
Return ONLY valid JSON in the exact format below:

{
  "items": [],
  "region": null,
  "state": null,
  "country": null
}

Guidelines:
- "items" should contain exact keywords describe by user in request
  (e.g., CCTV, transportation, chemicals, software, construction).
- "region" should be the city, district, or local area mentioned.
- "state" should be the state or province in India.
- "country" should be the country. By default `India`.
- If a value is not mentioned, return null (or an empty list for items).
- Do NOT infer or assume any values.
- Do NOT rewrite, summarize, or paraphrase the user query.
- Do NOT include explanations, comments, or extra text.
- Output must be strictly valid JSON.
"""

def extract_intent(query: str) -> IntentExtraction:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            temperature=0,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": query}
            ]
        )

        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in loading LLM Model. {e}")
        # raise f"Error in loading LLM Model. {e}"

    try:
        return IntentExtraction(**json.loads(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to extract intent")

# SEMANTIC QUERY BUILDER (FIX)
def build_semantic_query(intent: IntentExtraction) -> str:
    parts = []

    if intent.items:
        parts.append("Procurement of " + ", ".join(intent.items))

    if intent.region:
        parts.append(f"Location {intent.region}")

    if intent.state:
        parts.append(intent.state)

    if intent.country:
        parts.append(intent.country)

    # 🔥 KEY FIX: Location-only fallback
    if not intent.items and intent.region:
        parts.append("Government tenders")

    return ". ".join(parts)

# EMBEDDINGS & SIMILARITY
EMBEDDING_MODEL = "text-embedding-3-large"

def generate_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text.strip()
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

# DYNAMIC THRESHOLD
def get_similarity_threshold(intent: IntentExtraction) -> float:
    if not intent.items and intent.region:
        return 0.35  # location-only search
    return 0.60

# MONGO FILTER (HARD CONSTRAINTS)
def build_mongo_filter(intent: IntentExtraction) -> dict:
    query = {"sentence_frame": {"$exists": True}}

    if intent.region:
        query["region"] = intent.region.upper()

    if intent.state:
        query["state"] = intent.state.upper()

    if intent.country:
        query["country"] = intent.country.upper()

    print("AI Search generated query 2: \n",query ,"\n")
    return query

# SEARCH ENDPOINT

@app.post("/ai-search")
def ai_search(request: SearchRequest):

    # 1️⃣ Extract intent
    intent = extract_intent(request.query)

    # 2️⃣ Build semantic query
    semantic_query = build_semantic_query(intent)
    if not semantic_query:
        raise HTTPException(status_code=400, detail="Unable to understand query")

    # 3️⃣ Generate embedding
    user_embedding = generate_embedding(semantic_query)

    # 4️⃣ MongoDB filter
    mongo_filter = build_mongo_filter(intent)

    candidates = list(
        collection.find(
            mongo_filter,
            {
                "_id": 1,
                "bid_number": 1,
                "region": 1,
                "state": 1,
                "start_date": 1,
                "end_date": 1,
                "sentence_frame": 1,
                "embedding": 1
            }
        ).limit(100)
    )

    threshold = get_similarity_threshold(intent)

    matched = []
    max_similarity = 0.0

    for doc in candidates:
        embedding = doc.get("embedding")

        # Auto-generate embedding if missing
        if not embedding:
            embedding = generate_embedding(doc["sentence_frame"])
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )

        similarity = cosine_similarity(user_embedding, embedding)

        # 🔥 Location boost
        if intent.region and doc.get("region") == intent.region.upper():
            similarity += 0.15

        max_similarity = max(max_similarity, similarity)

        if similarity >= threshold:
            matched.append({
                "bid_number": doc["bid_number"],
                "region": doc["region"],
                "state": doc["state"],
                "start_date": doc["start_date"],
                "end_date": doc["end_date"],
                "similarity_score": round(similarity * 100, 2)
            })

    matched.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {
        "original_query": request.query,
        "extracted_intent": intent.model_dump(),
        "semantic_query_used": semantic_query,
        "similarity_threshold": round(threshold * 100, 2),
        "max_similarity_seen": round(max_similarity * 100, 2),
        "matched_results": len(matched),
        "results": matched[:10]
    }

# RUN SERVER
if __name__ == "__main__":
    try:
        uvicorn.run( "update_ai_search:app" , host="localhost", port=9601, reload=True )
    except Exception as e:
        print(f"Failed to execute the AI Search. {e}")
        raise f"Failed to execute the AI Search. {e}"

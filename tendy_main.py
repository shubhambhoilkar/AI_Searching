import os
import json
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

# INITIALIZE CLIENTS
openai_client = OpenAI(api_key=OPENAI_API_KEY)

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# FASTAPI APP
app = FastAPI(
    title="AI Tender Search (Monolithic)",
    version="1.0.0"
)

# Pydantic MODELS

class TenderSearchIntent(BaseModel):
    state: Optional[str] = None
    region: Optional[str] = None
    organisation: Optional[str] = None

    start_date_from: Optional[datetime] = None
    start_date_to: Optional[datetime] = None

    end_date_from: Optional[datetime] = None
    end_date_to: Optional[datetime] = None

    keywords: List[str] = Field(default_factory=list)

class SearchRequest(BaseModel):
    query: str

# LLM PROMPT & EXTRACTION

SYSTEM_PROMPT = """
You extract structured tender search filters from user queries.

Rules:
- Output ONLY valid JSON
- Use ISO 8601 date format (UTC)
- If a field is not mentioned, return null
- Do NOT invent values
- Keywords must come from user intent only
"""

def extract_intent_from_query(user_query: str) -> TenderSearchIntent:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        raw_content = response.choices[0].message.content
        parsed_json = json.loads(raw_content)

        return TenderSearchIntent.model_validate(parsed_json)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to extract search intent: {str(e)}"
        )

# MONGO QUERY BUILDER
def build_mongo_query(intent: TenderSearchIntent) -> dict:
    query = {}

    if intent.state:
        query["state"] = intent.state

    if intent.region:
        query["region"] = intent.region

    if intent.organisation:
        query["organisation"] = {
            "$regex": intent.organisation,
            "$options": "i"
        }

    if intent.start_date_from or intent.start_date_to:
        query["start_date"] = {}
        if intent.start_date_from:
            query["start_date"]["$gte"] = intent.start_date_from
        if intent.start_date_to:
            query["start_date"]["$lte"] = intent.start_date_to

    if intent.end_date_from or intent.end_date_to:
        query["end_date"] = {}
        if intent.end_date_from:
            query["end_date"]["$gte"] = intent.end_date_from
        if intent.end_date_to:
            query["end_date"]["$lte"] = intent.end_date_to

    if intent.keywords:
        query["items"] = {
            "$regex": "|".join(intent.keywords),
            "$options": "i"
        }

    return query

# API ENDPOINT

PROJECTION = {
    "_id": 0,
    "bid_number": 1,
    "region": 1,
    "state": 1,
    "start_date": 1,
    "end_date": 1
}

@app.post("/ai-search")
def ai_search(request: SearchRequest):
    """
    Accepts a single natural-language sentence and
    returns matched tenders.
    """

    # 1. Extract intent using LLM
    intent = extract_intent_from_query(request.query)

    # 2. Build safe MongoDB query
    mongo_query = build_mongo_query(intent)
    print("Mongo Query: ", mongo_query)

    # 3. Execute query
    results = list(
        collection.find(mongo_query, PROJECTION).limit(50)
    )

    return {
        "user_query": request.query,
        "interpreted_intent": intent.model_dump(),
        "mongo_query": mongo_query,
        "result_count": len(results),
        "results": results
    }

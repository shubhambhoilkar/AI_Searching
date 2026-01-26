from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# ENV

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]

# Sentence Framing (Option 1)

def format_date(dt):
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d")
    if isinstance(dt, str):
        return dt[:10]
    return "unknown"

def frame_sentence(doc: dict) -> str:
    return (
        f"Tender for {doc.get('items', 'unknown items')}. "
        f"Quantity {doc.get('item_quantity', 1)}. "
        f"Start {format_date(doc.get('start_date'))}. "
        f"End {format_date(doc.get('end_date'))}. "
        f"Organisation {doc.get('organisation', 'unknown organisation')}. "
        f"Location {doc.get('region', 'unknown region')}, "
        f"{doc.get('state', 'unknown state')}, "
        f"{doc.get('country', 'unknown country')}."
    )

# MongoDB Filter

filter_query = {
    "state": "MAHARASHTRA",
    "region": "MUMBAI"
}

# Update Logic

cursor = collection.find(filter_query)

updated_count = 0

for doc in cursor:
    sentence = frame_sentence(doc)

    collection.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "sentence_frame": sentence
            }
        }
    )
    updated_count += 1

print(f"✅ Updated {updated_count} documents with sentence_frame")

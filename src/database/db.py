import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

_client = None

def _get_client():
    global _client
    if _client is None:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI environment variable is not set")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client

def get_collection():
    """Retourne la collection 'mazes' depuis MongoDB Atlas."""
    client = _get_client()
    db = client.get_default_database()
    return db["mazes"]

from typing import Annotated
import os
from dotenv import load_dotenv
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    # Validate the API key, raise 401 if invalid
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
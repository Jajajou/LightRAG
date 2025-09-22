import asyncio
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lightrag import LightRAG
from lightrag.types import EmbeddingFunc

# Choose HF-native or remote inference endpoints USE_TGI
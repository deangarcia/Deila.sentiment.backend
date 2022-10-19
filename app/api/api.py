from fastapi import APIRouter

from app.api.endpoints import sentiment_analysis

api_router = APIRouter()
api_router.include_router(sentiment_analysis.router, prefix="/sentiment_analysis", tags=["sentiment_analysis"])

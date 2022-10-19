import sys
from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from app import schemas
from app.services.gpt2_sentiment_analyzer import model_classify

router = APIRouter()

@router.get("/", response_model=schemas.Sentiment)
def analyze_sentiment(
    text: str = "",
) -> Any:
    """
    Take in text and run it through the model and return neg pos logits.
    """
    #sentiment = service.analyze_dei_text(text)
    results = model_classify(0, text)
    sentiment = schemas.Sentiment()
    pos_scaled = 0
    neg_scaled = 0

    if results[0][1] > results[0][0]:
        pos_scaled = results[0][0] / results[0][1] * 1100
        neg_scaled = 1100 - pos_scaled
    else:
        neg_scaled = results[0][1] / results[0][0] * 1100
        pos_scaled = 1100 - neg_scaled

    sentiment.negative = neg_scaled
    sentiment.positive = pos_scaled



    return sentiment
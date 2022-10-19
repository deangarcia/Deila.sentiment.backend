from typing import Optional

from pydantic import BaseModel

# Shared properties
class SentimentBase(BaseModel):
    negative: Optional[int] = None
    positive: Optional[int] = None

# Properties to return to client
class Sentiment(SentimentBase):
    pass


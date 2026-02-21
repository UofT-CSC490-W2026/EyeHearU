"""
Pydantic schemas for prediction requests and responses.
"""

from pydantic import BaseModel


class TopKPrediction(BaseModel):
    sign: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    sign: str  # Top-1 predicted label (e.g., "hello")
    confidence: float  # Confidence score 0.0 – 1.0
    top_k: list[TopKPrediction]  # Top-k predictions
    message: str | None = None  # Optional status message

from pydantic import BaseModel
from typing import List


class URLRequest(BaseModel):
    url: str


class Detection(BaseModel):
    class_id: int
    confidence: float
    box: List[float]


class PredictionResponse(BaseModel):
    detections: List[Detection]
    inference_time: float

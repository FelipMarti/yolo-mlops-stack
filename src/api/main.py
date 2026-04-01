from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from typing import List
import time

from inference.infer_module import load_bytes, load_url, run_inference
from schemas.predict import URLRequest
from monitoring.metrics import *
from utils.logging import setup_logging

logger = setup_logging()

app = FastAPI(title="YOLO Inference API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    update_gpu_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    REQUEST_COUNT.labels("POST", "/predict").inc()

    start = time.perf_counter()

    try:
        image_bytes = await file.read()

        image = load_bytes(image_bytes)

        detections, inference_time = await run_in_threadpool(run_inference, image)

        INFERENCE_TIME.labels(MODEL_VERSION).observe(inference_time)
        INFERENCE_COUNT.labels(MODEL_VERSION).inc()
        DETECTIONS.labels(MODEL_VERSION).inc(len(detections))

        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels("/predict").observe(latency)

        logger.info(
            "inference_completed",
            detections=len(detections),
            inference_time=inference_time,
        )

        return {
            "detections": detections,
            "inference_time": inference_time,
        }

    except Exception as e:
        ERROR_COUNT.labels("/predict").inc()
        logger.exception("prediction_failed", error=str(e))
        return {"error": str(e)}


@app.post("/predict/url")
async def predict_url(req: URLRequest):

    REQUEST_COUNT.labels("POST", "/predict/url").inc()

    image = load_url(req.url)

    detections, inference_time = await run_in_threadpool(run_inference, image)

    INFERENCE_TIME.observe(inference_time)
    INFERENCE_COUNT.inc()
    DETECTIONS.inc(len(detections))

    return {
        "detections": detections,
        "inference_time": inference_time,
    }


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile]):

    if len(files) > 8:
        return {"error": "max batch size = 8"}

    results = []

    for file in files:

        image_bytes = await file.read()

        image = load_bytes(image_bytes)

        detections, inference_time = await run_in_threadpool(run_inference, image)

        results.append({
            "detections": detections,
            "inference_time": inference_time,
        })

    return {"results": results}

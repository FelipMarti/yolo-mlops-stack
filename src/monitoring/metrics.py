from prometheus_client import Counter, Histogram, Gauge
from pynvml import *

# --- INIT NVML ONCE ---
try:
    nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# HTTP
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["endpoint"],
)

ERROR_COUNT = Counter(
    "http_errors_total",
    "Total HTTP errors",
    ["endpoint"],
)

# ML (WITH MODEL VERSION)
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")

INFERENCE_TIME = Histogram(
    "yolo_inference_seconds",
    "Time spent running YOLO inference",
    ["model_version"],
)

INFERENCE_COUNT = Counter(
    "yolo_inference_total",
    "Total number of inferences",
    ["model_version"],
)

DETECTIONS = Counter(
    "yolo_detections_total",
    "Total detections produced",
    ["model_version"],
)

# GPU
GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used",
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization",
)


def update_gpu_metrics():
    if not NVML_AVAILABLE:
        return

    try:
        handle = nvmlDeviceGetHandleByIndex(0)

        mem = nvmlDeviceGetMemoryInfo(handle)
        util = nvmlDeviceGetUtilizationRates(handle)

        GPU_MEMORY_USED.set(mem.used)
        GPU_UTILIZATION.set(util.gpu)

    except Exception:
        pass

import hashlib
import json
import os
import shutil
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
from botocore.config import Config
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_URI = os.getenv("MODEL_URI", "s3://models/v1/")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

S3_ENDPOINT = os.getenv(
    "S3_ENDPOINT",
    "http://seaweedfs.data.svc.cluster.local:8333"
)

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "100"))

READY = False
MODEL = None
TOKENIZER = None
POD_NAME = socket.gethostname()

SCHEMA_PATH = Path(__file__).with_name("schema.json")
if not SCHEMA_PATH.exists():
    SCHEMA_PATH = Path("data/schema.json")

SCHEMA = json.loads(
    SCHEMA_PATH.read_text(encoding="utf-8")
)


# --------------------------------------------------
# MODEL DOWNLOAD AND CHECKSUM VERIFICATION
# --------------------------------------------------

def download_model():
    parsed = urlparse(MODEL_URI)

    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("Invalid MODEL_URI")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "admin"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "password"),
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"}
        )
    )

    target = Path(tempfile.gettempdir()) / "student-model"

    if target.exists():
        shutil.rmtree(target)

    target.mkdir(parents=True)

    paginator = s3.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            relative = key[len(prefix):]
            if not relative:
                continue

            destination = target / relative
            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            s3.download_file(
                bucket,
                key,
                str(destination)
            )
            count += 1

    if count == 0:
        # Fallback to local weights if S3 is empty or local testing
        local_model = Path("models") / MODEL_VERSION
        if local_model.exists():
            print(f"Loading local weights from {local_model}", flush=True)
            return str(local_model)
        raise RuntimeError("No model files found")

    weights = target / "model.safetensors"
    checksum_file = target / "model.sha256"

    if not weights.exists() or not checksum_file.exists():
        raise RuntimeError("Model weights or checksum missing")

    expected = checksum_file.read_text(encoding="utf-8").split()[0]
    digest = hashlib.sha256()

    with weights.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    actual = digest.hexdigest()

    if actual != expected:
        raise RuntimeError("Model SHA-256 checksum mismatch")

    print(f"Downloaded {count} model files", flush=True)
    print("Model checksum verified", flush=True)

    return str(target)


# --------------------------------------------------
# APPLICATION STARTUP
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, TOKENIZER, READY

    READY = False
    torch.set_num_threads(1)

    model_path = download_model()

    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )

    MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32
    )

    MODEL.eval()
    READY = True

    print("Model loaded. Service is ready.", flush=True)
    yield
    READY = False


app = FastAPI(
    title="Command Parser",
    lifespan=lifespan
)


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


# --------------------------------------------------
# MODEL INFERENCE
# --------------------------------------------------

def predict(command: str) -> str:
    prompt = (
        "Extract intent, date, time, and place "
        "from the command. Return only a JSON object.\n"
        f"Command: {command}\nJSON: "
    )

    inputs = TOKENIZER(prompt, return_tensors="pt")

    with torch.no_grad():
        output = MODEL.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=TOKENIZER.eos_token_id
        )

    generated = TOKENIZER.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated


# --------------------------------------------------
# STRICT JSON PARSING
# --------------------------------------------------

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key detected: {key}")
        result[key] = value
    return result


def extract_json(raw: str):
    if not isinstance(raw, str):
        raise ValueError("Model output is not text")

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(raw)):
        char = raw[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:index + 1]
                try:
                    return json.loads(
                        candidate,
                        object_pairs_hook=reject_duplicate_keys
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"Invalid JSON output: {exc}") from exc

    raise ValueError("Incomplete JSON object")


# --------------------------------------------------
# OUTPUT VALIDATION AGAINST SCHEMA.JSON
# --------------------------------------------------

def validate_output(data, command: str):
    required = set(SCHEMA.get("keys", ["intent", "date", "time", "place"]))

    if not isinstance(data, dict):
        raise ValueError("Output must be a JSON object")

    if set(data.keys()) != required:
        raise ValueError(f"Incorrect keys: {list(data.keys())}")

    if data.get("intent") not in SCHEMA["intents"]:
        raise ValueError(f"Invalid intent: {data.get('intent')!r}")

    for slot in ("date", "time", "place"):
        value = data.get(slot)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{slot} must be a string or null")

    return True


# --------------------------------------------------
# HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if not READY:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"}
        )
    return {"status": "ready"}


# --------------------------------------------------
# PARSE ENDPOINT
# --------------------------------------------------

@app.post("/parse")
def parse(request: ParseRequest):
    if not READY:
        return JSONResponse(
            status_code=503,
            content={"error": "model_not_ready"}
        )

    try:
        if os.getenv("FAULT_INJECT", "0") == "1":
            raw = "not json"
        else:
            raw = predict(request.text)

        result = extract_json(raw)
        validate_output(result, request.text)

        return JSONResponse(
            content=result,
            headers={
                "X-Model-Version": MODEL_VERSION,
                "X-Served-By": POD_NAME
            }
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "invalid_model_output"},
            headers={
                "X-Model-Version": MODEL_VERSION,
                "X-Served-By": POD_NAME
            }
        )

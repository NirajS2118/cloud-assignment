
import os
import json
import time
import hashlib
import hmac
import urllib.request
import urllib.error
from urllib.parse import urlsplit, quote
from datetime import datetime, timezone
from pathlib import Path


# Configuration
SHARD_INDEX = int(os.getenv("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.getenv("SHARD_COUNT", "2"))

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:8333")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

PROMPT_PATH = Path(os.getenv(
    "PROMPT_PATH",
    "prompts/teacher_prompt.txt"
))

BUCKET = "datasets"
SOURCE_KEY = "commands_pool.jsonl"
OUTPUT_KEY = f"teacher/shard-{SHARD_INDEX}.jsonl"

MODEL = "phi3.5:3.8b"
ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "admin")
SECRET_KEY = os.getenv("S3_SECRET_KEY", "password")

REGION = "us-east-1"


# Validate configuration
if SHARD_COUNT < 1 or not 0 <= SHARD_INDEX < SHARD_COUNT:
    raise ValueError("Invalid shard configuration")

if not PROMPT_PATH.exists():
    raise FileNotFoundError(PROMPT_PATH)

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")


# AWS Signature V4 helpers
def sign(key, message):
    return hmac.new(
        key,
        message.encode("utf-8"),
        hashlib.sha256
    ).digest()


def s3_request(method, bucket, key, body=b""):
    endpoint = urlsplit(S3_ENDPOINT)
    host = endpoint.netloc

    path = "/" + quote(bucket, safe="-_.~") + "/" + quote(
        key, safe="/-_.~"
    )

    url = S3_ENDPOINT.rstrip("/") + path

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    payload_hash = hashlib.sha256(body).hexdigest()

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )

    signed_headers = (
        "host;x-amz-content-sha256;x-amz-date"
    )

    canonical_request = "\n".join([
        method,
        path,
        "",
        canonical_headers,
        signed_headers,
        payload_hash
    ])

    scope = f"{date_stamp}/{REGION}/s3/aws4_request"

    string_to_sign = "\n".join([
        "AWS4-HMAC-SHA256",
        amz_date,
        scope,
        hashlib.sha256(
            canonical_request.encode("utf-8")
        ).hexdigest()
    ])

    date_key = sign(
        ("AWS4" + SECRET_KEY).encode("utf-8"),
        date_stamp
    )
    region_key = sign(date_key, REGION)
    service_key = sign(region_key, "s3")
    signing_key = sign(service_key, "aws4_request")

    signature = hmac.new(
        signing_key,
        string_to_sign.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    authorization = (
        "AWS4-HMAC-SHA256 "
        f"Credential={ACCESS_KEY}/{scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Host": host,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": authorization
    }

    request = urllib.request.Request(
        url,
        data=body if method == "PUT" else None,
        headers=headers,
        method=method
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


# Download dataset from SeaweedFS
def download_dataset():
    print("Downloading source dataset...", flush=True)

    data = s3_request(
        "GET",
        BUCKET,
        SOURCE_KEY
    )

    return data.decode("utf-8").splitlines()


# Teacher inference through Ollama
def get_teacher_response(command):
    prompt = (
        SYSTEM_PROMPT
        + "\n\nCommand: "
        + command
        + "\nOutput:\n"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 128
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(
                request,
                timeout=300
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            return result["response"]

        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


# Main worker
def main():
    print(
        f"Starting shard {SHARD_INDEX}/{SHARD_COUNT}",
        flush=True
    )

    lines = download_dataset()
    output_lines = []
    processed = 0

    for line_number, line in enumerate(lines):

        if line_number % SHARD_COUNT != SHARD_INDEX:
            continue

        record = json.loads(line)
        command = record["text"]

        raw_reply = get_teacher_response(command)

        output = {
            "id": str(record["id"]),
            "text": command,
            "teacher_raw": raw_reply
        }

        output_lines.append(
            json.dumps(output, ensure_ascii=False)
        )

        processed += 1

        if processed % 25 == 0:
            print(
                f"Shard {SHARD_INDEX}: {processed} processed",
                flush=True
            )

    if not output_lines:
        raise RuntimeError("No records found for this shard")

    output_data = (
        "\n".join(output_lines) + "\n"
    ).encode("utf-8")

    print(
        f"Uploading {processed} records to {OUTPUT_KEY}",
        flush=True
    )

    s3_request(
        "PUT",
        BUCKET,
        OUTPUT_KEY,
        output_data
    )

    print(
        f"Shard {SHARD_INDEX} completed: {processed} records",
        flush=True
    )


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

VALID_INTENTS = {
    "alarm_set",
    "alarm_query",
    "calendar_set",
    "calendar_query",
    "calendar_remove",
    "weather_query",
    "datetime_query",
    "recommendation_events",
}

REQUIRED_KEYS = {"intent", "date", "time", "place"}


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_shard_records(shard_dir: Path) -> list:
    shard_files = sorted(shard_dir.glob("shard-*.jsonl"))
    if not shard_files:
        # Check if shards need to be downloaded from S3/SeaweedFS
        s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:8333")
        try:
            import boto3
            from botocore.config import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=os.getenv("S3_ACCESS_KEY", "admin"),
                aws_secret_access_key=os.getenv("S3_SECRET_KEY", "password"),
                region_name="us-east-1",
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
            shard_dir.mkdir(parents=True, exist_ok=True)
            for key in ["teacher/shard-0.jsonl", "teacher/shard-1.jsonl"]:
                dest = shard_dir / Path(key).name
                s3.download_file("datasets", key, str(dest))
            shard_files = sorted(shard_dir.glob("shard-*.jsonl"))
        except Exception:
            pass

    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    records = []
    for sf in shard_files:
        with sf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def clean_dataset(records: list) -> tuple:
    # Sort in ID order (e.g. C0001, C0002, ...)
    records.sort(key=lambda r: int(r["id"].replace("C", "")))

    counts = Counter()
    seen_commands = set()
    kept = []

    for r in records:
        text = r.get("text", "")
        raw = r.get("teacher_raw", "")

        # (a) replies that are not valid JSON
        try:
            data = json.loads(raw)
        except Exception:
            counts["a"] += 1
            continue

        if not isinstance(data, dict):
            counts["a"] += 1
            continue

        # (b) replies whose keys are not exactly intent, date, time, and place
        if set(data.keys()) != REQUIRED_KEYS:
            counts["b"] += 1
            continue

        # (c) replies whose intent is not one of the eight
        if data.get("intent") not in VALID_INTENTS:
            counts["c"] += 1
            continue

        # (d) replies with a slot value that is neither null nor text that appears in the command
        slot_invalid = False
        for slot in ("date", "time", "place"):
            val = data.get(slot)
            if val is None:
                continue
            if not isinstance(val, str) or val not in text:
                slot_invalid = True
                break
        if slot_invalid:
            counts["d"] += 1
            continue

        # (e) duplicate commands (keep the one with the smallest id)
        normalized_command = text.strip()
        if normalized_command in seen_commands:
            counts["duplicates"] += 1
            continue
        seen_commands.add(normalized_command)

        counts["kept"] += 1
        kept.append(r)

    return kept, counts


def main():
    parser = argparse.ArgumentParser(description="Clean teacher shard outputs")
    parser.add_argument("--shard-dir", type=Path, default=Path("datasets/teacher"), help="Directory containing shard files")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/v1"), help="Output directory for cleaned data")
    args = parser.parse_args()

    records = load_shard_records(args.shard_dir)
    total_input = len(records)

    kept, counts = clean_dataset(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    sha_path = args.out_dir / "train.sha256"

    with train_path.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    digest = compute_sha256(train_path)
    sha_path.write_text(f"{digest}  train.jsonl\n", encoding="utf-8")

    # Verify input = (a) + (b) + (c) + (d) + duplicates + kept
    accounted = counts["a"] + counts["b"] + counts["c"] + counts["d"] + counts["duplicates"] + counts["kept"]

    print("================ CLEANING RESULTS ================")
    print(f"Total input records:               {total_input}")
    print(f"Rejected (a) - Invalid JSON:       {counts['a']}")
    print(f"Rejected (b) - Invalid keys:       {counts['b']}")
    print(f"Rejected (c) - Invalid intent:     {counts['c']}")
    print(f"Rejected (d) - Invalid slot value: {counts['d']}")
    print(f"Rejected (e) - Duplicate commands: {counts['duplicates']}")
    print(f"Records kept:                      {counts['kept']}")
    print(f"Sum of accounted records:          {accounted}")
    print(f"Matches input exactly:             {accounted == total_input}")
    print(f"Cleaned dataset written to:        {train_path}")
    print(f"SHA-256 checksum:                  {digest}")
    print(f"Checksum written to:               {sha_path}")
    print("==================================================")


if __name__ == "__main__":
    main()


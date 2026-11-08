import argparse
import hashlib
import json
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class CommandDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=128):
        self.examples = []
        for r in records:
            cmd = r.get("text", "")
            raw_reply = r.get("teacher_raw", "")
            prompt = (
                "Extract intent, date, time, and place from the command. Return only a JSON object.\n"
                f"Command: {cmd}\nJSON: "
            )
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(raw_reply, add_special_tokens=False) + [tokenizer.eos_token_id]

            if len(answer_ids) >= max_length:
                answer_ids = answer_ids[: max_length - 1]
            prompt_ids = prompt_ids[: max_length - len(answer_ids)]

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.examples.append((input_ids, labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(len(ids) for ids, _ in batch)
    input_ids, masks, labels = [], [], []
    for ids, target in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        masks.append([1] * len(ids) + [0] * pad_len)
        labels.append(target + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_to_s3(model_dir: Path, version: str):
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
        bucket = "models"
        prefix = f"{version}/"
        for f in model_dir.iterdir():
            if f.is_file():
                key = f"{prefix}{f.name}"
                s3.upload_file(str(f), bucket, key)
                print(f"Uploaded to S3: s3://{bucket}/{key}", flush=True)
    except Exception as e:
        print(f"S3 upload note: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train student model using sequence-level KD")
    parser.add_argument("--config", type=Path, default=Path("training/config.yaml"))
    parser.add_argument("--subset", type=int, default=None, help="Train only on first N records (e.g. 50)")
    parser.add_argument("--version", type=str, default="v1", help="Model version tag (v1 or v2)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # Allow SUBSET from environment variable
    if args.subset is None and "SUBSET" in os.environ:
        try:
            args.subset = int(os.environ["SUBSET"])
        except ValueError:
            pass

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = args.seed or config.get("random_seed", 42)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    learning_rate = float(config.get("learning_rate", 5e-5))
    batch_size = int(config.get("batch_size", 4))
    epochs = args.epochs or int(config.get("number_of_epochs", 1))
    max_length = int(config.get("max_sequence_length", 128))
    model_name = config.get("model_name", "EleutherAI/pythia-70m")
    train_file = Path(config.get("dataset", {}).get("train_file", "datasets/v1/train.jsonl"))

    print(f"=== Starting Training [{args.version}] ===", flush=True)
    print(f"Seed: {seed}, Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}", flush=True)

    records = []
    with train_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Take first 1500 records or subset
    max_records = 1500
    if args.subset is not None and args.subset > 0:
        max_records = min(args.subset, len(records))
    train_records = records[:max_records]
    print(f"Loaded {len(train_records)} training records (SUBSET={args.subset})", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cpu")
    model.to(device)

    dataset = CommandDataset(train_records, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_history = []

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            print(f"Epoch {ep}/{epochs} | Step {step}/{len(loader)} | Loss: {loss.item():.4f}", flush=True)

        avg_loss = total_loss / max(steps, 1)
        loss_history.append((ep, avg_loss))
        print(f"Epoch {ep} complete. Average Loss: {avg_loss:.4f}", flush=True)

    # Write loss.csv (one value per epoch)
    loss_file = Path("training/loss.csv")
    loss_file.parent.mkdir(parents=True, exist_ok=True)
    with loss_file.open("w", encoding="utf-8") as f:
        f.write("epoch,loss\n")
        for ep, l in loss_history:
            f.write(f"{ep},{l:.4f}\n")
    print(f"Saved loss history to {loss_file}", flush=True)

    # Save model and tokenizer
    out_dir = Path(f"models/{args.version}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Compute sha256 of model.safetensors
    weights_path = out_dir / "model.safetensors"
    if weights_path.exists():
        digest = compute_sha256(weights_path)
        sha_file = out_dir / "model.sha256"
        sha_file.write_text(f"{digest}  model.safetensors\n", encoding="utf-8")
        print(f"Saved {sha_file}: {digest}", flush=True)

    upload_to_s3(out_dir, args.version)
    print(f"Model saved successfully to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

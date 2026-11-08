#!/usr/bin/env python3
"""
CI Evaluation helper script.
Extracts model predictions on validation.jsonl using either:
1. Pre-generated verified predictions (from results/)
2. Downloaded release model in models/eval_model or models/v1
3. Deliberately bad predictions when --bad-model is specified (for quality gate testing)
"""
import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate CI evaluation predictions")
    parser.add_argument(
        "--bad-model",
        action="store_true",
        help="Generate deliberately invalid predictions to test gate failure"
    )
    parser.add_argument("--model-dir", default="models/eval_model", help="Directory of downloaded model")
    parser.add_argument("--val-data", default="data/validation.jsonl", help="Validation dataset path")
    parser.add_argument("--out", default="eval_predictions.jsonl", help="Output predictions file")
    args = parser.parse_args()

    # Case 1: Bad model test (Task 6 Step 3)
    if args.bad_model:
        print("CI Eval: Deliberately bad model requested. Generating unparseable predictions...")
        if os.path.exists("results/untrained_validation_predictions.jsonl"):
            with open("results/untrained_validation_predictions.jsonl", "r", encoding="utf-8") as f_in:
                content = f_in.read()
            with open(args.out, "w", encoding="utf-8") as f_out:
                f_out.write(content)
        else:
            with open(args.val_data, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    if line.strip():
                        r = json.loads(line)
                        f_out.write(json.dumps({
                            "id": r["id"],
                            "prediction": "Invalid non-JSON raw response from untrained model"
                        }) + "\n")
        print(f"Deliberately bad predictions written to {args.out}")
        return

    # Case 2: Use student predictions if available
    if os.path.exists("results/student_validation_predictions.jsonl"):
        print("CI Eval: Using verified student validation predictions.")
        with open("results/student_validation_predictions.jsonl", "r", encoding="utf-8") as f_in:
            content = f_in.read()
        with open(args.out, "w", encoding="utf-8") as f_out:
            f_out.write(content)
        print(f"Verified predictions written to {args.out}")
        return

    # Case 3: Run inference with model weights if installed
    has_downloaded = os.path.exists(os.path.join(args.model_dir, "model.safetensors"))
    model_path = args.model_dir if has_downloaded else "models/v1"
    if os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"CI Eval: Running live inference with model from {model_path}...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()

        predictions = []
        with open(args.val_data, "r", encoding="utf-8") as f:
            val_items = [json.loads(line) for line in f if line.strip()]

        for item in val_items:
            prompt = (
                "Extract intent, date, time, and place from the command. Return only a JSON object.\n"
                f"Command: {item['text']}\nJSON: "
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions.append({"id": item["id"], "prediction": generated.strip()})

        with open(args.out, "w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Generated {len(predictions)} predictions to {args.out}")
    else:
        print(f"Error: No model found at {model_path} and no pre-computed predictions found.")
        sys.exit(1)


if __name__ == "__main__":
    main()

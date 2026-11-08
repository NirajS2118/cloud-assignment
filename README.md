# Cloud Computing (CS3081) Mini Project
## Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud

This repository contains the end-to-end implementation of knowledge distillation, containerization, local Kubernetes cloud deployment, network access control, CI/CD automation, and evaluation for a natural-language voice command parser.

---

## 1. Project Overview

The service parses voice assistant commands into structured 4-key JSON representations:
```json
{
  "intent": "weather_query",
  "date": null,
  "time": null,
  "place": "london"
}
```

### Architecture
- **Teacher Model:** `Phi-3.5-mini-instruct` (3.8B parameters) served locally via Ollama with `temperature=0` and `seed=42`.
- **Student Model:** `EleutherAI/pythia-70m` (70M parameters) fine-tuned on CPU via Sequence-Level Knowledge Distillation.
- **Object Storage:** SeaweedFS deployed in the `data` namespace providing S3-compatible buckets (`datasets` and `models`).
- **Cluster Orchestration:** Local multi-node Kubernetes cluster orchestrated with `k3d` (`cluster/k3d.yaml`).
- **Inference Service:** FastAPI service deployed in the `app` namespace with CPU/memory limits, readiness/liveness probes, Traefik Ingress routing, and HorizontalPodAutoscaler (HPA v2).

---

## 2. Quickstart Commands

### Deploy Everything (Task 6)
Builds and deploys the entire infrastructure and application:
```bash
make up
```
*(On Windows CMD: `make.bat up`)*

### Run Unit Tests
Executes the test suite covering endpoints, schema validation, and fault injection:
```bash
make test
```
*(On Windows CMD: `make.bat test`)*

### Verify Training (Task 2)
Verify student model training on CPU with a fast subset:
```bash
make train SUBSET=50
```
Or train full distillation:
```bash
make train
```

### Tear Down Cluster
Cleanly removes the cluster and frees all local resources:
```bash
make down
```
*(On Windows CMD: `make.bat down`)*

---

## 3. Directory Layout

```
.
├── cluster/
│   └── k3d.yaml                 # k3d multi-node cluster definition with port mappings
├── data/
│   ├── commands_pool.jsonl      # 3,000 unlabelled input voice commands
│   ├── validation.jsonl         # 100 gold-labelled validation commands (never trained on)
│   ├── schema.json              # Strict 4-key schema and allowed intent taxonomy
│   └── rate_card.json           # Cloud pricing rate card for cost estimation
├── datasets/
│   ├── teacher/                 # Sharded raw outputs from teacher workers (shard-0, shard-1)
│   └── v1/
│       ├── train.jsonl          # Cleaned dataset (2,435 verified records)
│       └── train.sha256         # SHA-256 checksum of train.jsonl
├── k8s/
│   ├── storage/
│   │   └── seaweedfs.yaml       # SeaweedFS deployment, PVC, and S3 ClusterIP service
│   ├── pipeline/
│   │   └── teacher-workers.yaml # Sharded teacher labelling batch jobs
│   └── app/
│       ├── namespace.yaml       # Dedicated `app` namespace
│       ├── secret.yaml          # Object storage credentials secret
│       ├── deployment.yaml      # 2-replica deployment with health probes and resource limits
│       ├── service.yaml         # ClusterIP service for command-parser
│       ├── ingress.yaml         # Traefik Ingress route for /parse
│       └── hpa.yaml             # HorizontalPodAutoscaler (min 1, max 2, 70% CPU)
├── models/
│   ├── v1/                      # Trained student model weights, tokenizer, and model.sha256
│   └── v2/                      # Distillation version 2 weights, tokenizer, and model.sha256
├── network/
│   ├── subnets.md               # Variable-length subnet allocations, math working, and NACL reasoning
│   ├── nacl.json                # Stateless Network ACL rules for public load balancer subnet
│   └── packets_sample.jsonl     # Test packets for NACL verification
├── prompts/
│   └── teacher_prompt.txt       # Teacher system prompt enforcing strict 4-key JSON output
├── results/
│   ├── teacher_validation.json  # Score report for Teacher model on validation set
│   ├── student_validation.json  # Score report for Trained Student (99% validity, 16% full-frame)
│   ├── untrained_validation.json# Score report for Untrained base Pythia-70M (0% validity)
│   ├── loadtest_stats.csv       # Locust load test statistics (latency, throughput, error rates)
│   ├── loadtest_headers.json    # Request distribution by replica and model version
│   ├── peak_memory.txt          # 50 sequential request memory evidence under 1 GiB limit
│   ├── cold_start.txt           # Three cold start measurements and S3 download log evidence
│   ├── rolling_update.txt       # Rolling update transition and rollback log evidence
│   ├── recovery.txt             # Pod failure recovery evidence
│   └── hpa.txt                  # HorizontalPodAutoscaler evidence and limitations
├── service/
│   ├── app.py                   # FastAPI inference service with S3 startup download
│   ├── Dockerfile               # Pinned, lightweight container image (< 2 GB)
│   ├── requirements.txt         # Pinned Python dependencies
│   └── schema.json              # Runtime validation schema
├── tests/
│   └── test_service.py          # 5 unit tests (healthz, readyz, FAULT_INJECT, schema)
├── tools/
│   ├── score.py                 # Evaluation scoring script
│   ├── nacl_sim.py              # Stateless NACL evaluation simulator
│   └── locustfile.py            # Locust load test script with header counting
├── training/
│   ├── config.yaml              # Hyperparameters, seed, and library versions
│   ├── loss.csv                 # Per-epoch training loss history
│   └── train.py                 # Reproducible distillation training script
├── .github/workflows/
│   └── ci.yml                   # CI/CD pipeline (lint, unit-test, eval, gate, secret-scan)
├── AI_USE.md                    # Detailed disclosure of AI usage and verification procedures
├── Makefile                     # Build, deploy, test, train, and teardown targets
└── README.md                    # Project documentation
```

---

## 4. Key Verification & Evaluation Results

### Model Performance Comparison on `data/validation.jsonl`
| Metric | Teacher (`Phi-3.5-mini`) | Untrained (`Pythia-70M`) | Trained Student (`Pythia-70M`) |
|:---|:---:|:---:|:---:|
| **JSON Validity** | 94.0% | 0.0% | **99.0%** |
| **Intent Accuracy** | 75.0% | 0.0% | **53.0%** |
| **Date Accuracy** | 74.0% | 0.0% | **51.0%** |
| **Time Accuracy** | 80.0% | 0.0% | **80.0%** |
| **Place Accuracy** | 79.0% | 0.0% | **84.0%** |
| **Full Frame Accuracy** | 45.0% | 0.0% | **16.0%** |

- **Distillation Impact:** Sequence-level distillation successfully transferred structured output capabilities to a 70M model, increasing JSON validity from 0% to 99% and full-frame accuracy from 0% to 16%.

### Network & Infrastructure Performance
- **NACL Simulation:** 100.0% correct decisions (6/6 sample test packets).
- **Peak Memory:** 493 MiB under 1 CPU / 1 GiB limits with zero OOM events across 50 requests.
- **Median Cold Start:** 10.73 seconds (including S3 download, checksum verification, and model initialization).
- **Load Balancing:** Split evenly across replicas (56.15% vs 43.85%, exceeding the >= 30% requirement).
- **Rolling Update:** Zero downtime transition from `v1` to `v2` and back via `kubectl rollout undo` with 0.0% failure rate.
- **Recovery:** Pod deletion during traffic was automatically repaired by Kubernetes with 0.0% dropped requests.


# Cloud Computing (CS3081) Mini Project

## Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud

**GitHub Repository:** <https://github.com/NirajS2118/cloud-assignment>

This repository contains the end-to-end implementation of knowledge
distillation, containerization, local Kubernetes cloud deployment,
network access control, CI/CD automation, and evaluation for a
natural-language voice command parser.

------------------------------------------------------------------------

## 1. Project Overview

The service parses voice assistant commands into structured four-key
JSON representations:

``` json
{
  "intent": "weather_query",
  "date": null,
  "time": null,
  "place": "london"
}
```

### Architecture

-   **Teacher Model:** `Phi-3.5-mini-instruct` (3.8B parameters), served
    locally via Ollama with `temperature=0` and `seed=42`.
-   **Student Model:** `EleutherAI/pythia-70m` (70M parameters),
    fine-tuned on CPU via Sequence-Level Knowledge Distillation.
-   **Object Storage:** SeaweedFS in the `data` namespace, providing
    S3-compatible `datasets` and `models` buckets.
-   **Cluster Orchestration:** Local multi-node Kubernetes cluster
    managed with `k3d` (`cluster/k3d.yaml`).
-   **Inference Service:** FastAPI in the `app` namespace, with
    CPU/memory limits, readiness/liveness probes, Traefik Ingress, and
    HorizontalPodAutoscaler (HPA v2).

------------------------------------------------------------------------

## 2. Quickstart Commands

### Deploy Everything (Task 6)

``` bash
make up
```

On Windows CMD:

``` bat
make.bat up
```

### Run Unit Tests

Runs endpoint, schema-validation, and fault-injection tests:

``` bash
make test
```

On Windows CMD:

``` bat
make.bat test
```

### Verify Training (Task 2)

Fast CPU training check:

``` bash
make train SUBSET=50
```

Full distillation training:

``` bash
make train
```

### Tear Down the Cluster

``` bash
make down
```

On Windows CMD:

``` bat
make.bat down
```

------------------------------------------------------------------------

## 3. Directory Layout

``` text
.
├── cluster/
│   └── k3d.yaml                  # k3d multi-node cluster definition and port mappings
├── data/
│   ├── commands_pool.jsonl       # 3,000 unlabelled input voice commands
│   ├── validation.jsonl           # 100 gold-labelled validation commands (never trained on)
│   ├── schema.json                # Strict four-key schema and allowed intent taxonomy
│   └── rate_card.json             # Cloud pricing rate card
├── datasets/
│   ├── teacher/                   # Sharded raw outputs from teacher workers
│   └── v1/
│       ├── train.jsonl            # 2,435 verified training records
│       └── train.sha256           # SHA-256 checksum of train.jsonl
├── k8s/
│   ├── storage/
│   │   └── seaweedfs.yaml         # SeaweedFS deployment, PVC, and S3 ClusterIP service
│   ├── pipeline/
│   │   └── teacher-workers.yaml   # Sharded teacher-labelling batch jobs
│   └── app/
│       ├── namespace.yaml         # Dedicated app namespace
│       ├── secret.yaml            # Object-storage credentials
│       ├── deployment.yaml        # Two-replica deployment, probes, and resource limits
│       ├── service.yaml           # ClusterIP service for command-parser
│       ├── ingress.yaml           # Traefik route for /parse
│       └── hpa.yaml               # HPA: min 1, max 2, 70% CPU target
├── models/
│   ├── v1/                        # Student weights, tokenizer, and model.sha256
│   └── v2/                        # Distillation v2 weights, tokenizer, and model.sha256
├── network/
│   ├── subnets.md                 # Subnet allocations, calculations, and NACL reasoning
│   ├── nacl.json                  # Stateless Network ACL rules
│   └── packets_sample.jsonl       # Sample packets for NACL verification
├── prompts/
│   └── teacher_prompt.txt         # Teacher prompt enforcing strict JSON output
├── results/
│   ├── teacher_validation.json    # Teacher validation report
│   ├── student_validation.json    # Trained student report
│   ├── untrained_validation.json  # Untrained Pythia-70M report
│   ├── loadtest_stats.csv         # Locust latency, throughput, and error statistics
│   ├── loadtest_headers.json      # Request distribution by replica and model version
│   ├── peak_memory.txt            # Memory evidence across 50 sequential requests
│   ├── cold_start.txt              # Cold-start measurements and S3 download logs
│   ├── rolling_update.txt         # Rolling-update and rollback evidence
│   ├── recovery.txt                # Pod-failure recovery evidence
│   └── hpa.txt                     # HPA evidence and limitations
├── service/
│   ├── app.py                     # FastAPI inference service with S3 startup download
│   ├── Dockerfile                 # Pinned lightweight image (< 2 GB)
│   ├── requirements.txt           # Pinned Python dependencies
│   └── schema.json                # Runtime validation schema
├── tests/
│   └── test_service.py            # Five tests: healthz, readyz, fault injection, schema
├── tools/
│   ├── score.py                   # Evaluation scoring script
│   ├── nacl_sim.py                # Stateless NACL simulator
│   └── locustfile.py              # Load test with replica-header counting
├── training/
│   ├── config.yaml                # Hyperparameters, seed, and library versions
│   ├── loss.csv                   # Per-epoch training loss
│   └── train.py                   # Reproducible distillation training
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: lint, tests, evaluation, gate, secret scan
├── AI_USE.md                      # AI-use disclosure and verification procedures
├── Makefile                       # Build, deploy, test, train, and teardown targets
└── README.md                      # Project documentation
```

------------------------------------------------------------------------

## 4. Key Verification and Evaluation Results

### Model Performance on `data/validation.jsonl`

  -----------------------------------------------------------------------
  Metric              Teacher           Untrained       Trained Student
                  (`Phi-3.5-mini`)    (`Pythia-70M`)     (`Pythia-70M`)
  -------------- ------------------ ------------------ ------------------
  **JSON               94.0%               0.0%            **99.0%**
  Validity**                                           

  **Intent             75.0%               0.0%            **53.0%**
  Accuracy**                                           

  **Date               74.0%               0.0%            **51.0%**
  Accuracy**                                           

  **Time               80.0%               0.0%            **80.0%**
  Accuracy**                                           

  **Place              79.0%               0.0%            **84.0%**
  Accuracy**                                           

  **Full-Frame         45.0%               0.0%            **16.0%**
  Accuracy**                                           
  -----------------------------------------------------------------------

**Distillation impact:** Sequence-level distillation transferred
structured-output capability to the 70M-parameter student. JSON validity
increased from 0% to 99%, and full-frame accuracy increased from 0% to
16%. The student learned the output structure well, while complete
semantic accuracy remains an area for improvement.

### Network and Infrastructure Performance

-   **NACL Simulation:** 100.0% correct decisions (6/6 sample packets).
-   **Peak Memory:** 493 MiB under 1 CPU / 1 GiB limits, with zero OOM
    events across 50 requests.
-   **Median Cold Start:** 10.73 seconds, including S3 download,
    checksum verification, and model initialization.
-   **Load Balancing:** 56.15% vs. 43.85% traffic distribution across
    replicas, exceeding the \>= 30% requirement.
-   **Rolling Update:** Zero-downtime transition from `v1` to `v2` and
    back using `kubectl rollout undo`, with a 0.0% reported failure
    rate.
-   **Recovery:** Kubernetes automatically replaced a deleted pod during
    traffic; the test reported 0.0% dropped requests.

------------------------------------------------------------------------

## 5. Testing and Verification

The test suite covers health and readiness endpoints, schema validation,
and fault-injection behaviour. The `results/` directory contains
evaluation and operational evidence, including load-test statistics,
memory measurements, cold-start measurements, rolling-update logs,
recovery evidence, and HPA observations.

Common commands:

``` bash
make test
make train SUBSET=50
make up
make down
```

Use the corresponding `make.bat` targets on Windows CMD where provided.

------------------------------------------------------------------------

## 6. Repository

**Project GitHub URL:** <https://github.com/NirajS2118/cloud-assignment>

The repository contains the source code, training pipeline, Kubernetes
manifests, network configuration, tests, evaluation outputs, CI/CD
workflow, and project documentation.

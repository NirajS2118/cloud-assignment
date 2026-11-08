# Cloud Computing (CS3081) Mini Project Report
## Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud

**Student Name:** Niraj Kumar  
**Roll Number:** 230101037  
**Course:** CS3081 Cloud Computing  
**Institution:** Indian Institute of Information Technology Senapati, Manipur  

---

## 1. Task 1: Object Storage and Parallel Teacher Labelling

### 1.1 Architecture & Local Kubernetes Setup
The project infrastructure was established on a multi-node local Kubernetes cluster deployed via `k3d` using configuration file `cluster/k3d.yaml`. The cluster consists of one dedicated control-plane server (`k3d-cloud-assignment-server-0`) and one agent worker node (`k3d-cloud-assignment-agent-0`), with ingress traffic routed through a built-in proxy load balancer mapping host port `8081` to container port `80` and host port `8444` to `443`.

Object storage was provisioned by deploying **SeaweedFS** inside a dedicated `data` namespace (`k8s/storage/seaweedfs.yaml`). SeaweedFS provides high-throughput blob storage with an S3-compatible REST API on port `8333` backed by a 5 GiB PersistentVolumeClaim. Two S3 buckets were initialized: `datasets` for command corpora and teacher shards, and `models` for student model weights and metadata. The raw 3,000 unlabelled voice commands (`data/commands_pool.jsonl`) were uploaded to `s3://datasets/commands_pool.jsonl`.

### 1.2 Teacher Model & Parallel Worker Sharding
The teacher model selected was `Phi-3.5-mini-instruct` (3.8 billion parameters), executed locally via Ollama with strict deterministic parameters (`temperature: 0`, `seed: 42`). The system prompt engineered in `prompts/teacher_prompt.txt` constrains the teacher to return strictly one valid JSON object containing exactly four keys: `intent`, `date`, `time`, and `place`, adhering to the eight defined intent classes without Markdown formatting or prose commentary.

To parallelize inference across multiple cores and avoid a single-worker bottleneck, two worker jobs (`teacher-worker-0` and `teacher-worker-1`) were executed concurrently inside the `pipeline` namespace. Each worker dynamically reads environment variables `SHARD_INDEX` and `SHARD_COUNT=2`, downloading `commands_pool.jsonl` from SeaweedFS and filtering line indices where `line_number % SHARD_COUNT == SHARD_INDEX`. Worker 0 processed exactly 1,500 even-indexed commands writing to `datasets/teacher/shard-0.jsonl`, while Worker 1 processed exactly 1,500 odd-indexed commands writing to `datasets/teacher/shard-1.jsonl`. Verification confirmed zero overlap between shards and complete coverage of all 3,000 command IDs with zero gaps.

### 1.3 Dataset Cleaning Pipeline & Checksum Accounting
The cleaning script `worker/clean.py` was implemented to process all shard outputs in natural ID order (`C0001` to `C3000`). Records were evaluated against the five mutually exclusive criteria defined in the specification:

| Category | Rejection Criterion / Action | Record Count | Percentage |
|:---|:---|:---:|:---:|
| **Total Input** | Raw records aggregated across shard-0 and shard-1 | **3,000** | 100.00% |
| **Reason (a)** | Replies that are not valid JSON | 75 | 2.50% |
| **Reason (b)** | Replies whose keys are not exactly `intent, date, time, place` | 4 | 0.13% |
| **Reason (c)** | Replies whose intent is not one of the eight valid intents | 82 | 2.73% |
| **Reason (d)** | Replies with slot values neither null nor exact command substrings | 359 | 11.97% |
| **Reason (e)** | Duplicate commands (retaining lowest ID among valid records) | 45 | 1.50% |
| **Records Kept** | Verified clean records written to `datasets/v1/train.jsonl` | **2,435** | 81.17% |

$$\text{Accounting Verification: } \text{Input} = (a) + (b) + (c) + (d) + \text{duplicates} + \text{kept}$$
$$3000 = 75 + 4 + 82 + 359 + 45 + 2435 \quad (\text{Exact Match: True})$$

**SHA-256 Checksum (`datasets/v1/train.sha256`):**
`ee52ae7ba8a69bf2903dc828f69c46e5271a5c28715a041b32964668d79a5daf`

---

## 2. Task 2: Training the Student (Sequence-Level Distillation)

### 2.1 Distillation Methodology & Architectural Reasoning
- **Sequence-Level Knowledge Distillation:** The student model (`EleutherAI/pythia-70m`) is trained via standard autoregressive cross-entropy loss directly on the teacher's discrete generated JSON sequence targets ($-\sum \log P(y_t \mid y_{<t}, x)$).
- **Logit-Based Distillation:** Minimizes KL divergence between softened token probability distributions ($\mathcal{D}_{\text{KL}}(\sigma(z_T/\tau) \parallel \sigma(z_S/\tau))$).
- **Why Logit-Based Distillation is Infeasible in this Project:**
  1. *Vocabulary Discrepancy:* `Phi-3.5-mini` uses a 32,064-token SentencePiece vocabulary, while `Pythia-70M` uses a 50,304-token BPE vocabulary. Token alignments and probability spaces differ fundamentally.
  2. *Black-Box API Serving:* Serving the teacher via Ollama exposes only decoded text output, hiding intermediate logits.
- **Limitation of Sequence-Level KD:** Exposure bias during autoregressive inference; the model receives no penalty or signal regarding token uncertainty, causing compounding errors once an erroneous token is generated.

### 2.2 Hyperparameter Configuration & Execution
Hyperparameters declared in `training/config.yaml`:
- Model: `EleutherAI/pythia-70m`
- Seed: `42` (deterministic)
- Learning Rate: `0.00005` (AdamW)
- Batch Size: `4`
- Epochs: `1`
- Max Sequence Length: `128`
- Libraries: Python 3.11.9, PyTorch 2.5.1, Transformers 4.46.3, Boto3 1.35.71.

Training executes via `make train`, and subset verification via `make train SUBSET=50`. Per-epoch loss recorded in `training/loss.csv` (average loss: `0.6519`). Artifacts uploaded to SeaweedFS under `s3://models/v1/` and `s3://models/v2/`.

### 2.3 Model Evaluation
Evaluation against the 100 gold validation commands in `data/validation.jsonl`:

| Metric | Teacher (`Phi-3.5-mini`) | Untrained (`Pythia-70M`) | Trained Student (`Pythia-70M`) |
|:---|:---:|:---:|:---:|
| **JSON Validity** | 94.0% | 0.0% | **99.0%** |
| **Intent Accuracy** | 75.0% | 0.0% | **53.0%** |
| **Date Accuracy** | 74.0% | 0.0% | **51.0%** |
| **Time Accuracy** | 80.0% | 0.0% | **80.0%** |
| **Place Accuracy** | 79.0% | 0.0% | **84.0%** |
| **Full Frame Accuracy** | 45.0% | 0.0% | **16.0%** |

---

## 3. Task 3: The Inference Service in a Container

### 3.1 API & Endpoint Architecture
FastAPI microservice in `service/app.py`:
- `POST /parse`: Returns HTTP 200 with 4-key JSON; injects `X-Model-Version` and `X-Served-By`.
- `GET /healthz`: Liveness probe returning HTTP 200.
- `GET /readyz`: Readiness probe returning HTTP 503 until model is downloaded and verified, then HTTP 200.

### 3.2 Dynamic Configuration & Fault Injection
Configured via `MODEL_URI`, `MODEL_VERSION`, `S3_ENDPOINT`, and `MAX_NEW_TOKENS`. S3 credentials injected strictly from Kubernetes Secret `inference-s3-credentials`. Output verified against `service/schema.json`; invalid outputs return HTTP 500 with `{"error": "invalid_model_output"}`. When `FAULT_INJECT=1`, output is replaced with `"not json"`, returning HTTP 500.

### 3.3 Resource Limits & Stress Testing
Built with pinned dependencies and CPU wheels (< 2 GB). Stress test of 50 sequential requests under 1 CPU / 1 GiB limits:
- Total Requests: 50 | Success: 50 | OOM-Kills: 0
- Peak Memory Observed: **493 MiB** (48.1% of 1 GiB limit). Evidence saved in `results/peak_memory.txt`.

---

## 4. Task 4: Kubernetes Deployment, Scaling, and Reliability

### 4.1 Topology & Probes
Deployed in namespace `app`: Deployment (2 replicas), ClusterIP Service (port 8000), Traefik Ingress, and HPA (min 1, max 2, 70% CPU target). Startup, liveness, and readiness probes configured.

### 4.2 Reliability Experiments
- **Rolling Update:** Zero downtime transition from `v1` to `v2` and rollback via `kubectl rollout undo` with **0.0% failure rate**. Verified return to `v1`.
- **Pod Recovery:** Deleted running pod under active traffic; Kubernetes recreated replacement pod with **0.0% failed requests** (33/33 successes).
- **Cold Start:** Evaluated over three runs: Run 1: 10.72s, Run 2: 10.73s, Run 3: 10.76s. **Median Cold Start: 10.73 seconds**. Log evidence of S3 download captured.
- **Load Balancing:** 4 concurrent users over Ingress: Pod 1 served 146 requests (56.15%), Pod 2 served 114 requests (43.85%), satisfying >= 30% per replica requirement.
- **Cluster Limitation:** Single physical machine using k3d virtualization; nodes share underlying CPU and memory.

---

## 5. Task 5: Subnetting and Network Access Control

### 5.1 VLSM Allocation (`10.20.0.0/16`)
Calculated using $\text{Usable Hosts} = 2^{32-\text{prefix}} - 2$:

| Subnet | Network Address | Prefix | Usable Host Range | Broadcast Address | Usable Hosts |
|:---|:---|:---:|:---|:---|:---:|
| **app** | `10.20.0.0` | `/22` | `10.20.0.1` – `10.20.3.254` | `10.20.3.255` | 1,022 |
| **public** | `10.20.4.0` | `/24` | `10.20.4.1` – `10.20.4.254` | `10.20.4.255` | 254 |
| **data** | `10.20.5.0` | `/25` | `10.20.5.1` – `10.20.5.126` | `10.20.5.127` | 126 |
| **management** | `10.20.5.128` | `/27` | `10.20.5.129` – `10.20.5.158` | `10.20.5.159` | 30 |

### 5.2 Stateless NACL Theory
- **Reply Traffic:** Stateless NACLs do not track connection states; outbound return packets to client ephemeral ports (1024–65535) require explicit allow rules.
- **Rule Order:** Evaluated first-match-wins by ascending rule number. Malicious range `198.51.100.0/24` is placed at **Rule 90** before general allow rules (100 and 110) to prevent broad rules from admitting blacklisted traffic.
- **Simulation:** Passed 6/6 test packets (100.0% accuracy) using `tools/nacl_sim.py`.

---

## 6. Task 6: Reproducibility, Testing, and CI/CD

- **One-Command Setup:** `make up` creates the cluster and deploys all components in <= 3 commands; `make down` deletes the cluster.
- **CI/CD Workflow (`.github/workflows/ci.yml`):**
  1. Linting (`flake8`, `ruff`)
  2. Unit Tests (5 tests in `tests/test_service.py`)
  3. Model Evaluation on `data/validation.jsonl`
  4. Quality Gate (`score.py --min-validity 95`)
  5. Secret Scanning (`gitleaks`)
- **Quality Gate Demonstration:** Untrained Pythia-70M fails quality gate with 0.0% validity, correctly blocking pipeline.
  - **Failed Run (Quality Gate Rejection):** `https://github.com/NirajS2118/cloud-assignment/actions/runs/42489088090`
  - **Successful Production Run (Student Passed):** `https://github.com/NirajS2118/cloud-assignment/actions/runs/42488729084`
  - **GitHub Release v1 Asset:** `https://github.com/NirajS2118/cloud-assignment/releases/tag/v1`
- **AI Use:** Fully disclosed and documented in `AI_USE.md`.

---

## 7. Task 7: Performance, Cost, and Review

### 7.1 Measured Telemetry
- Median Latency (p50): **740 ms**
- 95th Percentile Latency (p95): **900 ms** (<= 5,000 ms)
- Throughput: **5.13 req/s** (~18,468 req/hour)
- Error Rate: **0.0% infrastructure errors**
- Pod Resources: Pod 1 (124m CPU, 493 MiB RAM), Pod 2 (2m CPU, 466 MiB RAM)
- Cold-Start Time: **10.73 s median**

### 7.2 Economic Cost Estimation
Using `data/rate_card.json` (₹3.50/vCPU-hr, ₹0.45/GiB-hr; limits: 1 vCPU, 1 GiB):
$$\text{Cost per replica/hr} = 1 \times 3.50 + 1 \times 0.45 = ₹3.95/\text{hr}$$
$$\text{Total hourly cost (2 replicas)} = 2 \times 3.95 = ₹7.90/\text{hr}$$
$$\text{Requests per hour} = 5.13 \times 3600 = 18,468\text{ req/hr}$$
$$\text{Cost per 1,000 requests} = \left(\frac{₹7.90}{18,468}\right) \times 1000 = \mathbf{₹0.4278 \approx ₹0.43}$$

### 7.3 SLO & Error Budget
- **SLO:** 99.0% of requests succeed with HTTP 200 within 5,000 ms.
- **Window:** One Locust run (302 requests).
- **Error Budget:** $302 \times 0.01 = \mathbf{3\text{ requests}}$.

### 7.4 Architectural Review Table

| Component | Evidence (Measured Number) | Limitation Found | Proposed Improvement |
|:---|:---|:---|:---|
| **Data and Storage** | 2,435 clean records kept from 3,000; 359 rejected for non-substring slots | Teacher prompt occasionally rephrased slot values. | Enforce grammar-constrained decoding (Outlines/Guidance). |
| **Model Quality** | 99.0% JSON validity, 16.0% full frame accuracy (vs 0% base) | 70M parameter capacity struggles with subtle date reasoning. | Fine-tune 1B parameter quantized SLM (Llama-3.2-1B) with LoRA. |
| **Latency & Throughput** | 740 ms p50, 900 ms p95, 5.13 req/s | CPU FP32 token generation limits throughput scalability. | Export to ONNX Runtime with INT8 quantization and AVX-512. |
| **Resource Utilisation**| 493 MiB peak memory per pod against 1 GiB limit | PyTorch baseline runtime overhead consumes ~450 MiB at idle. | Transition to C++ inference runtime (`llama.cpp` / GGML) (< 100 MiB). |
| **Reliability & Recovery**| 0.0% failure rate during recovery; 10.73 s median cold-start | Pods pull 281 MB weights over cluster network on startup. | Implement ReadWriteMany local volume cache or pre-baked images. |
| **Security and Cost** | 100% NACL pass rate; ₹0.43 estimated cost per 1,000 requests | Replicas billed continuously (₹7.90/hr) even during idle periods. | Introduce KEDA or Knative for serverless scale-to-zero. |


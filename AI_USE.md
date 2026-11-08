# AI Use and Verification Disclosure (`AI_USE.md`)

This document discloses the artificial intelligence tools, models, and assistants utilized during the completion of this Cloud Computing Mini Project (CS3081), along with the specific tasks they assisted with and the rigorous verification procedures applied to all generated artifacts.

---

## 1. Tools and Models Used

| Tool / Model | Type | Version / Host | Role in Project |
|:---|:---|:---|:---|
| **Phi-3.5-mini-instruct** | Large Language Model (Teacher) | 3.8B Parameters (Ollama 0.40.15) | Sequence-level distillation teacher; generated structured intent/slot targets for 3,000 commands. |
| **EleutherAI/pythia-70m** | Lightweight Causal LM (Student) | 70M Parameters (PyTorch 2.5 / Transformers) | Fine-tuned student model deployed as a lightweight inference microservice on Kubernetes. |
| **Google Antigravity / Gemini** | AI Coding Assistant | Gemini 3.8 Flash / Pro | Assisted with architecture planning, Kubernetes manifest authoring, FastAPI service implementation, unit test authoring, and documentation. |

---

## 2. Tasks and Breakdown of AI Usage

### 2.1 Teacher Dataset Labelling (Task 1)
- **Role:** `Phi-3.5-mini-instruct` was used strictly as the teacher model specified in Section 6.
- **Execution:** Run locally with Ollama at temperature `0` and seed `42`.
- **System Prompt:** Engineered in `prompts/teacher_prompt.txt` to enforce the strict 4-key JSON structure (`intent`, `date`, `time`, `place`).

### 2.2 Kubernetes Manifests and Service Architecture (Tasks 1, 3, 4)
- **Role:** AI assistant assisted in drafting declarative Kubernetes manifests for SeaweedFS PVC/Deployments, the `command-parser` service, Traefik Ingress routing, and HorizontalPodAutoscaler (HPA v2).
- **Service Logic:** Assisted in structuring FastAPI startup lifespan handlers (`boto3` model download from SeaweedFS S3 bucket, checksum validation against `model.sha256`, and graceful readiness probe gates).

### 2.3 Network ACL Simulation and Subnet Allocation (Task 5)
- **Role:** Assisted in verifying binary prefix math for VLSM (Variable Length Subnet Masking) from `10.20.0.0/16` and drafting stateless NACL rules according to AWS VPC semantics.

### 2.4 Testing, CI/CD, and Load Testing (Tasks 6 & 7)
- **Role:** Assisted in authoring GitHub Actions workflow stages (`lint`, `unit-tests`, `model-evaluation`, `quality-gate`, `secret-scan`), unit test cases in `tests/test_service.py`, and Locust load test scripts.

---

## 3. Verification and Validation Procedures

All code, configurations, and data produced with AI assistance were subjected to automated, deterministic checks:

1. **Teacher Output Validation & Cleaning:**
   - Teacher outputs were cleaned using `worker/clean.py` against five deterministic criteria: valid JSON syntax, exact required keys, valid intent enum membership, exact substring matching for slot values, and deduplication.
   - Rejection counts were strictly tracked: 75 invalid JSON, 4 invalid keys, 82 invalid intents, 359 invalid slot values, 45 duplicates, yielding 2,435 verified clean records ($75 + 4 + 82 + 359 + 45 + 2435 = 3000$).
   - The final training file SHA-256 checksum (`ee52ae7ba8a69bf2903dc828f69c46e5271a5c28715a041b32964668d79a5daf`) was recorded and verified.

2. **Automated Unit & Fault Injection Testing:**
   - 5 independent unit tests in `tests/test_service.py` executed via Python's standard `unittest` framework to verify health probes, readiness state transitions, header injection, and schema validation.
   - Fault injection behavior was tested with `FAULT_INJECT=1` to guarantee HTTP 500 error propagation upon malformed model outputs.

3. **NACL Rule Verification:**
   - Evaluated against the test harness `tools/nacl_sim.py` using `network/packets_sample.jsonl`, achieving 100.0% accuracy (6/6 packets correctly permitted/blocked).

4. **Cluster Performance and Load Testing:**
   - Ran 50 sequential requests to verify memory stability under 1 CPU and 1 GiB limits (peak memory observed: 493 MiB, no OOM-kill).
   - Executed Locust load testing with 4 concurrent users over Traefik Ingress to confirm load balancing (43.8% vs 56.2% split across pods, satisfying the >= 30% requirement).
   - Validated zero downtime and >= 99.5% success rate during rolling update (`v1` -> `v2` -> `kubectl rollout undo` rollback).

5. **Secret Scanning:**
   - Scanned the entire repository to ensure no API keys, private credentials, or plaintext secrets were committed. S3 credentials are dynamically provided via Kubernetes `Secret` objects.


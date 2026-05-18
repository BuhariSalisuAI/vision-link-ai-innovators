# Vision-Link AI Innovators 🏥🏢

> **Enterprise-Grade Autonomous AI Platform**
> Built for the [Lablab.ai Enterprise Hackathon](https://lablab.ai)
> Enabling smart data extraction, automated clinical analysis, and secure multilingual workflow processing.

---

## 📋 Table of Contents

- [Platform Overview](#platform-overview)
- [Architecture](#architecture)
- [Core Capabilities](#core-capabilities)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Smoke Testing](#smoke-testing)
- [Self-Healing Loop](#self-healing-loop)
- [Self-Evolving Backend](#self-evolving-backend)
- [Models & LLM Strategy](#models--llm-strategy)
- [Project Structure](#project-structure)
- [CI / CD](#ci--cd)
- [Team](#team)

---

## Platform Overview

Vision-Link AI Innovators is the enterprise extension of the Vision-Link AI Agent system. Where the AI Agent track focuses on autonomous multi-agent orchestration, this platform is designed for enterprise deployment — with emphasis on security, scalability, audit trails, and integration with existing healthcare infrastructure.

---

## Architecture

The Vision-Link AI Innovators platform is a **multi-node LangGraph orchestration engine** that coordinates five specialized CrewAI agents through an API gateway layer, with enterprise-grade output in JSON, HL7, and PDF formats.

```
Enterprise Client Request
          │
          ▼
┌──────────────────────┐
│   API Gateway Layer  │
│   (Auth + Rate Limit)│
└────────┬─────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                LangGraph Orchestration Engine                │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Data          │  │ Clinical     │  │ Multi-           │  │
│  │ Extraction   │─▶│ Analysis     │─▶│ lingual Agent    │  │
│  │ Agent        │  │ Agent        │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌───────────────────┐                   │
│  │ Validation   │  │ Audit &           │                   │
│  │ Agent        │─▶│ Compliance Agent  │                   │
│  └──────────────┘  └───────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│   Enterprise Output  │
│   (JSON / HL7 / PDF) │
└──────────────────────┘
```

### Data Flow

1. **API Gateway** → Authenticates request and enforces rate limits
2. **Data Extraction Agent** → Ingests and parses unstructured clinical data using Llama-3-8B
3. **Clinical Analysis Agent** → Performs differential diagnosis and urgency triage using MedLlama-3
4. **Multilingual Agent** → Localizes outputs across 40+ languages with region-specific terminology
5. **Validation Agent** → Validates output integrity, schemas, and confidence scores
   - **FAIL** → Routes to Self-Healing Loop (retry failed agent, max 3 attempts)
   - **PASS** → Routes to Self-Evolving Backend
6. **Audit & Compliance Agent** → Logs every agent decision; enforces HIPAA-aligned patterns
7. **Enterprise Output** → Returns structured response in JSON, HL7, or PDF

---

## Core Capabilities

### 1. Smart Data Extraction

Automated ingestion and parsing of unstructured clinical data — patient records, lab reports, referral letters — into structured, validated schemas using Llama-3-8B-Instruct.

### 2. Automated Clinical Analysis

Multi-agent pipeline performs differential diagnosis, urgency triage, and treatment recommendation with confidence scoring and audit trails at every step.

### 3. Secure Multilingual Workflow Processing

End-to-end language localization for clinical outputs across 40+ languages, with region-specific medical terminology validation and compliance checks.

### 4. Enterprise Security Layer

- All tokens managed via environment secrets — never in code
- Full audit log on every agent decision
- Role-based access control ready
- HIPAA-aligned data handling patterns

### 5. Self-Healing Loop

When the Validation Agent detects errors or missing fields:

1. **Identifies the failure source** — determines which agent produced the error
2. **Targeted re-routing** — re-executes **only that agent**, not the entire pipeline
3. **Retry mechanism** — attempts up to `MAX_RETRIES` (default: 3) with exponential backoff
4. **LLM-assisted patching** — for soft errors (e.g., low confidence scores), uses intelligent prompt engineering to refine outputs
5. **Graceful degradation** — marks run as `failed` only after all retries are exhausted; provides best-effort fallback

**Benefit:** Reduces latency by avoiding redundant pipeline re-runs while maximizing recovery chances.

### 6. Self-Evolving Backend

Every successful validation triggers an autonomous evolution cycle:

1. **Variant generation** — Concurrently generates **3 LLM-optimized code variants** of the orchestration pipeline
2. **Isolated sandboxing** — Each variant compiles and runs in a separate, containerized execution context
3. **Fitness scoring** — Evaluates each variant using latency, error rate, and throughput
4. **Atomic hot-swap** — Winner automatically replaces the running module via Python hot-reload (no process restarts, no downtime)
5. **Zero human intervention** — Fully autonomous; decisions logged but require no manual approval

**Benefit:** Continuous performance optimization without operational overhead.

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **HuggingFace account** with valid API token
- **Git** for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/BuhariSalisuAI/vision-link-ai-innovators.git
cd vision-link-ai-innovators
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy example .env file
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_your_token_here
```

### 4. Run Smoke Test

```bash
cd workspace
python orchestrator.py
```

> **First-run tip:** Set `EVOLUTION_ENABLED=false` in your `.env` to skip the self-evolving backend and focus on core pipeline validation.

---

## Environment Setup

### Required Variables

| Variable | Required | Type | Source | Example |
|----------|----------|------|--------|---------|
| `HF_TOKEN` | ✅ Yes | String | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `hf_a1B2c3D4e5F6g7H8i9J0k...` |

### Optional Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EVOLUTION_ENABLED` | `true` | Enable/disable self-evolving backend (`true` or `false`) |
| `MAX_RETRIES` | `3` | Maximum retry attempts in self-healing loop |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`INFO` / `DEBUG` / `ERROR`) |

### Example `.env` File

```dotenv
# HuggingFace API Token (required)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Feature flags
EVOLUTION_ENABLED=false

# Retry and performance tuning
MAX_RETRIES=3

# Logging
LOG_LEVEL=INFO
```

> **Security Note:** Never commit `.env` to version control. Use [GitHub Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets) for CI/CD.

---

## Smoke Testing

The smoke test validates the core pipeline without triggering the self-evolving backend:

### Basic Smoke Test

```bash
cd workspace
python orchestrator.py
```

### Expected Output

```json
{
  "run_id": "...",
  "patient_id": "TEST-001",
  "diagnosis": ["..."],
  "urgency": "high",
  "actions": ["..."],
  "confidence": 0.87,
  "validation_passed": true,
  "warnings": [],
  "evolution": {},
  "error_log": [],
  "stage": "finalized"
}
```

### What a Passing Smoke Test Looks Like

- `validation_passed` is `true`
- `stage` is `"finalized"`
- `error_log` is empty `[]`
- `diagnosis` contains at least one candidate
- `urgency` is one of: `low`, `medium`, `high`, `critical`

### Smoke Test with Evolution (Optional)

To test the full pipeline including self-evolving backend:

```bash
cd workspace
EVOLUTION_ENABLED=true python orchestrator.py
```

### Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `EnvironmentError: HF_TOKEN is not set` | Missing `.env` file or invalid token | Add `HF_TOKEN=your_token` to `.env` |
| `403 Forbidden from HuggingFace` | Token lacks read permissions | Regenerate token with Read role at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `ModuleNotFoundError` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `ValidationError on PipelineState` | Schema mismatch | Check `state_schema.py` matches agent outputs |
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch size or set `device="cpu"` in `model_loader.py` |

---

## Self-Healing Loop

### How It Works

```
Validation Failed
    │
    ├─→ Identify Failed Node (Extraction, Clinical, or Multilingual)
    │
    ├─→ Retry Loop (Max 3 attempts):
    │   ├─→ Attempt 1: Re-run agent with same input
    │   ├─→ Attempt 2: Inject corrective prompt guidance
    │   └─→ Attempt 3: Combine outputs with LLM merging
    │
    ├─→ Soft Error Detection:
    │   ├─→ Low confidence? → LLM-based refinement
    │   ├─→ Missing fields? → Intelligent imputation
    │   └─→ Ambiguous output? → Prompt clarification
    │
    └─→ If still failing → Mark failed, return partial result
```

### Configuration

In `workspace/orchestrator.py`:

```python
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
```

### Example Scenario

**Input:** Patient with incomplete symptom data
**Clinical Agent Output:** `{"diagnosis": "...", "confidence": 0.42}`
**Validation:** Confidence below threshold → FAIL

**Self-Healing:**
1. Retry 1: Re-run Clinical Agent → Still low confidence
2. Retry 2: Inject prompt: *"Provide the most likely diagnosis with supporting evidence"* → Confidence improves to 0.68
3. Validation: ✓ PASS → Proceed to self-evolving backend

---

## Self-Evolving Backend

### Evolution Cycle

```
Pipeline Validation PASSED
    │
    ├─→ Generate 3 Variants:
    │   ├─→ Variant A: Optimized for latency (streaming outputs)
    │   ├─→ Variant B: Optimized for accuracy (ensemble voting)
    │   └─→ Variant C: Optimized for throughput (batch processing)
    │
    ├─→ Compile & Sandbox:
    │   ├─→ Each variant spun up in isolated Python process
    │   ├─→ Syntax/logic validation performed
    │   └─→ Ready for execution
    │
    ├─→ Race Execution (on same test dataset):
    │   ├─→ Variant A: 1200ms, 2 errors, 50 req/sec
    │   ├─→ Variant B: 1800ms, 0 errors, 25 req/sec
    │   └─→ Variant C: 950ms, 1 error, 100 req/sec
    │
    ├─→ Fitness Scoring:
    │   └─→ Winner selected by highest overall fitness score
    │
    └─→ Atomic Hot-Swap:
        ├─→ Replace orchestrator.py with winning variant
        ├─→ Reload modules in-process
        └─→ No restarts, no downtime
```

### Fitness Formula

```
fitness_score = (1000 / avg_latency_ms)
              + (100 × accuracy_rate)
              − (5 × error_count)
              + (throughput_requests_per_sec / 10)
```

Minimum threshold to replace current: **FITNESS_THRESHOLD** (default: `0.75`)

### Evolution Configuration

```dotenv
EVOLUTION_ENABLED=true   # Set to false to disable
```

---

## Models & LLM Strategy

| Agent | Model | Rationale |
|-------|-------|-----------|
| **Data Extraction** | `meta-llama/Meta-Llama-3-8B-Instruct` | Strong structured parsing; efficient context understanding |
| **Clinical Analysis** | `ProbeMedicalYonseiMAILab/medllama3-v20` | Specialized medical knowledge; highest diagnostic accuracy via domain fine-tuning |
| **Multilingual** | `meta-llama/Meta-Llama-3-8B-Instruct` | Strong multilingual support; region-specific terminology |
| **Validation** | `meta-llama/Meta-Llama-3-8B-Instruct` | Reliable structured output generation; excellent for schema validation |
| **Self-Healing Engine** | `meta-llama/Meta-Llama-3-8B-Instruct` | Balanced reasoning capability + inference speed; effective at error analysis |
| **Fallback** | `mistralai/Mistral-7B-Instruct-v0.3` | Lightweight alternative; used if primary models unavailable |

---

## Project Structure

```
vision-link-ai-innovators/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD pipeline
├── workspace/
│   ├── state_schema.py               # Pydantic v2 state definitions
│   ├── orchestrator.py               # LangGraph orchestration engine
│   ├── model_loader.py               # HuggingFace model initialization
│   └── crew_agents.py                # CrewAI agent definitions
├── .env.example                       # Environment template (copy to .env)
├── .gitignore                         # Git ignore patterns
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Dependencies

Key packages and versions:

```
langgraph>=0.2.28              # LangGraph orchestration framework
langchain-core>=0.3.0          # LangChain core utilities
crewai>=0.80.0                 # CrewAI multi-agent framework
langchain-huggingface>=0.1.0   # HuggingFace LLM integration
huggingface-hub>=0.23.0        # HuggingFace Hub API
transformers>=4.44.0           # Transformer models library
pydantic>=2.7.0                # Data validation (v2)
python-dotenv>=1.0.0           # Environment variable management
```

See `requirements.txt` for complete list.

---

## CI / CD

GitHub Actions workflows run on every push to `main` and `dev` branches:

**Checks performed:**

- **Ruff linting** — Code style validation
- **Pydantic schema validation** — State schema integrity
- **LangGraph compilation check** — Graph topology validation
- **HuggingFace model loader health check** — Model availability verification

**Workflow file:** `.github/workflows/ci.yml`

Add `HF_TOKEN` as a repository secret at:
**Settings → Secrets and variables → Actions → New repository secret**

To view CI status: [Vision-Link AI Innovators — Actions](https://github.com/BuhariSalisuAI/vision-link-ai-innovators/actions)

---

## Team

| Member | GitHub | Role |
|--------|--------|------|
| **Buhari Salisu** | [@BuhariSalisuAI](https://github.com/BuhariSalisuAI) | Founder & CEO |
| **Subhalaxmi** | [@Subhalaxmi](https://github.com/Subhalaxmi) | Data Engineer — CrewAI Agents & Tasks |
| **Zentomo** | [@Zentomo](https://github.com/Zentomo) | Systems Engineer — LangGraph Orchestration & Infrastructure |
| **Ramitha** | [@Ramitha](https://github.com/Rami2212) | Documentation |
| **Meghana** | [@Meghana](https://github.com/Meghana) | TBD |

---

**Vision-Link AI Innovators** · AI Agent Olympics — Enterprise Track · Lablab.ai · May 2026

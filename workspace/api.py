"""
vision-link-ai-innovators / workspace / api.py

FastAPI Enterprise Endpoint
===========================
Wraps the LangGraph pipeline in a production-ready REST API.

Endpoints:
  POST /analyze       — run full pipeline on a patient record
  GET  /health        — pipeline + model health check
  GET  /metrics       — last evolution metrics

Run locally:
  uvicorn workspace.api:app --reload --port 8000

Example request:
  curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"patient_id":"P001","age":45,"gender":"female",
         "symptoms":["chest pain"],"language":"en"}'
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_loader import health_check
from orchestrator import run_pipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Vision-Link AI Innovators — Enterprise API",
    description=(
        "Multi-Agent Autonomous Healthcare Assistant. "
        "Powered by LangGraph + CrewAI + Llama-3-8B-Instruct."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PatientRequest(BaseModel):
    patient_id: str
    age: int = Field(ge=0, le=130)
    gender: str
    symptoms: List[str] = Field(min_length=1)
    medical_history: List[str] = []
    location: Optional[str] = None
    language: str = "en"
    evolution_enabled: bool = False     # default off for API — faster response


class PipelineResponse(BaseModel):
    run_id: str
    patient_id: str
    diagnosis: List[str]
    urgency: str
    actions: List[str]
    confidence: float
    validation_passed: bool
    warnings: List[str]
    evolution: Dict[str, Any]
    error_log: List[str]
    stage: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    pipeline: str
    hf_endpoint: str


class MetricsResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {"name": "Vision-Link AI Innovators", "status": "running", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Check pipeline and HuggingFace model endpoint health."""
    hf_status = health_check("primary")
    return HealthResponse(
        status="ok" if hf_status["status"] == "ok" else "degraded",
        model=hf_status.get("model", "unknown"),
        pipeline="compiled",
        hf_endpoint=hf_status["status"],
    )


@app.post("/analyze", response_model=PipelineResponse, tags=["Pipeline"])
def analyze_patient(request: PatientRequest):
    """
    Run the full multi-agent pipeline on a patient record.

    Returns structured diagnosis, urgency level, recommended actions,
    and full audit trail from the LangGraph orchestration engine.
    """
    t0 = time.perf_counter()
    try:
        result = run_pipeline(
            patient_data=request.model_dump(
                exclude={"evolution_enabled"}
            ),
            evolution_enabled=request.evolution_enabled,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")

    latency = (time.perf_counter() - t0) * 1000

    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline returned no result.")

    return PipelineResponse(**result, latency_ms=round(latency, 2))


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
def get_metrics():
    """Returns evolution engine metrics from the last pipeline run."""
    return MetricsResponse(
        message="Metrics are embedded in each /analyze response under the 'evolution' key."
    )

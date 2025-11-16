#!/usr/bin/env bash
set -euo pipefail

echo "Scaffolding rah-voice-agent..."

# --- Directories ---
mkdir -p apps/agent/tools
mkdir -p apps/agent/guards
mkdir -p apps/agent/routing
mkdir -p apps/agent/rag_clients
mkdir -p apps/agent/cfg

mkdir -p services/wellsky_adapter
mkdir -p services/scheduler
mkdir -p services/notifier/providers

mkdir -p packages/common
mkdir -p packages/clients

mkdir -p data/kb_company
mkdir -p data/kb_scenarios
mkdir -p data/kb_examples

mkdir -p offline/analyzers
mkdir -p offline/datasets
mkdir -p offline/repro_harness
mkdir -p offline/reports

mkdir -p storage/sql
mkdir -p policy
mkdir -p config

mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/e2e_repro

# --- Top-level files ---

cat > README.md << 'EOF'
# RAH Voice Agent (Production-Oriented Skeleton)

This repo is structured for a production-style AI voice/phone agent for a Right at Home–style franchise.

Key pieces:
- apps/agent: LiveKit + LLM agent, tools, guards, RAG, routing
- services/*: domain microservices (WellSky adapter, scheduler, notifier)
- packages/*: shared libraries (logging, audit, http clients)
- data/*: company/scenario/example knowledge bases (no PHI)
- offline/*: post-call analysis, datasets, repro harness
- storage/*: SQL schemas for analytics + audit logs

Quick start (after you fill code in):
1. cp config/env.example .env
2. docker compose -f config/compose.yaml up --build
3. Connect a LiveKit web client to LIVEKIT_URL and test the agent.
EOF

cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rah-voice-agent"
version = "0.1.0"
description = "Production-style AI voice agent for Right at Home scheduling"
requires-python = ">=3.11"
EOF

cat > Makefile << 'EOF'
run:
	docker compose -f config/compose.yaml up --build

test:
	pytest -q

repro CALL_ID?=latest
repro:
	python -m offline.repro_harness.run_repro --call-id $(CALL_ID)
EOF

cat > .cursorrules << 'EOF'
{
  "ignore": [
    "data/**",
    "offline/reports/**",
    "storage/sql/**"
  ],
  "test": {
    "command": "pytest -q"
  }
}
EOF

# --- config ---

cat > config/env.example << 'EOF'
LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=lk_key
LIVEKIT_API_SECRET=lk_secret
OPENAI_API_KEY=sk-...

WELLSKY_ADAPTER_URL=http://wellsky-adapter:7001
SCHEDULER_URL=http://scheduler:7002
NOTIFIER_URL=http://notifier:7003

REALTIME_MODEL=gpt-4o-realtime-preview
INTENT_MODEL=gpt-4o-mini
STRICT_JUDGE_MODEL=gpt-5-thinking

TRANSCRIPT_RETENTION_DAYS=90
MASK_NAMES=true
EOF

cat > config/compose.yaml << 'EOF'
version: "3.9"
services:
  agent:
    build: ./apps/agent
    env_file: .env
    environment:
      - WELLSKY_ADAPTER_URL=${WELLSKY_ADAPTER_URL}
      - SCHEDULER_URL=${SCHEDULER_URL}
      - NOTIFIER_URL=${NOTIFIER_URL}
    depends_on:
      - wellsky-adapter
      - scheduler
      - notifier

  wellsky-adapter:
    build: ./services/wellsky_adapter
    ports:
      - "7001:7001"

  scheduler:
    build: ./services/scheduler
    ports:
      - "7002:7002"

  notifier:
    build: ./services/notifier
    ports:
      - "7003:7003"
EOF

# --- apps/agent ---

cat > apps/agent/requirements.txt << 'EOF'
livekit-agents
livekit-plugins-openai
httpx
EOF

cat > apps/agent/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "agent.py"]
EOF

cat > apps/agent/agent.py << 'EOF'
import os
from typing import Dict, Any
from livekit.agents import Agent, JobContext, cli
from livekit.plugins.openai import RealtimeModel

from tools.scheduling import tool_report_callout, tool_reschedule_shift
from tools.wellsky import tool_check_visit_status
from tools.notify import tool_notify

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are an after-hours assistant for a Right at Home franchise. Keep clients safe, escalate when unsure, and obey HIPAA-style guardrails.",
)

TOOLS = {
    "check_visit_status": tool_check_visit_status,
    "report_callout": tool_report_callout,
    "reschedule_shift": tool_reschedule_shift,
    "notify_parties": tool_notify,
}

async def entrypoint(ctx: JobContext):
    model = RealtimeModel(
        model=os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview"),
        api_key=os.environ["OPENAI_API_KEY"],
    )

    agent = Agent(
        llm=model,
        instructions=SYSTEM_PROMPT,
        tools=TOOLS,
        # TODO: later: guards + RAG + routing integration
    )
    await ctx.connect_agent(agent)

if __name__ == "__main__":
    cli.run_app(entrypoint)
EOF

cat > apps/agent/tools/scheduling.py << 'EOF'
from typing import Dict, Any
import os, httpx

SCHEDULER_URL = os.environ.get("SCHEDULER_URL", "http://scheduler:7002")
WELLSKY_ADAPTER_URL = os.environ.get("WELLSKY_ADAPTER_URL", "http://wellsky-adapter:7001")

async def tool_report_callout(args: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: call wellsky-adapter to mark callout, scheduler to rank, notifier to alert
    return {"status": "stub", "args": args}

async def tool_reschedule_shift(args: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: call wellsky-adapter to reschedule visit
    return {"status": "stub", "args": args}
EOF

cat > apps/agent/tools/wellsky.py << 'EOF'
from typing import Dict, Any
import os, httpx

WELLSKY_ADAPTER_URL = os.environ.get("WELLSKY_ADAPTER_URL", "http://wellsky-adapter:7001")

async def tool_check_visit_status(args: Dict[str, Any]) -> Dict[str, Any]:
    visit_id = args["visit_id"]
    async with httpx.AsyncClient(base_url=WELLSKY_ADAPTER_URL, timeout=10) as client:
        r = await client.get(f"/visits/{visit_id}/status")
        r.raise_for_status()
        data = r.json()
    # TODO: mask PHI before returning to LLM
    return data
EOF

cat > apps/agent/tools/notify.py << 'EOF'
from typing import Dict, Any
import os, httpx

NOTIFIER_URL = os.environ.get("NOTIFIER_URL", "http://notifier:7003")

async def tool_notify(args: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=NOTIFIER_URL, timeout=10) as client:
        r = await client.post("/notify", json=args)
        r.raise_for_status()
    return {"ok": True}
EOF

cat > apps/agent/guards/injection_guard.py << 'EOF'
def is_injection(text: str) -> bool:
    # TODO: cheap heuristic or tiny LLM call
    return False
EOF

cat > apps/agent/guards/pii_guard_lite.py << 'EOF'
def redact_pii(text: str) -> str:
    # TODO: mask names / phone numbers in online path
    return text
EOF

cat > apps/agent/guards/hallucination_lite.py << 'EOF'
def looks_inconsistent_with_kb(response: str, kb_snippets: str) -> bool:
    # TODO: minimal consistency check vs fetched KB
    return False
EOF

cat > apps/agent/routing/intent_classifier.py << 'EOF'
from typing import Literal

Intent = Literal["callout", "eta_query", "cancel_or_resched", "smalltalk", "unknown"]

def classify_intent(text: str) -> Intent:
    # TODO: call a small LLM or use regex-based bootstrap
    return "unknown"
EOF

cat > apps/agent/routing/prompt_renderer.py << 'EOF'
from typing import List, Dict, Any

def render_prompt(system_prompt: str,
                  conversation: List[Dict[str, str]],
                  kb_company: str,
                  kb_scenarios: str,
                  kb_examples: str) -> List[Dict[str, str]]:
    # TODO: actually merge RAG chunks + examples into messages
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation,
    ]
    return messages
EOF

cat > apps/agent/routing/tool_policy.py << 'EOF'
from typing import List

def allowed_tools_for_intent(intent: str) -> List[str]:
    # TODO: restrict tools by intent (safety + clarity)
    return ["check_visit_status", "report_callout", "reschedule_shift", "notify_parties"]
EOF

cat > apps/agent/rag_clients/company_kb_client.py << 'EOF'
def get_company_kb_snippets(intent: str) -> str:
    # TODO: load from data/kb_company; later use a vector DB
    return ""
EOF

cat > apps/agent/rag_clients/scenario_kb_client.py << 'EOF'
def get_scenario_snippets(intent: str) -> str:
    # TODO: scenario rules: if caregiver calls out within X hours, do Y
    return ""
EOF

cat > apps/agent/rag_clients/examples_kb_client.py << 'EOF'
def get_example_snippets(intent: str) -> str:
    # TODO: golden few-shot examples for this intent
    return ""
EOF

cat > apps/agent/cfg/agent.yaml << 'EOF'
# Agent config (models, thresholds, timeouts)
model: gpt-4o-realtime-preview
intent_model: gpt-4o-mini
max_latency_ms: 1500
EOF

# --- services: wellsky_adapter ---

cat > services/wellsky_adapter/requirements.txt << 'EOF'
fastapi
uvicorn
pydantic
httpx
EOF

cat > services/wellsky_adapter/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001"]
EOF

cat > services/wellsky_adapter/main.py << 'EOF'
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="wellsky-adapter (stub)")

VISITS = {
  "v_100": {"status": "scheduled", "eta": "08:10 AM", "caregiver_name": "Alice Brown", "client_id": "cli_123"}
}

class CalloutReq(BaseModel):
    caregiver_id: str
    reason: str

class ReassignReq(BaseModel):
    caregiver_id: str

@app.get("/visits/{visit_id}/status")
def status(visit_id: str):
    v = VISITS.get(visit_id)
    if not v:
        raise HTTPException(404, "visit not found")
    return v

@app.post("/visits/{visit_id}/callout")
def callout(visit_id: str, body: CalloutReq):
    v = VISITS.get(visit_id)
    if not v:
        raise HTTPException(404, "visit not found")
    v["status"] = "open"
    return {"ok": True}

@app.post("/visits/{visit_id}/reassign")
def reassign(visit_id: str, body: ReassignReq):
    v = VISITS.get(visit_id)
    if not v:
        raise HTTPException(404, "visit not found")
    v["status"] = "scheduled"
    v["caregiver_name"] = f"CG {body.caregiver_id}"
    return {"ok": True, "visit_id": visit_id}
EOF

# --- services: scheduler ---

cat > services/scheduler/requirements.txt << 'EOF'
fastapi
uvicorn
pydantic
EOF

cat > services/scheduler/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7002"]
EOF

cat > services/scheduler/constraints.py << 'EOF'
def apply_constraints(visit_id, candidates):
    # TODO: enforce skills, continuity of care, OT, distance, DND
    return candidates
EOF

cat > services/scheduler/scoring.py << 'EOF'
def score_candidates(visit_id, candidates):
    # TODO: add weights + (later) ML predictions (acceptance/no-show)
    return candidates
EOF

cat > services/scheduler/main.py << 'EOF'
from fastapi import FastAPI
from typing import Dict
from .constraints import apply_constraints
from .scoring import score_candidates

app = FastAPI(title="scheduler (stub)")

@app.get("/rank")
def rank(visit_id: str) -> Dict:
    candidates = [
        {"caregiver_id":"cg_201","name":"Beth Chan"},
        {"caregiver_id":"cg_305","name":"Dan Fox"},
        {"caregiver_id":"cg_122","name":"E. Lee"},
    ]
    feasible = apply_constraints(visit_id, candidates)
    ranked = score_candidates(visit_id, feasible)
    return {"visit_id": visit_id, "top": ranked}
EOF

# --- services: notifier ---

cat > services/notifier/requirements.txt << 'EOF'
fastapi
uvicorn
pydantic
EOF

cat > services/notifier/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7003"]
EOF

cat > services/notifier/main.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="notifier (stub)")

class Notify(BaseModel):
    visit_id: str
    type: str
    to: Optional[str] = None
    message: Optional[str] = None
    caregiver_initials: Optional[str] = None

@app.post("/notify")
def notify(body: Notify):
    print("NOTIFY:", body.dict())
    return {"ok": True}

class Escalate(BaseModel):
    summary: str
    priority: str = "high"

@app.post("/escalate")
def escalate(body: Escalate):
    print("ESCALATE ON-CALL:", body.dict())
    return {"ok": True}
EOF

cat > services/notifier/providers/sms_provider.py << 'EOF'
# TODO: Twilio/Telnyx SMS integration
EOF

cat > services/notifier/providers/email_provider.py << 'EOF'
# TODO: SendGrid/SES email integration
EOF

# --- packages ---

cat > packages/common/__init__.py << 'EOF'
# Common utilities package
EOF

cat > packages/common/logging.py << 'EOF'
# TODO: json logging helpers with call_id, turn_id
EOF

cat > packages/common/audit.py << 'EOF'
# TODO: append-only audit logging to DB or file
EOF

cat > packages/common/security.py << 'EOF'
# TODO: family/caregiver auth helpers (passphrases, roles)
EOF

cat > packages/common/timeouts.py << 'EOF'
# TODO: timeout & retry utilities
EOF

cat > packages/clients/__init__.py << 'EOF'
# HTTP client helpers package
EOF

cat > packages/clients/http.py << 'EOF'
# TODO: shared httpx client factory with retries & backoff
EOF

# --- data ---

cat > data/kb_company/README.md << 'EOF'
# Company Knowledge Base

Non-PHI facts about the franchise: hours, policies, cancellation rules, etc.
EOF

cat > data/kb_scenarios/README.md << 'EOF'
# Scenario Playbooks

Rules like: "If caregiver calls out within 2 hours of shift, attempt auto-refill then escalate."
EOF

cat > data/kb_examples/README.md << 'EOF'
# Golden Examples

Short, perfect dialogues showing ideal tool usage for each intent (callout, ETA, cancel/resched, etc.).
EOF

# --- offline ---

cat > offline/analyzers/intent_judge.py << 'EOF'
# TODO: offline intent accuracy analyzer
EOF

cat > offline/analyzers/instr_follow_judge.py << 'EOF'
# TODO: instruction-following analyzer
EOF

cat > offline/analyzers/hallucination_judge.py << 'EOF'
# TODO: hallucination analyzer
EOF

cat > offline/analyzers/pii_guard_strict.py << 'EOF'
# TODO: stricter PII checker
EOF

cat > offline/analyzers/tts_error_detector.py << 'EOF'
# TODO: detect TTS glitches
EOF

touch offline/datasets/intents.jsonl
touch offline/datasets/examples.jsonl

cat > offline/repro_harness/load_artifacts.py << 'EOF'
# TODO: load call artifacts into a reproducible format
EOF

cat > offline/repro_harness/run_repro.py << 'EOF'
# TODO: replay a call deterministically against current code
EOF

cat > offline/reports/README.md << 'EOF'
Use this folder for notebooks / KPIs (containment, time-to-fill, failure reasons, etc.).
EOF

# --- storage ---

cat > storage/sql/001_init.sql << 'EOF'
-- TODO: create tables for calls, turns, tool_calls, audits, evals
EOF

cat > storage/sql/002_indexes.sql << 'EOF'
-- TODO: indexes for call_id, created_at, failure_reason, etc.
EOF

cat > storage/README.md << 'EOF'
DB schema and migrations for analytics + audit logs.
EOF

# --- policy ---

cat > policy/franchise.default.yaml << 'EOF'
franchise_id: default
notice_windows:
  cancel_hours: 12
overtime_limits:
  max_hours_week: 40
dnd:
  caregiver_default: "22:00-06:00"
EOF

cat > policy/rah_demo.yaml << 'EOF'
franchise_id: rah_demo
extends: franchise.default
EOF

echo "✅ Skeleton created."
echo "Next:"
echo "  1) cp config/env.example .env  # and fill keys"
echo "  2) docker compose -f config/compose.yaml up --build"
echo "  3) Open this folder in Cursor and start filling in logic."

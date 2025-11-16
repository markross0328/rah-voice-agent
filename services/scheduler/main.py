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

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

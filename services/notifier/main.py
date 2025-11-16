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

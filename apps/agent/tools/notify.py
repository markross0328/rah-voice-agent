from typing import Dict, Any
import os, httpx

NOTIFIER_URL = os.environ.get("NOTIFIER_URL", "http://notifier:7003")

async def tool_notify(args: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(base_url=NOTIFIER_URL, timeout=10) as client:
        r = await client.post("/notify", json=args)
        r.raise_for_status()
    return {"ok": True}

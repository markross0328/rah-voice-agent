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

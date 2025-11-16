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

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

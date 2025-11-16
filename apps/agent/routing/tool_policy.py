from typing import List

def allowed_tools_for_intent(intent: str) -> List[str]:
    # TODO: restrict tools by intent (safety + clarity)
    return ["check_visit_status", "report_callout", "reschedule_shift", "notify_parties"]

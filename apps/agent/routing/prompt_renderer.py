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

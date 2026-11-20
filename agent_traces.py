"""Helpers shared by stateful_env.py and traces_to_training.py.

Both scripts produce the same training record: OpenAI-style chat messages with
assistant ``tool_calls`` and ``tool`` results, plus the tool definitions.
"""
from __future__ import annotations

import json
from typing import Any


def _maybe_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def tool_result_text(content: Any) -> str:
    """Return the tool payload from an MCP result as recorded by MCPAgentBlock.

    MCPAgentBlock stores a tool result as a JSON list of MCP content items, for
    example ``[{"type": "text", "text": "{...}"}]``. Training data keeps the
    payload itself. Several text items become one JSON array.
    """
    if not isinstance(content, str):
        return json.dumps(content)
    items = _maybe_json(content)
    if isinstance(items, list) and items and all(isinstance(item, dict) and item.get("type") == "text" for item in items):
        texts = [item["text"] for item in items]
        if len(texts) == 1:
            return texts[0]
        return json.dumps([_maybe_json(text) for text in texts])
    return content


def normalize_messages(raw_messages: list[dict]) -> list[dict]:
    """Keep only the fields a tool-calling training conversation needs.

    Accepts messages as MCPAgentBlock records them (LiteLLM message dumps with
    extra provider fields) and returns system, user, assistant and tool messages.
    """
    names_by_call_id: dict[str, str] = {}
    messages: list[dict] = []
    for raw in raw_messages:
        role = raw.get("role")
        if role in ("system", "user"):
            messages.append({"role": role, "content": raw.get("content") or ""})
        elif role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": raw.get("content")}
            calls = raw.get("tool_calls") or []
            if calls:
                message["tool_calls"] = []
                for call in calls:
                    function = call["function"]
                    names_by_call_id[call["id"]] = function["name"]
                    message["tool_calls"].append({
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": function["name"], "arguments": function.get("arguments") or "{}"},
                    })
            messages.append(message)
        elif role == "tool":
            call_id = raw["tool_call_id"]
            message = {"role": "tool", "tool_call_id": call_id, "content": tool_result_text(raw.get("content"))}
            name = names_by_call_id.get(call_id) or raw.get("name")
            if name:
                message["name"] = name
            messages.append(message)
        else:
            raise ValueError(f"Unexpected message role: {role!r}")
    return messages

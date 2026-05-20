from typing import Any, Dict, List


def extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def latest_user_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return extract_text(message.get("content", ""))
    return ""


def baseline_reply(messages: List[Dict[str, Any]]) -> str:
    user_text = latest_user_message(messages)
    if not user_text:
        return "Pulse proxy is running in baseline mode. Set OPENAI_API_KEY or pass an Authorization header to enable upstream proxying."
    return (
        "Pulse proxy received the request but no upstream API key is configured. "
        "Set OPENAI_API_KEY or PULSE_UPSTREAM_API_KEY, then retry. "
        f"Latest user message: {user_text[:500]}"
    )

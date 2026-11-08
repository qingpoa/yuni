import json

from yuni_agent.model.client import ModelStreamEvent, ModelToolCall


def reply_delivered(messages):
    if not messages or messages[-1].get("role") != "tool":
        return False
    try:
        return json.loads(messages[-1]["content"]).get("toolName") == "reply"
    except (ValueError, TypeError, AttributeError):
        return False


def stop_call():
    return ModelToolCall(id="stop-now", name="stop", arguments={"reason": "已回应，等待用户"})


def stop_event():
    return ModelStreamEvent(type="tool_call", tool_call=stop_call())


def reply_call(reference="依据当前用户原话回应。", **intent):
    return ModelToolCall(id="reply-now", name="reply", arguments={
        "reply_reference": reference,
        "expression_intent": intent or {"focus": "当前对话"},
    })


def reply_event(reference="依据当前用户原话回应。", **intent):
    return ModelStreamEvent(type="tool_call", tool_call=reply_call(reference, **intent))

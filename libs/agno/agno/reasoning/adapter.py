from __future__ import annotations

from typing import List, Optional

from agno.models.message import Message
from agno.utils.log import logger


def _normalize_system_role(messages: List[Message]) -> None:
    for message in messages:
        if message.role == "developer":
            message.role = "system"


def _extract_reasoning_content(run_response: object) -> str:
    reasoning_content = ""
    response_messages = getattr(run_response, "messages", None)
    if response_messages:
        for msg in response_messages:
            content = getattr(msg, "reasoning_content", None)
            if content:
                reasoning_content = content
                break

    if not reasoning_content:
        reasoning_content = getattr(run_response, "reasoning_content", "") or ""

    return reasoning_content


def get_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:  # type: ignore  # noqa: F821
    from agno.run.agent import RunOutput

    _normalize_system_role(messages)

    try:
        reasoning_agent_response: RunOutput = reasoning_agent.run(messages=messages)
    except Exception as e:
        logger.warning(f"Reasoning error: {e}")
        return None

    reasoning_content = _extract_reasoning_content(reasoning_agent_response)

    return Message(
        role="assistant",
        content=f"<thinking>\n{reasoning_content}\n</thinking>",
        reasoning_content=reasoning_content,
    )


async def aget_reasoning(reasoning_agent: "Agent", messages: List[Message]) -> Optional[Message]:  # type: ignore  # noqa: F821
    from agno.run.agent import RunOutput

    _normalize_system_role(messages)

    try:
        reasoning_agent_response: RunOutput = await reasoning_agent.arun(messages=messages)
    except Exception as e:
        logger.warning(f"Reasoning error: {e}")
        return None

    reasoning_content = _extract_reasoning_content(reasoning_agent_response)

    return Message(
        role="assistant",
        content=f"<thinking>\n{reasoning_content}\n</thinking>",
        reasoning_content=reasoning_content,
    )

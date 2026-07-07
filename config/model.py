import os
from types import MappingProxyType
from langchain_openai import ChatOpenAI

_COMMON_MODEL_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "base_url": os.getenv("OPENAI_BASE_URL"),
}

MODEL_AGENT_CONFIG = MappingProxyType(
    {
        "default": os.getenv("OPENAI_MODEL", "gpt-5.5"),
        "fast": "gpt-5.4-mini",
        "reasoning": "gpt-5.5",
    }
)

model_agent = MappingProxyType(
    {
        agent_name: ChatOpenAI(
            model=model_name,
            **_COMMON_MODEL_CONFIG,
        )
        for agent_name, model_name in MODEL_AGENT_CONFIG.items()
    }
)

DEFAULT_MODEL_AGENT_NAME = os.getenv("DEFAULT_MODEL_AGENT_NAME", "default")
if DEFAULT_MODEL_AGENT_NAME not in model_agent:
    raise ValueError(f"Unknown model agent: {DEFAULT_MODEL_AGENT_NAME}")

CHAT_MODEL = model_agent[DEFAULT_MODEL_AGENT_NAME]

__all__ = [
    "CHAT_MODEL",
    "DEFAULT_MODEL_AGENT_NAME",
    "MODEL_AGENT_CONFIG",
    "model_agent",
]

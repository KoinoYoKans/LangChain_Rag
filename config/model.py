from langchain_openai import ChatOpenAI
from config.settings import AppSettings


#构建聊天模型
def build_chat_model(settings: AppSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )

__all__ = [
    "build_chat_model",
]

from langchain_openai import ChatOpenAI
from config.settings import AppSettings


#构建聊天模型
def build_chat_model(settings: AppSettings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

__all__ = [
    "build_chat_model",
]

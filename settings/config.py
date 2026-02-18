from typing import Optional, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    deepseek_api_key: Optional[str] = Field()

    channel_to_send: Literal["TELEGRAM", "EMAIL"] = Field(
        description="Output channel: TELEGRAM or EMAIL"
    )

    telegram_bot_token: Optional[str] = Field(
        default=None,
        description="Telegram bot token"
    )

    telegram_chat_id: Optional[int] = Field(
        default=None,
        description="chat id for Telegram"
    )


settings = Settings()

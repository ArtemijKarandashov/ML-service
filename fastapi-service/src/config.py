"""
Importing settings from .env file
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MASTER_PASSWORD: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra = "ignore"
        )

Config: Settings = Settings()
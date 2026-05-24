"""
Importing settings from .env file
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    #EXAMPLE_ENV_VAR: str
    STORAGE_SERVICE_API: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra = "ignore"
        )

Config: Settings = Settings()
from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    PRODUCTION = "production"


class CoreSettings(BaseSettings):
    """Settings every RAGMed service shares. Services subclass this with their own fields."""

    model_config = SettingsConfigDict(
        env_prefix="RAGMED_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    service_name: str = "ragmed"
    log_level: LogLevel = "INFO"
    log_json: bool | None = None

    @property
    def use_json_logs(self) -> bool:
        # Human-readable logs on a laptop, JSON everywhere else unless overridden.
        if self.log_json is not None:
            return self.log_json
        return self.environment is not Environment.LOCAL


class ProviderKeys(BaseSettings):
    """Model provider API keys, under their usual names (no RAGMED_ prefix).

    Read from the environment, the repository's .env, or backend/.env.
    """

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore", frozen=True)

    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    cal_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> CoreSettings:
    return CoreSettings()

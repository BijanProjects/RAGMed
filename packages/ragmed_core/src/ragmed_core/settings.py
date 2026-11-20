from enum import StrEnum
from functools import lru_cache
from typing import Literal

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


@lru_cache(maxsize=1)
def get_settings() -> CoreSettings:
    return CoreSettings()

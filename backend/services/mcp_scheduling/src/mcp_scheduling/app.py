"""Build the Scheduling MCP server from settings: Cal.com when configured, else in memory."""

from pathlib import Path

from fastmcp import FastMCP
from pydantic import Field, SecretStr
from pydantic_settings import SettingsConfigDict

from mcp_scheduling.auth import ServiceTokenVerifier
from mcp_scheduling.providers import (
    CalComProvider,
    InMemoryProvider,
    SchedulingProvider,
    discover_event_types,
)
from mcp_scheduling.server import build_server
from mcp_scheduling.store import AppointmentStore
from ragmed_core.settings import CoreSettings, Environment, ProviderKeys


class SchedulingSettings(CoreSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGMED_", env_file=("../.env", ".env"), extra="ignore", frozen=True
    )

    service_name: str = "mcp-scheduling"
    host: str = "127.0.0.1"
    port: int = 9002
    clinic_time_zone: str = "America/St_Johns"
    # "calcom" needs CAL_API_KEY; "memory" is a built-in calendar for tests and offline demos.
    calendar: str = "calcom"
    cal_event_15_id: int | None = None  # found automatically when not set
    cal_event_30_id: int | None = None
    appointments_db: Path = Path("data/artifacts/clinic/appointments.db")
    service_tokens: dict[str, SecretStr] = Field(default_factory=dict)
    clinician_tokens: dict[str, SecretStr] = Field(default_factory=dict)


async def make_provider(settings: SchedulingSettings) -> SchedulingProvider:
    if settings.calendar == "memory":
        return InMemoryProvider(time_zone=settings.clinic_time_zone)
    key = ProviderKeys().cal_api_key
    if key is None:
        raise RuntimeError("CAL_API_KEY is not set; add it to .env or use RAGMED_CALENDAR=memory")
    ids = (
        {15: settings.cal_event_15_id, 30: settings.cal_event_30_id}
        if settings.cal_event_15_id and settings.cal_event_30_id
        else await discover_event_types(key.get_secret_value())
    )
    return CalComProvider(key.get_secret_value(), ids, settings.clinic_time_zone)


def create_server(settings: SchedulingSettings, provider: SchedulingProvider) -> FastMCP:
    tokens = {n: t.get_secret_value() for n, t in settings.service_tokens.items()}
    if not tokens and settings.environment is Environment.PRODUCTION:
        raise RuntimeError("RAGMED_SERVICE_TOKENS must be set in production")
    return build_server(
        provider,
        AppointmentStore(settings.appointments_db),
        time_zone=settings.clinic_time_zone,
        auth=ServiceTokenVerifier(tokens) if tokens else None,
        clinician_tokens={n: t.get_secret_value() for n, t in settings.clinician_tokens.items()},
    )

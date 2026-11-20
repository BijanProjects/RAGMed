"""Run the Scheduling MCP server: python -m mcp_scheduling  (MCP on RAGMED_PORT, default 9002)."""

import anyio

from mcp_scheduling.app import SchedulingSettings, create_server, make_provider
from ragmed_core.logging import configure_logging


def main() -> None:  # pragma: no cover - starts a server
    settings = SchedulingSettings()
    configure_logging(settings)
    provider = anyio.run(make_provider, settings)
    server = create_server(settings, provider)
    server.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()

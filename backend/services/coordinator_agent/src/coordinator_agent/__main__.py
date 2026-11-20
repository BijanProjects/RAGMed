"""Run the Care Coordinator: python -m coordinator_agent  (A2A on RAGMED_PORT, default 8000)."""

import uvicorn

from coordinator_agent.app import CoordinatorSettings, create_coordinator_app
from ragmed_core.logging import configure_logging


def main() -> None:  # pragma: no cover - starts a server
    settings = CoordinatorSettings()
    configure_logging(settings)
    uvicorn.run(create_coordinator_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()

"""Run the Care Coordinator: python -m coordinator_agent  (A2A on RAGMED_PORT, default 8000).

With RAGMED_DATABASE_URL, conversations are saved in Postgres; with RAGMED_VALKEY_URL, rate
limits are shared through Valkey.
"""

from contextlib import AsyncExitStack

import anyio
import uvicorn

from coordinator_agent.app import CoordinatorSettings, create_coordinator_app
from ragmed_core.logging import configure_logging
from ragmed_core.ratelimit import make_rate_limiter


async def serve(settings: CoordinatorSettings) -> None:  # pragma: no cover - starts a server
    async with AsyncExitStack() as stack:
        checkpointer = None
        if settings.database_url:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

            url = settings.database_url.get_secret_value()
            checkpointer = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(url))
            await checkpointer.setup()  # creates its tables the first time
        app = create_coordinator_app(
            settings, checkpointer=checkpointer, limiter=make_rate_limiter(settings.valkey_url)
        )
        config = uvicorn.Config(app, host=settings.host, port=settings.port, log_level="warning")
        await uvicorn.Server(config).serve()


def main() -> None:  # pragma: no cover - starts a server
    settings = CoordinatorSettings()
    configure_logging(settings)
    anyio.run(serve, settings)


if __name__ == "__main__":  # pragma: no cover
    main()

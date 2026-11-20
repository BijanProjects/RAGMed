"""Run the Diagnosis agent: python -m diagnosis_agent  (A2A on RAGMED_PORT, default 8001)."""

import uvicorn

from diagnosis_agent.app import DiagnosisSettings, create_diagnosis_app
from ragmed_core.logging import configure_logging


def main() -> None:  # pragma: no cover - starts a server
    settings = DiagnosisSettings()
    configure_logging(settings)
    uvicorn.run(create_diagnosis_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":  # pragma: no cover
    main()

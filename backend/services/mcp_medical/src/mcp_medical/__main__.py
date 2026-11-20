"""Run the server, or load the knowledge base into the vector store.

python -m mcp_medical ingest --vectors <file.npy>   # once, after `make export-onnx`
python -m mcp_medical serve                          # MCP over HTTP on RAGMED_PORT (9001)
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import anyio

from mcp_medical.app import MedicalSettings, create_server, ingest
from ragmed_core.logging import configure_logging


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - starts a server
    parser = argparse.ArgumentParser(prog="mcp-medical")
    sub = parser.add_subparsers(dest="command", required=True)
    load = sub.add_parser("ingest", help="load passages and vectors into the vector store")
    load.add_argument("--vectors", type=Path, help="pre-computed passage vectors (.npy)")
    sub.add_parser("serve", help="run the MCP server over streamable HTTP")
    args = parser.parse_args(argv)

    settings = MedicalSettings()
    configure_logging(settings)
    if args.command == "ingest":
        count = anyio.run(ingest, settings, args.vectors)
        print(f"indexed {count} passages into {settings.collection!r}")
        return 0
    server = create_server(settings)
    server.run(transport="http", host=settings.host, port=settings.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Serve an agent over A2A: its Agent Card, JSON-RPC endpoint, bearer auth and /healthz."""

import hmac
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from a2a.helpers import new_data_part, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    TaskState,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

RPC_PATH = "/a2a"
BEARER = "bearer"
MIN_TOKEN_LENGTH = 32


@dataclass(frozen=True, slots=True)
class Skill:
    id: str
    name: str
    description: str
    tags: Sequence[str] = ()
    examples: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class CardSpec:
    name: str
    description: str
    url: str
    skills: Sequence[Skill]
    version: str = "0.1.0"
    input_modes: Sequence[str] = ("text/plain",)
    output_modes: Sequence[str] = ("text/plain", "application/json")
    requires_auth: bool = True
    extra: Mapping[str, str] = field(default_factory=dict)


def agent_card(spec: CardSpec) -> AgentCard:
    """An Agent Card advertising JSON-RPC at `url` + /a2a, streaming, and bearer auth."""
    card = AgentCard(
        name=spec.name,
        description=spec.description,
        version=spec.version,
        supported_interfaces=[
            AgentInterface(
                url=spec.url.rstrip("/") + RPC_PATH,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=list(spec.input_modes),
        default_output_modes=list(spec.output_modes),
        skills=[
            AgentSkill(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=list(s.tags),
                examples=list(s.examples),
            )
            for s in spec.skills
        ],
    )
    if spec.requires_auth:
        card.security_schemes[BEARER].CopyFrom(
            SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer", description="Service token issued per caller"
                )
            )
        )
        card.security_requirements.append(
            SecurityRequirement(schemes={BEARER: StringList(list=[])})
        )
    return card


class BearerAuth:
    """Requires a known bearer token on the JSON-RPC path; the Agent Card stays public.

    Tokens are compared in constant time. Each caller has its own token, so one can be revoked
    without affecting the others.
    """

    def __init__(self, app: ASGIApp, tokens: Mapping[str, str], path: str = RPC_PATH) -> None:
        if any(len(t) < MIN_TOKEN_LENGTH for t in tokens.values()):
            raise ValueError(f"service tokens must be at least {MIN_TOKEN_LENGTH} characters")
        self._app = app
        self._tokens = [t.encode() for t in tokens.values()]
        self._path = path

    def _allowed(self, header: str) -> bool:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        presented = token.strip().encode()
        matched = False
        for expected in self._tokens:  # compare against all, to keep timing constant
            matched |= hmac.compare_digest(presented, expected)
        return matched

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(self._path):
            headers = dict(scope["headers"])
            if not self._allowed(headers.get(b"authorization", b"").decode("latin-1")):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)


async def _healthz(_request: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_app(
    card: AgentCard,
    executor: AgentExecutor,
    *,
    tokens: Mapping[str, str] | None = None,
    task_store: TaskStore | None = None,
) -> ASGIApp:
    """The ASGI app for one agent. With `tokens`, every JSON-RPC call needs one of them."""
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=task_store or InMemoryTaskStore(), agent_card=card
    )
    routes = [
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, RPC_PATH),
        Route("/healthz", _healthz, methods=["GET"]),
    ]
    app: ASGIApp = Starlette(routes=routes)
    return BearerAuth(app, tokens) if tokens else app


async def start_task(context: RequestContext, event_queue: EventQueue) -> TaskUpdater:
    """Create the task for a new request (the protocol needs it before any update)."""
    if context.message is None or context.task_id is None or context.context_id is None:
        raise ValueError("A2A request has no message, task id or context id")
    if context.current_task is None:
        await event_queue.enqueue_event(new_task_from_user_message(context.message))
    return TaskUpdater(event_queue, context.task_id, context.context_id)


async def report_progress(updater: TaskUpdater, text: str, agent: str = "") -> None:
    """A "working" status update. `agent` names who is acting (for example "Knowledge base"),
    sent as a data part beside the text so a client can show which agent is doing what."""
    parts = [new_text_part(text)]
    if agent:
        parts.append(new_data_part({"agent": agent}))
    await updater.update_status(TaskState.TASK_STATE_WORKING, updater.new_agent_message(parts))

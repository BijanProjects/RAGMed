# One image for every backend service; docker compose picks the service with its command.
# Built on the server itself (Arm64), so no cross-compiling is needed.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The same user id as the server's login user, so files in the mounted data folder stay
# readable and writable from both sides.
RUN useradd --create-home --uid 1000 ragmed
WORKDIR /app

COPY requirements/prod.txt requirements/prod.txt
COPY packages packages
COPY services services
RUN pip install -r requirements/prod.txt

USER ragmed
# Listen on the container network; only Caddy is reachable from the internet.
ENV RAGMED_HOST=0.0.0.0 \
    RAGMED_ENVIRONMENT=production \
    RAGMED_LOG_LEVEL=INFO

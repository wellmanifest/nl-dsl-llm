# Multi-stage production container for paxlet-com/nl-dsl-llm runtime daemon
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels .

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS runner

LABEL org.opencontainers.image.title="paxlet-nl-runtime" \
      org.opencontainers.image.description="High-performance gRPC IPC daemon for wellmanifest/nl-dsl-llm" \
      org.opencontainers.image.version="0.2.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NL_DSL_SOCKET="/run/paxlet/nl-dsl-llm.sock" \
    NL_DSL_PORT=50051

WORKDIR /app
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

RUN mkdir -p /run/paxlet && chmod 777 /run/paxlet

EXPOSE 50051
VOLUME ["/run/paxlet"]

ENTRYPOINT ["python3", "-m", "paxlet_nl_dsl_llm.server"]
CMD ["--uds", "/run/paxlet/nl-dsl-llm.sock", "--port", "50051"]

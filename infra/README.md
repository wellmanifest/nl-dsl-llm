# Runtime Infrastructure & Deployment

## Purpose and Scope
This package provides production deployment artifacts (Docker container, Docker Compose, and Systemd service unit) for running the shared `paxlet-com/nl-dsl-llm` daemon across Taskand, Clonerd, and `nl-dsl-sh` on an 8 vCPU / 8 GB RAM VPS.

## Deployment Manifest
- `Dockerfile`: Multi-stage Python 3.13-slim build with non-root runtime and socket volume exposure.
- `compose.yml`: Compose topology mounting shared `/run/paxlet/` volume and wiring host Ollama / llama-server gateways.
- `infra/systemd/nl-runtime.service`: Native systemd service unit for bare-metal VPS execution.

## Deployment Guide

### Docker Compose Mode
Mount the shared `paxlet_run` volume across consumer containers (Taskand and Clonerd):
```bash
docker compose up -d
# Verify daemon health
curl -s http://127.0.0.1:50051 || true
```

### Systemd Host Mode
```bash
sudo cp infra/systemd/nl-runtime.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nl-runtime
sudo systemctl status nl-runtime
```

## Safety and Resource Allocations
- Container CPU limit: 2.0 vCPU; memory limit: 1024 MB.
- Socket permissions: `/run/paxlet` configured with `chmod 777` inside volume to allow unprivileged Node.js and Python processes read/write access.

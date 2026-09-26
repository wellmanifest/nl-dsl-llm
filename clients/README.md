# Downstream Client Adapters

## Purpose and Scope
This package provides lightweight, zero-overhead client adapters for Taskand, Clonerd, and `nl-dsl-sh` to interact with the `paxlet-com/nl-dsl-llm` daemon over Unix Domain Sockets (`/run/paxlet/nl-dsl-llm.sock`) or TCP (`:50051`).

## Client Manifest
- `clients/node/taskand_nl_client.mjs`: Native Node.js stream client for Taskand `dev/chat` and `dev/llm`.
- `clients/python/nl_dsl_client.py`: Python client helper with in-memory fallback for Python daemons.
- `clients/bash/nl-dsl`: CLI executable for shell integration, testing, and side-by-side Qwen benchmarking.

## Usage Guide

### Taskand Integration
Import `TaskandNLClient` into `taskand/generated/dev/chat/taskand.dev/v1/intent.mjs`:
```javascript
import { TaskandNLClient } from 'paxlet-nl-dsl-llm/clients/node/taskand_nl_client.mjs';
const client = new TaskandNLClient();
const { uri, arguments, latencyMs } = await client.interpret("pokaż zużycie ramu");
```

### nl-dsl-sh Shell Integration
```bash
# Natural language command execution
$ ./clients/bash/nl-dsl ask "pokaż status cpu"

# Dual-Qwen comparison
$ ./clients/bash/nl-dsl bench "restartuj bazę danych na produkcji"
```

## Safety and Operational Bounds
- All clients fail closed: if daemon or LLM is unreachable, they emit structured error envelopes rather than throwing unhandled exceptions.
- Socket timeouts default to 5000ms. In high-load scenarios, fast-path regex and semantic cache resolve within <15ms.

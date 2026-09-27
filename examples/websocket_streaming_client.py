#!/usr/bin/env python3
"""Interactive client and latency benchmark for NL-DSL-LLM WebSocket Streaming & Option Network."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
import statistics
import sys
import time
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from standard.nl_dsl_llm_ws import (
    NLStreamFrame,
    NLStreamServer,
    StreamFrameType,
    StreamingOptionNetworkEngine,
    decode_ws_frame,
    encode_ws_frame,
)


def run_benchmark(iterations: int = 1000) -> None:
    """Benchmark Option Network autocomplete latency over streaming prefixes."""
    print(f"=== Option Network Streaming Autocomplete Benchmark ({iterations} iterations) ===")
    engine = StreamingOptionNetworkEngine()

    prefixes = [
        "z", "za", "zat", "zatr", "zatrz", "zatrzy", "zatrzymaj",
        "d", "do", "dod", "doda", "dodaj",
        "p", "po", "pok", "poka", "pokaż",
        "s", "st", "sta", "stan",
        "r", "re", "res", "rest", "restart",
    ]

    latencies: List[float] = []

    # Warmup
    for p in prefixes:
        engine.update_prefix(p)

    for i in range(iterations):
        p = prefixes[i % len(prefixes)]
        t0 = time.perf_counter()
        res = engine.update_prefix(p)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    max_lat = max(latencies)
    min_lat = min(latencies)
    mean_lat = statistics.mean(latencies)

    print(f"Iterations : {iterations}")
    print(f"Min Latency: {min_lat:.3f} ms")
    print(f"Mean       : {mean_lat:.3f} ms")
    print(f"p50 Median : {p50:.3f} ms")
    print(f"p95        : {p95:.3f} ms")
    print(f"p99        : {p99:.3f} ms")
    print(f"Max Latency: {max_lat:.3f} ms")
    print(f"SLA Target : < 50.000 ms")
    print(f"SLA Status : {'PASS (Sub-50ms verified)' if p99 < 50.0 else 'FAIL'}")

    assert p99 < 50.0, f"Option Network p99 latency {p99:.2f}ms exceeded 50ms SLA"


async def run_live_simulation(port: int = 0) -> None:
    """Run an embedded server and simulate streaming speech tokens with Option Network updates."""
    print("=== Starting NL-DSL-LLM Streaming Server ===")
    server = NLStreamServer(host="127.0.0.1", port=port)
    await server.start()
    bound_port = server.server.sockets[0].getsockname()[1]
    print(f"Server listening on ws://127.0.0.1:{bound_port}/ws/stream")

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", bound_port)

        # Handshake
        client_key = base64.b64encode(b"benchmark_key_1234").decode("ascii")
        req = (
            f"GET /ws/stream HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{bound_port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {client_key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        writer.write(req.encode("ascii"))
        await writer.drain()
        await reader.readuntil(b"\r\n\r\n")

        def send_text(text: str):
            writer.write(encode_ws_frame(text, opcode=1, mask=True))

        rx_buf = bytearray()
        async def read_msg() -> str:
            while True:
                res = decode_ws_frame(rx_buf)
                if res:
                    _, payload, consumed = res
                    del rx_buf[:consumed]
                    return payload.decode("utf-8")
                chunk = await reader.read(2048)
                if not chunk:
                    raise ConnectionResetError("Server closed connection")
                rx_buf.extend(chunk)

        # 1. STREAM_START
        start = NLStreamFrame(
            type=StreamFrameType.STREAM_START,
            session_id="sim_session",
            sequence=1,
            timestamp_ms=time.time() * 1000,
            payload={"client": "cli-demo"},
        )
        send_text(start.to_json())
        await writer.drain()
        resp = await read_msg()
        print(f"<- {resp}")

        # 2. Simulate streaming speech tokens
        tokens = ["z", "zatr", "zatrzymaj", "zatrzymaj uslugę", "zatrzymaj uslugę auth-service"]
        for i, token in enumerate(tokens, 2):
            print(f"\n[Voice Stream Chunk #{i-1}]: \"{token}\"")
            frame = NLStreamFrame(
                type=StreamFrameType.TRANSCRIPTION_PARTIAL,
                session_id="sim_session",
                sequence=i,
                timestamp_ms=time.time() * 1000,
                payload={"text": token},
            )
            send_text(frame.to_json())
            await writer.drain()
            opt_resp = await read_msg()
            data = json.loads(opt_resp)
            cands = data["payload"]["candidates"]
            print(f"  Option Network DAG ({data['payload']['latency_ms']}ms):")
            for c in cands[:3]:
                badge = c.get('badge') or 'CMD'
                conf = c.get('confidence', 0.9)
                print(f"    * [{badge}] {c['label']} (confidence: {conf})")

        # 3. Commit
        print("\n[User Spoke / Committed Action]")
        commit = NLStreamFrame(
            type=StreamFrameType.COMMAND_COMMITTED,
            session_id="sim_session",
            sequence=len(tokens) + 2,
            timestamp_ms=time.time() * 1000,
            payload={"text": "zatrzymaj uslugę auth-service"},
        )
        send_text(commit.to_json())
        await writer.drain()
        final_resp = await read_msg()
        final_data = json.loads(final_resp)
        print("<- Final Execution Result:")
        print(f"   Success     : {final_data['payload'].get('success')}")
        print(f"   Status      : {final_data['payload'].get('status')}")
        print(f"   Data        : {final_data['payload'].get('data')}")
        print(f"   Execution Time: {final_data['payload'].get('execution_ms')}ms")

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


def main():
    parser = argparse.ArgumentParser(description="NL-DSL-LLM WebSocket Streaming Client")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--simulate", action="store_true", help="Run simulated voice streaming session")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        asyncio.run(run_live_simulation())


if __name__ == "__main__":
    main()

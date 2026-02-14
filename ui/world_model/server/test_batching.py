"""Test ORCA batching: synchronized vs random keypresses."""
import argparse
import asyncio
import json
import random
import time
import websockets

SERVER_URL = "ws://localhost:8000/ws"
KEYS = ["w", "a", "s", "d"]
# Think time range for random mode (ms)
MIN_THINK_MS = 200
MAX_THINK_MS = 1500


async def connect_and_wait(client_id, ws):
    """Drain messages until initial frame_batch is received."""
    while True:
        msg = json.loads(await ws.recv())
        if msg["type"] == "frame_batch":
            print(f"  Client {client_id}: got initial frame")
            return


async def synced_client(client_id: int, barrier: asyncio.Barrier, results: list,
                        num_rounds: int):
    """Synchronized keypresses — all clients press at the same time."""
    async with websockets.connect(
        SERVER_URL, max_size=50_000_000, ping_interval=None,
    ) as ws:
        await connect_and_wait(client_id, ws)

        for round_idx in range(num_rounds):
            await barrier.wait()

            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "key", "key": "w"}))

            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "frame_batch":
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(elapsed_ms)
            print(f"  Client {client_id} round {round_idx}: {elapsed_ms:.0f}ms")


async def random_client(client_id: int, start_event: asyncio.Event, results: list,
                        num_rounds: int):
    """Random keypresses with think time between presses."""
    async with websockets.connect(
        SERVER_URL, max_size=50_000_000, ping_interval=None,
    ) as ws:
        await connect_and_wait(client_id, ws)
        start_event.set()

        # Small stagger so clients don't all start at once
        await asyncio.sleep(random.uniform(0, 1.0))

        for _ in range(num_rounds):
            think_ms = random.uniform(MIN_THINK_MS, MAX_THINK_MS)
            await asyncio.sleep(think_ms / 1000)

            t0 = time.perf_counter()
            await ws.send(json.dumps({"type": "key", "key": random.choice(KEYS)}))

            while True:
                msg = json.loads(await ws.recv())
                if msg["type"] == "frame_batch":
                    break

            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(elapsed_ms)

        print(f"  Client {client_id}: done")


def print_stats(times: list[float], label: str):
    """Print summary statistics and histogram."""
    times.sort()
    n = len(times)
    print(f"\n--- {label} ({n} keypresses) ---")
    print(f"  Min:    {times[0]:.0f}ms")
    print(f"  p25:    {times[n//4]:.0f}ms")
    print(f"  Median: {times[n//2]:.0f}ms")
    print(f"  p75:    {times[3*n//4]:.0f}ms")
    print(f"  p95:    {times[int(n*0.95)]:.0f}ms")
    print(f"  Max:    {times[-1]:.0f}ms")
    print(f"  Avg:    {sum(times)/n:.0f}ms")

    buckets = [0] * 10
    labels = ["<1s", "<1.5s", "<2s", "<2.5s", "<3s",
              "<3.5s", "<4s", "<4.5s", "<5s", "5s+"]
    for ms in times:
        idx = min(int(ms / 500), 9)
        buckets[idx] += 1
    print(f"\n  Distribution:")
    for lbl, count in zip(labels, buckets):
        if count > 0:
            bar = "#" * count
            pct = count / n * 100
            print(f"    {lbl:>6s}: {bar} ({count}, {pct:.0f}%)")


async def run_synced(num_clients: int, num_rounds: int):
    print(f"=== SYNCHRONIZED MODE ({num_clients} clients, {num_rounds} rounds) ===")
    barrier = asyncio.Barrier(num_clients)
    results = []
    tasks = [synced_client(i, barrier, results, num_rounds) for i in range(num_clients)]
    await asyncio.gather(*tasks)
    print_stats(results, "Synchronized (worst case)")


async def run_random(num_clients: int, num_rounds: int):
    print(f"=== RANDOM MODE ({num_clients} clients, {num_rounds} presses each) ===")
    print(f"    Think time: {MIN_THINK_MS}-{MAX_THINK_MS}ms")
    results = []
    events = [asyncio.Event() for _ in range(num_clients)]
    tasks = [random_client(i, events[i], results, num_rounds) for i in range(num_clients)]
    await asyncio.gather(*tasks)
    print_stats(results, "Random (realistic)")


def parse_args():
    parser = argparse.ArgumentParser(description="Test ORCA multi-user batching")
    parser.add_argument("mode", nargs="?", default="both",
                        choices=["synced", "random", "both"])
    parser.add_argument("-c", "--clients", type=int, default=4,
                        help="Number of concurrent clients (default: 4)")
    parser.add_argument("-r", "--rounds", type=int, default=10,
                        help="Number of rounds per client (default: 10)")
    return parser.parse_args()


async def main():
    args = parse_args()
    print(f"Connecting {args.clients} clients to {SERVER_URL}...\n")

    if args.mode in ("synced", "both"):
        await run_synced(args.clients, args.rounds)
    if args.mode in ("random", "both"):
        if args.mode == "both":
            print("\n" + "=" * 60 + "\n")
        await run_random(args.clients, args.rounds)


if __name__ == "__main__":
    asyncio.run(main())

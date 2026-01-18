#!/bin/bash
# Kill all GPU processes shown by nvidia-smi

pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')

if [ -z "$pids" ]; then
    echo "No GPU processes found."
    exit 0
fi

echo "Killing GPU processes: $pids"
for pid in $pids; do
    kill -9 "$pid" 2>/dev/null && echo "Killed PID $pid" || echo "Failed to kill PID $pid"
done

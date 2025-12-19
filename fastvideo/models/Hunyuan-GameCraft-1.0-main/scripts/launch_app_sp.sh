#!/bin/bash

mkdir -p ./gradio_results
pkill -f inference_worker || true
pkill -f api_server || true
pkill -f app_gamecraft || true
echo "🧹 Cleaning up potential stale shutdown signal file..."
rm -f ./gradio_results/shutdown_worker.signal

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE="weights/stdmodels"
export CKPT_PATH="weights/gamecraft_models/mp_rank_00_model_states_distill.pt"
export API_PORT="8082" # For distributed inference server port
export GRADIO_PORT="8080" # For GradioUI port
export VIDEO_ENC="vp09" # "avc1"(faster)
NUM_GPUS=8

PROJECT_DIR="$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
WORKER_SCRIPT="$PROJECT_DIR/hymm_sp/gradio/inference_worker.py"
API_SCRIPT="$PROJECT_DIR/hymm_sp/gradio/api_server.py"
GRADIO_APP="$PROJECT_DIR/hymm_sp/gradio/app_gamecraft.py" 

echo "🚀 Starting Inference Worker..."
torchrun \
--nnodes=1 \
--nproc_per_node=$NUM_GPUS \
--master_port 29605 \
"$WORKER_SCRIPT" > gradio_results/worker.log 2>&1 &

WORKER_PID=$!
echo "✅ Inference Worker started with PID $WORKER_PID. Logs in worker.log"

echo "🌐 Starting API Server..."
python3 "$API_SCRIPT" > gradio_results/api.log 2>&1 &
API_PID=$!

echo "⏳ Waiting for Worker to load model and initialize..."
sleep 40 
echo "🔍 Checking API Server health..."

HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL=15
HEALTHY=false

for ((i=1; i<=HEALTH_CHECK_RETRIES; i++)); do
  echo "---- Health Check Attempt $i ----"
  RESPONSE=$(curl -s --noproxy localhost http://localhost:${API_PORT}/health)
  CURL_EXIT_CODE=$?
  
  echo "Curl Exit Code: $CURL_EXIT_CODE"
  echo "Response Body (first 500 chars): $(echo "$RESPONSE" | head -c 500)"
  
  if [ $CURL_EXIT_CODE -eq 0 ] && echo "$RESPONSE" | grep -q '"status"[[:space:]]*:[[:space:]]*"healthy"'; then
    echo "✅ API Server is healthy."
    HEALTHY=true
    break
  else
    if [ $CURL_EXIT_CODE -ne 0 ]; then
      echo "⚠️  Curl command failed."
    else
      echo "⚠️  API Server responded, but status is not healthy or response format unexpected."
      echo "Checking ./gradio_results directory status:"
      ls -ld ./gradio_results 2>&1 || echo "Directory check failed or directory does not exist"
    fi
  fi
  echo "--------------------"
  
  if [ $i -lt $HEALTH_CHECK_RETRIES ]; then
    echo "⏳ Waiting for API Server to be healthy... (attempt $i/$HEALTH_CHECK_RETRIES)"
    sleep $HEALTH_CHECK_INTERVAL
  fi
done


if [ "$HEALTHY" = false ]; then
  echo "❌ API Server failed to become healthy. Check api.log and worker.log"
  # try kill
  kill $WORKER_PID $API_PID 2>/dev/null || true
  wait $WORKER_PID $API_PID 2>/dev/null || true
  exit 1
fi

echo "✅ API Server is healthy."
echo "🔧 Temporarily disabling proxy again for Gradio UI startup..."

echo "🎨 Starting Gradio UI..."
python3 "$GRADIO_APP"



echo "🛑 Gradio UI stopped. Shutting down services..."
echo "shutdown" > ./gradio_results/shutdown_worker.signal
echo "📤 Shutdown signal sent to worker."
echo "⏳ Waiting for services to shut down..."
pkill -f app_gamecraft || true
timeout 30 tail --pid=$WORKER_PID -f /dev/null 2>/dev/null || true
timeout 10 tail --pid=$API_PID -f /dev/null 2>/dev/null || true
# forcing
kill $WORKER_PID $API_PID 2>/dev/null || true
wait $WORKER_PID $API_PID 2>/dev/null || true
echo "🔚 All services shut down."



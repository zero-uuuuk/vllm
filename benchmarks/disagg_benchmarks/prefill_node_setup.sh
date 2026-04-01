#!/bin/bash

# --- Prefill Node Orchestration Script ---
# Run this on the Master node (g5 instance).

set -e

# --- Configuration ---
# Set these to actual IPs of your instances
PREFILL_IP=${PREFILL_IP:-"127.0.0.1"}
DECODE_IP=${DECODE_IP:-"127.0.0.1"}
MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# --- 1. Environment & Dependency Setup ---
echo "Checking dependencies..."
(which jq) || (sudo apt-get -y install jq)
(which socat) || (sudo apt-get -y install socat)
uv pip install quart aiohttp ray httpx datasets msgpack

RESULT_DIR="./results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

# --- 2. Cleanup Logic ---
cleanup() {
    echo "Cleaning up local processes..."
    pkill -f "vllm serve" || true
    pkill -f "disagg_prefill_proxy_server.py" || true
    ray stop || true
}
trap cleanup EXIT

# --- 3. Ray Head Setup ---
echo "Initializing Ray Head..."
ray start --head --node-ip-address "$PREFILL_IP" --port 6379

# --- 4. [Sync] Wait for Worker Node ---
# Ray Cluster에 Worker Node가 접속할 때까지 대기합니다. (최소 2개 노드 필요)
echo "Waiting for Decode Node to join the Ray Cluster..."
echo "Please run 'decode_node_setup.sh' on the Decode Node now."

until [ "$(ray status 2>/dev/null | grep -c '^ [0-9]* node_')" -ge 2 ] 2>/dev/null; do
    sleep 5
    echo "... Waiting for worker node (Current node count: $(ray status 2>/dev/null | grep -c '^ [0-9]* node_'))"
done
echo "Ray Cluster is fully formed!"
ray status

export NCCL_COMM_TIMEOUT=60

# --- 5. Launch vLLM Prefill Service ---
echo "Launching vLLM Prefill Service on $PREFILL_IP..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --port 8100 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.8 \
    --quantization awq \
    --no-enable-chunked-prefill \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":1e9,\"kv_connector_extra_config\":{\"mem_pool_size_gb\":4}}" > prefill_server.log 2>&1 &

# Wait for local server
echo "Waiting for local Prefill server at http://localhost:8100..."
until curl -s "http://localhost:8100/v1/completions" > /dev/null; do
    sleep 5
    echo "Still waiting..."
done
echo "Prefill server is ready."

# Wait for remote server (manual check)
echo "Checking if Decode server is ready at http://$DECODE_IP:8200..."
until curl -s "http://$DECODE_IP:8200/v1/completions" > /dev/null; do
    echo "Waiting for Decode Node ($DECODE_IP:8200). Please ensure decode_node_setup.sh is running there."
    sleep 10
done
echo "Decode server is ready."

cd "$(dirname "$0")"
# --- 6. Start Disagg Proxy ---
echo "Starting Disagg Proxy..."
python3 disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url "http://localhost:8100" \
    --decode-url "http://${DECODE_IP}:8200" \
    --prefill-kv-host "$PREFILL_IP" \
    --decode-kv-host "$DECODE_IP" \
    --prefill-kv-port 14579 \
    --decode-kv-port 14580 > proxy.log 2>&1 &

sleep 5

# --- 7. Benchmark Loop ---
LAMBDAS=(0.5 1 2 4 8)
CASES=("case1" "case2")

for case in "${CASES[@]}"; do
    input_len=512; output_len=128
    if [ "$case" == "case2" ]; then
        input_len=128; output_len=512
    fi

    echo "========================================"
    echo "Running Workload: $case (Input: $input_len, Output: $output_len)"
    echo "========================================"

    for lambda in "${LAMBDAS[@]}"; do
        echo "Running with Lambda: $lambda"
        
        vllm bench serve \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 0.0 \
            --num-prompts 2000 \
            --request-rate "$lambda" \
            --seed 42 \
            --port 8000 \
            --save-result \
            --result-dir "$RESULT_DIR" \
            --result-filename "${case}_lambda${lambda}.json"
        
        echo "Finished Lambda $lambda"
    done
done

echo "Benchmarks completed successfully."
echo "Results stored in: $RESULT_DIR"

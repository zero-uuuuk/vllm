#!/bin/bash

# --- Decode Node Setup Script ---
# Run this on the Worker node (g4dn instance).

set -e

# --- Configuration ---
# PREFILL_IP must be the actual IP of the Master node
PREFILL_IP=${PREFILL_IP:-"127.0.0.1"}
MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

# --- 1. Environment & Dependency Setup ---
echo "Checking dependencies..."
(which jq) || (sudo apt-get -y install jq)
(which socat) || (sudo apt-get -y install socat)
uv pip install ray msgpack

# --- 2. Cleanup Logic ---
cleanup() {
    echo "Cleaning up local processes..."
    pkill -f "vllm serve" || true
    ray stop || true
}
trap cleanup EXIT

# --- 3. Connect to Ray Cluster ---
echo "Connecting to Ray Head at $PREFILL_IP:6379..."
ray start --address "$PREFILL_IP:6379"
echo "Successfully requested connection to Ray Cluster."

export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1

# --- 4. Launch vLLM Decode Service ---
# vLLM Decode 서비스를 기동합니다. (KV Cache Consumer 역할)
echo "Launching vLLM Decode Service on port 8200..."
# 참고: 2-노드 설정에서 Rank 1이 Consumer 역할을 수행합니다.

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
    --port 8200 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.7 \
    --quantization awq \
    --no-enable-chunked-prefill \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_rank\":1,\"kv_parallel_size\":2,\"kv_buffer_size\":1e9,\"kv_ip\":\"${PREFILL_IP}\",\"kv_port\":14580,\"kv_connector_extra_config\":{\"mem_pool_size_gb\":8}}" > decode_server.log 2>&1 &

echo "-------------------------------------------------------"
echo "Decode Service is running in the background."
echo "Monitoring logs: tail -f decode_server.log"
echo "-------------------------------------------------------"

# 서비스가 실행되는 동안 스크립트 유지
wait

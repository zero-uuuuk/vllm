# Troubleshooting

## 1. 의존성 누락 (ModuleNotFoundError)

vLLM 기본 설치에는 disagg 벤치마크에 필요한 패키지가 포함되어 있지 않음.

```bash
uv pip install quart aiohttp ray httpx datasets msgpack
```

> [!NOTE]
> `prefill_node_setup.sh`를 사용하면 스크립트 상단에서 자동으로 설치함. 수동 실행 시에는 위 명령을 먼저 실행할 것.

## 2. 단일 노드 스크립트를 멀티 노드로 확장

기존 `disagg_overhead_benchmark.sh`, `disagg_performance_benchmark.sh`는 단일 EC2에서 `CUDA_VISIBLE_DEVICES`로 Prefill/Decode를 분리하는 구조였음. 서로 다른 EC2 인스턴스에서 실행하려면 두 가지를 수정해야 했음.

### Ray 클러스터 연결

두 노드가 서로를 인식할 수 있도록 Ray 클러스터를 구성해야 함.

- Prefill 노드: `ray start --head --node-ip-address $PREFILL_IP --port 6379`
- Decode 노드: `ray start --address $PREFILL_IP:6379`

### Proxy 인자 수정

기존 프록시는 `--kv-host localhost` 단일 호스트만 지원했음. Prefill과 Decode가 서로 다른 IP를 가지므로, 각각을 별도로 지정할 수 있도록 `--prefill-kv-host`, `--decode-kv-host` 인자를 추가함.

```bash
python3 disagg_prefill_proxy_server.py \
    --prefill-url "http://localhost:8100" \
    --decode-url "http://${DECODE_IP}:8200" \
    --prefill-kv-host "$PREFILL_IP" \
    --decode-kv-host "$DECODE_IP"
```

기존 `--kv-host` 인자는 하위 호환을 위해 유지됨. 인자를 아무것도 지정하지 않으면 각 서비스 URL에서 hostname을 추출하여 사용함.

### Decode 노드 join 대기

Prefill 노드가 Ray Head를 띄운 직후 바로 vLLM을 기동하면, Decode 노드가 아직 클러스터에 합류하지 않은 상태에서 실행이 진행됨. Prefill 노드에서 Decode 노드가 join할 때까지 명시적으로 대기해야 함.

```bash
until [ "$(ray status 2>/dev/null | grep -c '^ [0-9]* node_')" -ge 2 ]; do
    sleep 5
done
```

이 루프가 없으면 Decode 서버가 준비되기 전에 프록시와 벤치마크가 먼저 실행되어 연결에 실패함.

### kv-transfer-config IP/포트 설정

단일 노드에서는 `kv_ip`를 신경 쓸 필요가 없었지만, 멀티 노드에서는 각 서버가 상대 노드의 실제 IP를 명시해야 함.

| 서버 | `kv_ip` | `kv_port` | 의미 |
|------|---------|-----------|------|
| Prefill (`kv_producer`) | `$DECODE_IP` | 14579 | KV를 **보낼** 대상 |
| Decode (`kv_consumer`) | `$PREFILL_IP` | 14580 | KV를 **받을** 소켓 바인딩 주소 |

Prefill 설정 예시:

```bash
--kv-transfer-config \
"{\"kv_connector\":\"BatchedNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":2.5e9,\"kv_ip\":\"${DECODE_IP}\",\"kv_port\":14579,\"kv_connector_extra_config\":{\"mem_pool_size_gb\":8,\"kv_granularity\":${KV_GRANULARITY}}}"
```

`kv_ip`에 `localhost`나 `127.0.0.1`을 그대로 두면 상대 노드에 도달하지 못해 KV 전송이 무한 대기 상태에 빠짐.

## 3. Request ID 불일치로 Decode가 KV Cache를 인식하지 못함

vLLM v1 엔진은 내부적으로 모든 `request_id`에 랜덤 suffix를 추가함. Prefill과 Decode 서버가 각각 독립적으로 suffix를 생성하면 KV Cache의 `tensor_id`(`request_id#layer_N`)가 불일치하여 Decode가 해당 KV Cache를 찾지 못하고 타임아웃됨.

**해결**: 양쪽 vLLM 서버 기동 전에 환경변수를 설정함.

```bash
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
```

이 변수가 없으면 Decode 서버의 `recv_tensor()`가 일치하는 tensor_id를 찾지 못해 KV 전송이 실패함.

## 4. gpu-memory-utilization 0.8에서 OOM 발생

`--gpu-memory-utilization 0.8`로 설정하면 런타임에 OOM이 발생함.

`gpu_memory_utilization`은 vLLM이 점유할 GPU 메모리 비율로, 모델 가중치/CUDA graph/KV cache 페이지 풀이 모두 이 안에 포함됨. 그런데 `P2pNcclEngine`의 `recv_store` 버퍼(`kv_buffer_size=2.5e9`, 2.5GB)는 이 계산 **밖**에서 런타임에 동적으로 GPU 메모리를 추가 점유함.

```
실제 GPU 사용량 = vLLM (gpu_memory_utilization%) + recv_store 버퍼 (최대 2.5GB)
```

**해결**: `--gpu-memory-utilization 0.7`으로 낮춰 recv_store 버퍼가 올라올 여유 공간을 확보함.

## 5. vLLM은 이미 layer-wise 전송이었음

실험 설계 당시 layer-wise KV 전송을 직접 구현해야 한다고 가정했으나, vLLM의 `P2pNcclConnector`는 이미 레이어 단위로 전송하는 구조였음.

Prefill 서버는 forward pass 중 레이어를 실행할 때마다 `save_kv_layer()`가 호출되어 해당 레이어의 KV Cache를 즉시 전송함. 즉, granularity=1(layer-wise)이 기본 동작임.

이를 기반으로 `BatchedNcclConnector`를 구현하여 granularity=1/8/32를 설정할 수 있도록 확장함.

## 6. CPU Fallback으로 인한 TPOT 노이즈 — Lambda 상한 제한

`recv_store` 누적 크기가 `kv_buffer_size`를 초과하면 KV Cache가 CPU 메모리 풀로 fallback됨. 이 경우 CPU↔GPU 복사 오버헤드가 발생하여 TPOT가 부하와 무관하게 튀어, saturation point를 찾기 어려워짐.

**해결**: `kv_buffer_size`를 `2.5e9`로 늘리고, lambda를 **2 이하**로 제한하여 fallback이 발생하지 않는 부하 범위 내에서 실험함.

> [!NOTE]
> `mem_pool_size_gb: 8`은 fallback 풀의 크기를 늘려 OOM을 방지하지만, fallback 자체를 막지는 않음. 근본적인 해결은 부하를 낮추거나 `kv_buffer_size`를 늘리는 것.

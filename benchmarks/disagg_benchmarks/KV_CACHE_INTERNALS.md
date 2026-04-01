# KV Cache 전송 내부 동작

vLLM P/D Disaggregation 환경에서 KV Cache가 어떻게 생성되고, 전송되고, 수신되는지를 설명함.

> [!NOTE]
> `NODE_SETUP.md`, `TROUBLE_SHOOTING.md`와 일부 내용이 겹칠 수 있음. 이 문서는 KV Cache 전송 메커니즘 자체에 집중하며, 실행 방법은 `NODE_SETUP.md`, 실험 중 발견한 문제는 `TROUBLE_SHOOTING.md`를 참고.

---

## 전체 흐름

```
┌─────────┐   HTTP    ┌───────┐   HTTP (max_tokens=1)   ┌─────────────────┐
│ Client  │ ────────▶ │ Proxy │ ──────────────────────▶ │ Prefill Server  │
└─────────┘           │       │                         │                 │
                      │       │   HTTP (원본 요청)        │  forward pass   │
                      │       │ ──────────────────────▶ │  save_kv_layer()│
                      └───────┘                         └────────┬────────┘
                           ▲                                     │ NCCL
                           │ streaming tokens                    ▼
                      ┌────┴──────────────────────────────────────────────┐
                      │                  Decode Server                    │
                      │  listen_for_requests() ──▶ recv_store             │
                      │  start_load_kv()       ──▶ inject into KV buffer  │
                      │  forward pass          ──▶ token generation       │
                      └───────────────────────────────────────────────────┘
```

> **핵심**: HTTP 요청/응답은 Proxy를 경유하지만, KV Cache 데이터는 Prefill → Decode 간 **NCCL로 직접** 전송됨. Proxy는 조율자 역할만 함.

---

## 1. Proxy가 필요한 이유

Prefill 서버와 Decode 서버는 각각 독립적인 vLLM 인스턴스로, 서로의 존재를 모름. Proxy가 세 가지를 처리함.

**① 순서 제어**

Decode 서버에 요청이 먼저 도착하면 KV Cache가 없으므로 `recv_tensor()`가 타임아웃까지 블로킹됨. Proxy가 Prefill 완료 후에만 Decode에 요청을 전달함.

```
Proxy → Prefill (max_tokens=1) → 완료 대기
      → Decode  (원본 요청)    → 스트리밍 relay
```

**② `X-KV-Target` 헤더 주입**

Prefill 서버가 KV Cache를 어느 Decode 노드로 보낼지 알 수 있도록, Proxy가 헤더를 주입함.

```
X-KV-Target: {DECODE_IP}:{port}
```

이 헤더가 없으면 Prefill 서버는 KV Cache 전송 대상을 알 수 없음.

**③ `X-Request-Id` 인코딩**

Prefill과 Decode가 동일한 tensor_id를 사용하도록, ZMQ/NCCL 주소를 request_id에 인코딩함.

```
X-Request-Id: ___prefill_addr_10.0.x.1:14579___decode_addr_10.0.x.2:14580_<uuid>
```

---

## 2. Prefill 서버: KV Cache 생성 및 전송

### 전송 시점

Forward pass 중 레이어를 실행할 때마다 `save_kv_layer()`가 호출됨. 레이어 0 완료 → 즉시 전송 시작 → 레이어 1 연산과 병렬로 전송됨.

```
Layer 0 연산 ──▶ save_kv_layer() ──▶ send_queue에 enqueue ──┐
Layer 1 연산 ──▶ save_kv_layer() ──▶ send_queue에 enqueue   │ (별도 스레드)
...                                                          ▼
                                                  send_async 스레드:
                                                    ZMQ: PUT 메시지 전송
                                                    ZMQ: Decode 준비 신호 대기
                                                    NCCL: 텐서 데이터 전송
```

`send_type=PUT_ASYNC`(기본값)이므로 enqueue 후 즉시 반환. 실제 전송은 별도 스레드가 처리함.

### HTTP 응답 반환 시점

`wait_for_save()` → send_queue가 빌 때까지 대기 → HTTP 응답 반환.
즉, **모든 레이어의 전송이 완료된 후** Proxy에 응답이 돌아옴.

---

## 3. Decode 서버: KV Cache 수신

### listen_for_requests 스레드 (상시 대기)

```
while True:
    ZMQ: PUT 메시지 수신 (tensor_id, shape, dtype)
    GPU 메모리 할당 (torch.empty)
    ZMQ: "0" 응답 → Prefill에게 준비됐다고 알림
    NCCL: 텐서 수신 (블로킹)
    recv_store[tensor_id] = tensor
    recv_store_cv.notify()
```

단일 스레드이므로 레이어를 순차적으로 수신함.

### start_load_kv (forward pass 직전)

```
for request in batch:
    for layer in all_layers:
        tensor_id = request_id + "#" + layer_name
        kv = recv_tensor(tensor_id)   ← recv_store에 없으면 최대 300초 블로킹 대기
        inject_kv_into_paged_buffer(layer, kv, block_ids)
```

모든 레이어의 KV Cache가 수신될 때까지 GPU 워커가 블로킹됨.

> [!WARNING]
> **Head-of-Line Blocking**: 배치 내 요청 1개의 KV 전송이 늦어지면, 이미 디코딩 중이던 나머지 모든 요청의 연산도 함께 멈춤. 네트워크 지연 시 GPU utilization이 급락하는 원인.

---

## 4. tensor_id 매칭

Prefill과 Decode가 동일한 tensor_id를 사용해야 KV Cache가 올바르게 매칭됨.

```
tensor_id = request_id + "#" + layer_name

예시:
  cmpl-___prefill_addr_10.0.x.1:14579___decode_addr_10.0.x.2:14580_abc123#model.layers.0.self_attn.attn
```

> [!WARNING]
> vLLM v1 엔진은 request_id에 랜덤 suffix를 추가함. Prefill/Decode가 각각 독립적으로 suffix를 생성하면 tensor_id가 불일치하여 `recv_tensor()`가 타임아웃됨.
>
> **해결**: `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1`

---

## 5. 메모리 구조

```
Decode 서버 GPU 메모리
┌──────────────────────────────────────────────────────────┐
│   vLLM 점유 영역 (gpu_memory_utilization=0.7 → 70%)      │
│  ┌───────────────┬──────────────┬───────────────────────┐ │
│  │  모델 가중치  │  CUDA graph  │    KV cache pages     │ │
│  └───────────────┴──────────────┴───────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│   recv_store 버퍼 (kv_buffer_size=1e9, ~1GB)             │ ← vLLM 계산 밖
│   런타임에 동적 할당                                       │
└──────────────────────────────────────────────────────────┘
```

`recv_store` 누적 크기가 `kv_buffer_size`를 초과하면 CPU 메모리 풀(`mem_pool_size_gb`)로 fallback됨. CPU↔GPU 복사 오버헤드가 발생하여 TPOT가 튐.

`gpu_memory_utilization=0.8`로 설정하면 vLLM 80% + recv_store 1GB가 합산되어 OOM 발생. **0.7로 설정하여 여유 공간 확보.**

---

## 6. Chunked Prefill 비활성화 이유

Chunked prefill이 활성화되면 KV 전송이 **마지막 chunk 완료 후**에만 시작됨.

```
Chunked prefill ON:
  chunk 1 처리 → wait_for_save() → 전송할 것 없음
  chunk 2 처리 → wait_for_save() → 전송할 것 없음
  chunk N 처리 → wait_for_save() → 전체 KV 전송 시작  ← 지연 발생

Chunked prefill OFF:
  전체 프롬프트 처리 → 레이어 0 완료 즉시 전송 시작  ← 즉시 파이프라이닝
```

`--no-enable-chunked-prefill`로 비활성화하면 KV 전송이 즉시 시작되고 동작이 단순해짐.

---

## 7. 알려진 한계

| 항목 | 현재 구현 | 개선 방향 |
|------|-----------|-----------|
| 동기화 방식 | 전체 레이어 수신 후 연산 시작 (Blocking) | 레이어 수신과 연산 파이프라이닝 |
| Head-of-Line Blocking | KV 전송 지연 시 배치 전체 멈춤 | KV 준비된 요청만 스케줄링 |
| buffer_size 버그 | 완료된 요청의 GPU 텐서 해제 시 `buffer_size` 카운터 미감소 → 이후 요청이 모두 CPU fallback | 수정 필요 |

---

## 관련 소스

| 파일 | 역할 |
|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` | `save_kv_layer`, `start_load_kv` |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py` | `send_async`, `listen_for_requests`, `recv_tensor` |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/tensor_memory_pool.py` | CPU fallback 메모리 풀 |
| `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | Proxy 구현 |

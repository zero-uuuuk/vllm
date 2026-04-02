# KV Transfer Time 측정 방식

KV Cache 전송 시간을 송신/수신 양쪽에서 완료 시각(Unix timestamp)으로 기록한다. 같은 `tensor_id`의 `recv ts - send ts`로 전송 시간을 계산한다.

## 로그 형식

```
⏱️KV send: ts=<unix_ts>, tensor_id:<id>, rank:<n>
⏱️KV recv: ts=<unix_ts>, tensor_id:<id>
```

- `ts`: `time.time()` (Unix epoch 초)

## 로그 포인트

### 송신 측: `⏱️KV send`

- 파일: `p2p_nccl_engine.py` — `send_sync()`
- 시점: ncclSend **시작 직전** (ZMQ 핸드셰이크 완료 후, 실제 GPU 전송 시작 전)
- PUT/PUT_ASYNC 모두 커버 (PUT_ASYNC는 백그라운드 스레드에서 `send_sync()`를 호출하므로)
- **주의**: 엔진 레벨 로그이므로 P2pNcclConnector, BatchedNcclConnector 등 P2pNcclEngine을 사용하는 모든 커넥터에서 찍힌다.

### 수신 측: `⏱️KV recv`

- 시점: `recv_tensor()` 리턴 직후 (텐서가 GPU에 도착 완료)

| granularity | 파일 | 메서드 | 로그 단위 |
|---|---|---|---|
| ≤ 1 | `p2p_nccl_connector.py` | `start_load_kv()` | 레이어 1개 도착마다 |
| > 1 | `batched_nccl_connector.py` | `start_load_kv()` | 배치(N개 레이어 묶음) 도착마다 |

## 전송 시간 계산

같은 `tensor_id`의 `recv ts - send ts` = **전송 시간** (보내기 시작 → 받기 완료)

> **주의**: 노드 간 시각 비교는 NTP 동기화 정확도에 의존한다.

## 흐름도

```
Prefill (Producer)                              Decode (Consumer)
──────────────────                              ──────────────────
save_kv_layer()                                 start_load_kv()
  |                                               |
  +-- granularity <= 1:                           +-- granularity <= 1:
  |     send_tensor() per layer                   |     recv_tensor() per layer
  |                                               |     -> KV recv (per layer)
  +-- granularity > 1:                            |
  |     buffer layers                             +-- granularity > 1:
  |     -> _flush_concat()                        |     recv_tensor() per batch
  |          concat + send_tensor()               |     -> KV recv (per batch)
  |                                               |
  v                                               |
  send_tensor()                                   |
  +-- PUT:       send_sync() directly             |
  +-- PUT_ASYNC: enqueue, return immediately      |
  |              -> background thread             |
  |                 calls send_sync()             |
  v                                               |
  send_sync()                                     |
  +-- ZMQ handshake                               |
  +-- KV send (timestamp logged)                  |
  +-- ncclSend + stream.synchronize()             |
  +-- done                                        |
```

## 예시 로그 (granularity=8, 32 레이어 모델)

```
# Prefill 측 (4번 send)
⏱️KV send: ts=1712345678.100, tensor_id:req123#batch_0, rank:0
⏱️KV send: ts=1712345678.145, tensor_id:req123#batch_1, rank:0
⏱️KV send: ts=1712345678.190, tensor_id:req123#batch_2, rank:0
⏱️KV send: ts=1712345678.235, tensor_id:req123#batch_3, rank:0

# Decode 측 (4번 recv)
⏱️KV recv: ts=1712345678.142, tensor_id:req123#batch_0
⏱️KV recv: ts=1712345678.187, tensor_id:req123#batch_1
⏱️KV recv: ts=1712345678.232, tensor_id:req123#batch_2
⏱️KV recv: ts=1712345678.277, tensor_id:req123#batch_3
```

### 읽는 법

batch_0: 1712345678.142 - 1712345678.100 = **0.042s** (보내기 시작 → 받기 완료)
batch_1: 1712345678.187 - 1712345678.145 = **0.042s**

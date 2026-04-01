# Task 06: Disaggregated Prefill (Single Node)

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> Disaggregated Prefill 환경은 설정 오류가 많고 디버깅이 어렵다.
> 오류 발생 시 [`TROUBLE_SHOOTING.md`](../benchmarks/disagg_benchmarks/TROUBLE_SHOOTING.md)를 먼저 확인할 것.

## TODO

- [ ] Step 1: Prefill 서버 기동 (`CUDA_VISIBLE_DEVICES=0,1`, TP=2, kv_producer, port 8100)
- [ ] Step 2: Decode 서버 기동 (`CUDA_VISIBLE_DEVICES=2,3`, TP=2, kv_consumer, port 8200)
- [ ] Step 3: Proxy 서버 기동 (`disagg_prefill_proxy_server.py`, port 8000)
- [ ] Step 4: `vllm bench serve`로 성능 측정 + task-05 TP=2 결과와 TTFT/TPOT 비교
- [ ] `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`, `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` 소스 직접 읽기

---

## 목표

단일 노드(g4dn.12xlarge, GPU 4장)에서 `CUDA_VISIBLE_DEVICES`로 GPU를 분리하여 Prefill/Decode disaggregation을 실습한다.
GPU 2장을 Prefill(TP=2), 나머지 2장을 Decode(TP=2)로 구성하고, task-05의 TP=2 단일 서버 대비 TTFT/TPOT 변화를 확인한다.

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Disaggregated Prefill** | Prefill 단계와 Decode 단계를 별도 서버(프로세스)로 분리하여 각각 독립 스케일링 |
| **TTFT 지배 요인** | TTFT(Time To First Token)는 prefill 연산 시간에 지배됨 → Prefill 서버만 스케일업 가능 |
| **TPOT 지배 요인** | TPOT(Time Per Output Token)는 decode 연산 시간에 지배됨 → Decode 서버만 스케일업 가능 |
| **KV Cache 전송** | Prefill 완료 후 생성된 KV Cache를 NCCL P2P로 Decode 서버에 직접 전송 |
| **Proxy 역할** | 클라이언트 요청을 받아 Prefill → Decode 순서로 라우팅, `X-KV-Target` 헤더 주입 |
| **kv_producer / kv_consumer** | Prefill 서버는 KV를 생성(producer), Decode 서버는 KV를 수신(consumer) |

> Disaggregated Prefill의 핵심 이점: TTFT가 긴 요청(긴 프롬프트)과 TPOT가 긴 요청(긴 출력)을
> 각각 독립적으로 최적화할 수 있다. 단일 서버에서는 두 단계가 같은 GPU를 공유하므로 트레이드오프가 발생한다.

---

## 단계별 실습

### Step 1: Prefill 서버 기동

GPU 0,1을 Prefill 서버에 할당한다. 별도 터미널에서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve facebook/opt-125m \
    --port 8100 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --no-enable-chunked-prefill \
    --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":5e9}'
```

살펴볼 것:
- 서버 로그에서 `kv_role=kv_producer`, `kv_rank=0` 확인
- `P2pNcclConnector` 초기화 로그 확인

---

### Step 2: Decode 서버 기동

GPU 2,3을 Decode 서버에 할당한다. 별도 터미널에서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1 \
vllm serve facebook/opt-125m \
    --port 8200 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --no-enable-chunked-prefill \
    --kv-transfer-config \
        '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":5e9}'
```

살펴볼 것:
- 서버 로그에서 `kv_role=kv_consumer`, `kv_rank=1` 확인
- Prefill 서버와 NCCL 연결이 수립되는 로그 확인

---

### Step 3: Proxy 서버 기동

Prefill/Decode 서버가 모두 준비된 후 Proxy를 기동한다. 별도 터미널에서 실행한다.

```bash
python3 benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url http://localhost:8100 \
    --decode-url http://localhost:8200
```

살펴볼 것:
- Proxy가 8000 포트에서 수신 대기하는지 확인
- 요청이 들어오면 Prefill → Decode 순으로 포워딩되는 로그 확인

---

### Step 4: 벤치마크 + task-05 결과 비교

세 서버가 모두 준비되면 Proxy(8000)를 통해 벤치마크를 실행한다.

```bash
mkdir -p ./results

vllm bench serve \
    --model facebook/opt-125m \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 5 \
    --port 8000 \
    --save-result \
    --result-dir ./results \
    --result-filename disagg_single_rps5.json
```

task-05 TP=2 결과와 비교한다.

```bash
for f in tp2_rps5.json disagg_single_rps5.json; do
    echo "=== $f ==="
    jq '{request_throughput, mean_ttft_ms, mean_tpot_ms}' ./results/$f
done
```

예상 결과 패턴 (모델/환경에 따라 다름):

| 구성 | request_throughput | mean_ttft_ms | mean_tpot_ms |
|------|--------------------|--------------|--------------|
| TP=2 단일 서버 (task-05) | 기준 | 기준 | 기준 |
| Disagg Prefill (단일 노드) | 유사 또는 감소 | 감소 가능 | 감소 가능 |

관찰 포인트:
- **TTFT**: Prefill 서버가 decode 부담 없이 prefill에만 집중하므로 감소 가능
- **TPOT**: Decode 서버가 prefill 부담 없이 decode에만 집중하므로 감소 가능
- **throughput**: 단일 노드에서는 GPU 수가 동일하므로 throughput 향상은 제한적. KV 전송 오버헤드가 추가됨
- opt-125m처럼 작은 모델은 KV 전송 오버헤드가 상대적으로 크게 느껴질 수 있음

---

## 코드 흐름 이해

> 클라이언트 요청이 Proxy → Prefill → (KV 전송) → Decode 순으로 처리된다.
> KV Cache는 NCCL P2P로 직접 전송되며, Proxy는 HTTP 레벨에서 순서를 제어한다.

```
클라이언트 → Proxy (port 8000)
    └─ disagg_prefill_proxy_server.py
           ├─ 1. Prefill 서버(8100)에 요청 전송 (X-KV-Target: decode 주소 헤더 포함)
           │       └─ p2p_nccl_connector.py → save_kv_layer()
           │              └─ KV Cache 생성 후 NCCL P2P로 Decode 서버에 전송
           └─ 2. Decode 서버(8200)에 요청 전송
                   └─ p2p_nccl_connector.py → start_load_kv()
                          └─ Prefill에서 전송된 KV Cache 수신 후 decode 진행
```

소스 읽기 시작점:
- `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` — `main()`: Prefill/Decode 순서 제어 및 헤더 주입 방식
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` — `save_kv_layer()`: KV Cache를 어떻게 전송하는지, `start_load_kv()`: KV Cache를 어떻게 수신하는지

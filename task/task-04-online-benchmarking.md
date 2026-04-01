# Task 04: Online Benchmarking

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: `vllm serve` 기동 + 헬스체크
- [ ] Step 2: `vllm bench serve` 기본 실행
- [ ] Step 3: `--request-rate` 실험 (낮은 값 vs 높은 값)
- [ ] Step 4: 결과 JSON으로 저장 후 주요 키 확인
- [ ] `vllm/benchmarks/serve.py`, `vllm/benchmarks/lib/endpoint_request_func.py` 소스 직접 읽기

---

## 목표

`vllm serve`로 HTTP 서버를 띄우고 `vllm bench serve`로 online serving 성능을 측정하는 방법을 익힌다.
TTFT/TPOT/ITL 지표의 의미를 이해하고, request rate 변화에 따른 성능 변화를 직접 확인한다.

---

## 핵심 개념

| 지표 | 의미 | 측정 방식 |
|------|------|-----------|
| **TTFT** (Time To First Token) | 요청 전송 후 첫 토큰이 도착하기까지의 시간 | 요청 시작 ~ 첫 청크 수신 |
| **TPOT** (Time Per Output Token) | 출력 토큰 한 개를 생성하는 데 걸리는 평균 시간 | (E2E latency - TTFT) / (출력 토큰 수 - 1) |
| **ITL** (Inter-Token Latency) | 연속된 두 토큰 사이의 간격 | 스트리밍 청크 간 시간 차이 |

> TTFT는 prefill 단계의 지연을 반영하고, TPOT/ITL은 decode 단계의 속도를 반영한다.
> offline throughput과 달리 online 벤치마크는 실제 HTTP 요청을 포아송 프로세스로 전송하므로 queueing 효과가 포함된다.

---

## 단계별 실습

### Step 1: vllm serve 기동 + 헬스체크

별도 터미널에서 서버를 기동한다.

```bash
# 서버 기동
vllm serve facebook/opt-125m \
    --port 8000

# 헬스체크 (서버가 준비될 때까지 대기)
curl localhost:8000/health
# 정상 응답: 200 OK
```

살펴볼 것:
- 서버 로그에서 모델 로딩 완료 시점 확인
- `/v1/models` 엔드포인트로 로드된 모델 목록 확인 (`curl localhost:8000/v1/models`)

---

### Step 2: vllm bench serve 기본 실행

서버가 기동된 상태에서 다른 터미널에서 실행한다.

```bash
# 기본 실행 (random dataset, 100 요청)
vllm bench serve \
    --model facebook/opt-125m \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 100 \
    --port 8000

# 인자 목록 확인
vllm bench serve --help
```

주요 출력:
```
Traffic request rate: inf
...
Successful requests: 100
Benchmark duration: X.XX s
Total input tokens: XXXXX
Total generated tokens: XXXXX
Request throughput: XX.XX req/s
Output token throughput: XXXXX.XX tok/s
Total Token throughput: XXXXX.XX tok/s
Mean TTFT (ms): XX.XX
Median TTFT (ms): XX.XX
P99 TTFT (ms): XX.XX
Mean TPOT (ms): XX.XX
Median TPOT (ms): XX.XX
P99 TPOT (ms): XX.XX
Mean ITL (ms): XX.XX
Median ITL (ms): XX.XX
P99 ITL (ms): XX.XX
```

살펴볼 것:
- `--request-rate`를 지정하지 않으면 기본값 `inf` (모든 요청을 즉시 전송)
- `inf` 상태는 offline throughput 측정과 유사한 조건

---

### Step 3: --request-rate 실험

request rate를 조절해 queueing 효과와 latency 변화를 관찰한다.

```bash
# 낮은 request rate (서버에 여유 있음)
vllm bench serve \
    --model facebook/opt-125m \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 2 \
    --port 8000

# 높은 request rate (서버 포화 근접)
vllm bench serve \
    --model facebook/opt-125m \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts 100 \
    --request-rate 20 \
    --port 8000
```

살펴볼 것:
- request rate가 낮을 때: TTFT/TPOT 낮음, throughput도 낮음
- request rate가 높을 때: throughput 증가, TTFT/TPOT도 증가 (queueing 지연)
- offline throughput(`vllm bench throughput`)과 비교 — online은 queueing 효과가 포함되어 실제 서비스 조건에 더 가깝다

---

### Step 4: 결과 JSON 저장 및 주요 키 확인

```bash
# 결과를 JSON으로 저장
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
    --result-filename serve_rps5.json
```

JSON 주요 키:
```json
{
    "request_throughput": 5.0,
    "output_token_throughput": 640.0,
    "mean_ttft_ms": 45.2,
    "median_ttft_ms": 42.1,
    "p99_ttft_ms": 98.3,
    "mean_tpot_ms": 12.4,
    "median_tpot_ms": 11.8,
    "p99_tpot_ms": 28.6,
    "mean_itl_ms": 12.4,
    "median_itl_ms": 11.9,
    "p99_itl_ms": 27.1,
    "mean_e2el_ms": 1620.5
}
```

| 키 | 설명 |
|----|------|
| `request_throughput` | 초당 처리 요청 수 |
| `output_token_throughput` | 초당 생성 토큰 수 |
| `mean_ttft_ms` | 평균 TTFT (ms) |
| `p99_ttft_ms` | 99th percentile TTFT (ms) |
| `mean_tpot_ms` | 평균 TPOT (ms) |
| `p99_tpot_ms` | 99th percentile TPOT (ms) |
| `mean_itl_ms` | 평균 ITL (ms) |
| `mean_e2el_ms` | 평균 end-to-end latency (ms) |

---

## 코드 흐름 이해

> `vllm bench serve`는 실제 HTTP 서버에 요청을 전송하고 응답 스트림을 파싱해 지표를 계산한다.
> 요청은 포아송 프로세스(`--request-rate`)로 전송되어 실제 트래픽 패턴을 시뮬레이션한다.

```
vllm bench serve
    └─ vllm/entrypoints/cli/benchmark/serve.py  →  BenchmarkServeSubcommand
           └─ vllm/benchmarks/serve.py  →  main()
                  └─ benchmark()
                       ├─ 데이터셋 로드 + 요청 샘플링
                       ├─ asyncio로 포아송 간격 요청 전송
                       └─ get_request_func()  →  endpoint_request_func.py
                              └─ 실제 HTTP 요청 전송 + 스트리밍 응답 파싱
                                   → TTFT, ITL 타임스탬프 기록
```

소스 읽기 시작점:
- `vllm/benchmarks/serve.py` — `benchmark()`, `main()` 함수: 요청 스케줄링과 지표 집계 방식
- `vllm/benchmarks/lib/endpoint_request_func.py` — 실제 HTTP 요청 전송 및 스트리밍 청크에서 TTFT/ITL 측정 방식

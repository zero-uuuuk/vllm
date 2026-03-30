# Task 03: Offline Benchmarking

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: `vllm bench latency` 기본 실행
- [ ] Step 2: `vllm bench throughput` 기본 실행
- [ ] Step 3: 결과 JSON으로 저장 후 비교
- [ ] `vllm/benchmarks/latency.py`, `throughput.py` 소스 직접 읽기

---

## 목표

vLLM의 내장 벤치마크 CLI를 사용해 모델의 **latency**와 **throughput**을 측정하는 방법을 익힌다.
두 지표의 차이를 이해하고, 파라미터 변화에 따른 성능 변화를 직접 확인한다.

---

## 핵심 개념

| 지표 | 의미 | 측정 방식 |
|------|------|-----------|
| **Latency** | 단일 배치 처리에 걸리는 시간 | 고정 배치를 반복 실행, 평균/백분위 측정 |
| **Throughput** | 단위 시간당 처리 토큰/요청 수 | 전체 요청을 처리하는 총 시간으로 계산 |

> 두 지표는 트레이드오프 관계다.
> 배치 크기를 키우면 throughput은 올라가지만 latency도 늘어난다.

**참고**: `benchmarks/benchmark_latency.py`, `benchmarks/benchmark_throughput.py`는 deprecated.
현재는 `vllm bench` CLI로 통합되었다.

---

## 단계별 실습

### Step 1: Latency 측정

단일 배치를 반복 실행해 처리 시간의 평균과 백분위를 측정한다.

```bash
# 기본 실행 (opt-125m, input 32 tokens, output 128 tokens, batch 8)
vllm bench latency \
    --model facebook/opt-125m \
    --input-len 32 \
    --output-len 128 \
    --batch-size 8

# 인자 목록 확인
vllm bench latency --help
```

주요 출력:
```
Avg latency: X.XX seconds
50% percentile latency: X.XX seconds
90% percentile latency: X.XX seconds
99% percentile latency: X.XX seconds
```

살펴볼 것:
- warmup iteration이 왜 필요한가 (`--num-iters-warmup`)
- `--num-iters` 횟수가 결과 안정성에 미치는 영향
- batch-size를 1, 8, 32로 바꿔가며 latency 변화 확인

---

### Step 2: Throughput 측정

전체 요청 집합을 처리하는 총 시간으로 초당 토큰 수를 계산한다.

```bash
# 랜덤 데이터셋으로 기본 실행
vllm bench throughput \
    --model facebook/opt-125m \
    --dataset-name random \
    --num-prompts 100 \
    --input-len 128 \
    --output-len 128

# 인자 목록 확인
vllm bench throughput --help
```

주요 출력:
```
Throughput: XX.XX requests/s, XX.XX output tokens/s, XX.XX total tokens/s
```

살펴볼 것:
- `--num-prompts`를 늘릴수록 throughput이 어떻게 변하는가
- latency 결과와 비교했을 때 배치 크기의 영향 차이

---

### Step 3: 결과 JSON 저장 및 비교

```bash
# latency 결과 저장
vllm bench latency \
    --model facebook/opt-125m \
    --batch-size 1 \
    --output-json results_latency_bs1.json

vllm bench latency \
    --model facebook/opt-125m \
    --batch-size 16 \
    --output-json results_latency_bs16.json
```

JSON 구조:
```json
{
    "avg_latency": 0.123,
    "latencies": [...],
    "percentiles": {"10": ..., "50": ..., "90": ..., "99": ...}
}
```

---

## 코드 흐름 이해

> Task 02에서 직접 호출한 `LLM.generate()`가 벤치마크 내부에서도 동일하게 사용된다.
> `vllm bench latency/throughput`은 그 위에 시간 측정과 반복 실행 로직을 얹은 것이다.

```
vllm bench latency
    └─ vllm/entrypoints/cli/benchmark/latency.py  →  BenchmarkLatencySubcommand
           └─ vllm/benchmarks/latency.py  →  main()
                  └─ LLM.generate() 반복 실행 + 시간 측정

vllm bench throughput
    └─ vllm/entrypoints/cli/benchmark/throughput.py  →  BenchmarkThroughputSubcommand
           └─ vllm/benchmarks/throughput.py  →  main()
                  └─ 데이터셋 로드 → run_vllm() → 총 처리 시간 / 총 토큰 수
```

소스 읽기 시작점:
- `vllm/benchmarks/latency.py` — `add_cli_args()`, `main()` 함수
- `vllm/benchmarks/throughput.py` — `run_vllm()`, 데이터셋 처리 방식

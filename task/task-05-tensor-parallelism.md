# Task 05: Tensor Parallelism

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: TP=1로 `vllm serve` 기동 + `vllm bench serve` 기준 측정
- [ ] Step 2: TP=2로 서버 재기동 + 동일 벤치마크
- [ ] Step 3: TP=4로 서버 재기동 + 동일 벤치마크
- [ ] Step 4: throughput/TTFT/TPOT 변화 비교 및 관찰 포인트 정리
- [ ] `vllm/distributed/parallel_state.py`, `vllm/model_executor/layers/linear.py` 소스 직접 읽기

---

## 목표

`--tensor-parallel-size` 옵션으로 TP=1/2/4 환경을 구성하고 `vllm bench serve`로 성능 변화를 측정한다.
Tensor Parallelism이 레이어를 GPU 간 어떻게 분할하는지 이해하고, GPU 수 증가에 따른 throughput/latency 트레이드오프를 직접 확인한다.

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Tensor Parallelism (TP)** | 단일 레이어의 weight matrix를 여러 GPU에 분산하여 병렬 연산 |
| **Attention head 분할** | `num_heads`를 GPU 수로 나눠 각 GPU가 일부 head만 담당 (`num_heads / tp_size` per GPU) |
| **ColumnParallelLinear** | MLP의 첫 번째 projection — weight를 column 방향으로 분할, 각 GPU가 출력의 일부를 계산 |
| **RowParallelLinear** | MLP의 두 번째 projection — weight를 row 방향으로 분할, 계산 후 AllReduce로 합산 |
| **AllReduce** | 각 레이어 끝에서 GPU 간 부분 결과를 합산하는 집합 통신 연산 |
| **--tensor-parallel-size** | vllm serve에 전달하는 TP 크기 인자 (기본값: 1) |

> TP는 단일 레이어를 쪼개므로 GPU 간 통신(AllReduce)이 매 레이어마다 발생한다.
> GPU 수가 늘수록 연산은 빨라지지만 통신 오버헤드도 증가한다.

---

## 단계별 실습

### Step 1: TP=1 기준 측정

별도 터미널에서 서버를 기동한다.

```bash
# TP=1 서버 기동 (기본값이므로 생략 가능)
vllm serve facebook/opt-125m \
    --tensor-parallel-size 1 \
    --port 8000

# 헬스체크
curl localhost:8000/health
```

서버가 준비되면 다른 터미널에서 벤치마크를 실행한다.

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
    --result-filename tp1_rps5.json
```

살펴볼 것:
- 서버 로그에서 `tensor_parallel_size=1` 확인
- 결과 JSON의 `request_throughput`, `mean_ttft_ms`, `mean_tpot_ms` 기록

---

### Step 2: TP=2 서버 재기동 + 벤치마크

기존 서버를 종료하고 TP=2로 재기동한다.

```bash
# TP=2 서버 기동
vllm serve facebook/opt-125m \
    --tensor-parallel-size 2 \
    --port 8000
```

```bash
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
    --result-filename tp2_rps5.json
```

살펴볼 것:
- 서버 로그에서 GPU 2장이 초기화되는 과정 확인
- TP=1 대비 TTFT/TPOT 변화 방향 확인

---

### Step 3: TP=4 서버 재기동 + 벤치마크

```bash
# TP=4 서버 기동 (g4dn.12xlarge: T4 GPU 4장)
vllm serve facebook/opt-125m \
    --tensor-parallel-size 4 \
    --port 8000
```

```bash
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
    --result-filename tp4_rps5.json
```

살펴볼 것:
- GPU 4장 모두 사용되는지 `nvidia-smi`로 확인
- opt-125m처럼 작은 모델은 TP=4에서 통신 오버헤드가 연산 이득을 상쇄할 수 있음

---

### Step 4: 결과 비교 및 관찰 포인트

세 결과 파일을 비교한다.

```bash
# 주요 지표 추출
for f in tp1_rps5.json tp2_rps5.json tp4_rps5.json; do
    echo "=== $f ==="
    jq '{request_throughput, mean_ttft_ms, mean_tpot_ms}' ./results/$f
done
```

예상 결과 패턴 (모델/환경에 따라 다름):

| TP 크기 | request_throughput | mean_ttft_ms | mean_tpot_ms |
|---------|--------------------|--------------|--------------|
| TP=1    | 기준               | 기준         | 기준         |
| TP=2    | 증가 또는 유사     | 감소 가능    | 감소 가능    |
| TP=4    | 모델에 따라 다름   | AllReduce 오버헤드로 증가 가능 | 증가 가능 |

관찰 포인트:
- **TTFT**: prefill 연산이 분산되므로 TP가 클수록 감소하는 경향. 단, AllReduce 횟수도 증가
- **TPOT**: decode 단계도 분산되지만 매 스텝마다 AllReduce 발생 — 작은 모델에서는 오히려 증가할 수 있음
- **throughput**: 대형 모델에서는 TP 증가로 배치 크기를 늘릴 수 있어 throughput 향상. 소형 모델은 통신 비용이 지배적
- opt-125m은 단일 GPU에 충분히 올라가는 모델이므로 TP=4가 반드시 빠르지 않음

---

## 코드 흐름 이해

> `--tensor-parallel-size N`을 전달하면 vllm은 N개의 워커 프로세스를 생성하고,
> 각 레이어의 weight를 N등분하여 각 GPU에 배치한다.
> 매 레이어 끝에서 AllReduce로 부분 결과를 합산한 뒤 다음 레이어로 전달한다.

```
vllm serve --tensor-parallel-size N
    └─ WorkerGroup: N개 워커 프로세스 생성
           └─ parallel_state.py  →  initialize_model_parallel()
                  └─ TP 그룹 초기화 (NCCL process group)
                         └─ 모델 레이어 로드
                                ├─ Attention: num_heads / N 개씩 각 GPU에 배치
                                └─ MLP:
                                     ├─ ColumnParallelLinear  →  weight[:, col_start:col_end]
                                     └─ RowParallelLinear     →  weight[row_start:row_end, :]
                                          └─ forward() 끝에서 AllReduce
```

소스 읽기 시작점:
- `vllm/distributed/parallel_state.py` — `initialize_model_parallel()`: TP 그룹을 어떻게 구성하는지
- `vllm/model_executor/layers/linear.py` — `ColumnParallelLinear`, `RowParallelLinear`: weight 분할 방식과 AllReduce 호출 위치

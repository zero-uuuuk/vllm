# Task 07: Pipeline Parallelism

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: PP=2로 서버 기동 + `vllm bench serve` 측정
- [ ] Step 2: PP=2 + TP=2 혼합 구성 + 동일 벤치마크
- [ ] Step 3: throughput/TTFT/TPOT 변화 비교 및 관찰 포인트 정리
- [ ] Step 4: 마이크로 배치 크기 조절 실험 (`--num-scheduler-steps`)
- [ ] Step 5: 파이프라인 버블 시각화 (타임라인 차트)
- [ ] `vllm/distributed/parallel_state.py`, `vllm/v1/worker/gpu_worker.py` 소스 직접 읽기

---

## 목표

`--pipeline-parallel-size` 옵션으로 PP=1/2 환경을 구성하고 `vllm bench serve`로 성능 변화를 측정한다.
Pipeline Parallelism이 레이어를 스테이지 단위로 GPU 간 어떻게 분할하는지 이해하고,
TP와의 차이 및 혼합 구성(PP+TP)에서의 트레이드오프를 직접 확인한다.

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **Pipeline Parallelism (PP)** | 모델의 레이어 전체를 N개 스테이지로 나눠 각 GPU가 연속된 레이어 묶음을 담당 |
| **PP 스테이지** | 예) 24레이어 모델, PP=2 → GPU 0이 레이어 0-11, GPU 1이 레이어 12-23 담당 |
| **마이크로 배치 (micro-batch)** | 하나의 미니배치를 더 작은 단위로 쪼개 파이프라인 버블을 줄이는 기법 |
| **파이프라인 버블 (bubble)** | 전방 스테이지가 끝나기를 기다리느라 후방 GPU가 idle한 구간 — PP의 핵심 비효율 |
| **P2P 통신** | 스테이지 간 활성화(activation) 값을 `send`/`recv`로 주고받는 방식. AllReduce 없음 |
| **TP vs PP** | TP: 레이어 내부를 쪼갬 + AllReduce 다발 / PP: 레이어 자체를 쪼갬 + P2P 통신만 |
| **--pipeline-parallel-size** | vllm serve에 전달하는 PP 크기 인자 (기본값: 1) |

> PP는 레이어를 순서대로 나누므로 GPU 간 통신은 스테이지 경계에서만 발생한다.
> TP에 비해 통신량이 적지만, 파이프라인 버블로 인한 GPU idle이 발생할 수 있다.

---

## 단계별 실습

### Step 1: PP=2 서버 기동 + 벤치마크

기존 서버를 종료하고 PP=2로 재기동한다.

```bash
# PP=2 서버 기동 (GPU 2장 사용)
vllm serve facebook/opt-125m \
    --pipeline-parallel-size 2 \
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
    --result-filename pp2_rps5.json
```

살펴볼 것:
- 서버 로그에서 PP 스테이지가 몇 개 레이어씩 나뉘는지 확인
- `nvidia-smi`에서 GPU 0(앞쪽 레이어)과 GPU 1(뒤쪽 레이어)의 메모리 사용량 비교
- PP=1 대비 TTFT 변화 방향 확인 (스테이지 간 대기가 TTFT에 미치는 영향)

---

### Step 2: PP=2 + TP=2 혼합 구성 (총 GPU 4장)

PP와 TP를 동시에 사용하면 `world_size = PP × TP = 4`가 된다.

```bash
# PP=2, TP=2 혼합 서버 기동
vllm serve facebook/opt-125m \
    --pipeline-parallel-size 2 \
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
    --result-filename pp2tp2_rps5.json
```

살펴볼 것:
- GPU가 4장 모두 사용되는지 `nvidia-smi`로 확인
- 각 PP 스테이지 내에서 TP=2로 레이어가 추가로 분할됨
- 단순 TP=4와 이 구성의 throughput/latency 차이 관찰

---

### Step 3: 결과 비교 및 관찰 포인트

세 결과 파일을 비교한다.

```bash
# 주요 지표 추출
for f in pp2_rps5.json pp2tp2_rps5.json; do
    echo "=== $f ==="
    jq '{request_throughput, mean_ttft_ms, mean_tpot_ms}' ./results/$f
done
```

예상 결과 패턴 (모델/환경에 따라 다름):

| 구성       | request_throughput | mean_ttft_ms | mean_tpot_ms |
|------------|--------------------|--------------|--------------| 
| PP=2       | 기준               | 기준         | 기준         |
| PP=2, TP=2 | 모델에 따라 다름   | PP 버블 + TP AllReduce 복합 | 복합 영향 |

관찰 포인트:
- **TTFT**: PP 스테이지 간 P2P 통신과 버블이 prefill 시간에 누적됨 — 스테이지 수가 많을수록 TTFT 증가 경향
- **TPOT**: decode는 토큰 1개씩 파이프라인을 통과 — 버블이 크지 않아 TP보다 TPOT 영향이 상대적으로 작을 수 있음
- **throughput**: 모델이 GPU 1장에 충분히 올라간다면 PP는 throughput 이점이 거의 없음. 대형 모델에서 의미 있음
- **PP vs TP 차이**: TP는 매 레이어마다 AllReduce, PP는 스테이지 경계에서만 P2P — 통신 빈도 vs 통신량의 차이

---

### Step 4: 마이크로 배치 크기 조절 실험

vllm은 `--num-scheduler-steps` 옵션으로 한 스케줄러 스텝에 처리할 forward 단계 수를 조절한다.
값을 키우면 파이프라인 버블 비율이 줄어들어 GPU 활용률이 올라갈 수 있다.

```bash
# num-scheduler-steps=1 (기본값)
vllm serve facebook/opt-125m \
    --pipeline-parallel-size 2 \
    --num-scheduler-steps 1 \
    --port 8000

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
    --result-filename pp2_steps1.json

# num-scheduler-steps=8
vllm serve facebook/opt-125m \
    --pipeline-parallel-size 2 \
    --num-scheduler-steps 8 \
    --port 8000

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
    --result-filename pp2_steps8.json
```

살펴볼 것:
- `num-scheduler-steps` 값을 높였을 때 throughput과 TTFT가 어떻게 변화하는지
- 스텝 수가 너무 크면 오히려 지연이 증가하는 지점(sweet spot) 확인
- vllm 서버 로그에서 배치 크기 변화 관찰

---

### Step 5: 파이프라인 버블 시각화

PP=2 실행 중 각 스테이지의 작업 구간과 idle 구간을 타임라인으로 시각화한다.

```python
# visualize_bubble.py
# vllm bench serve 결과 JSON에서 요청별 TTFT/TPOT를 추출해
# 각 스테이지의 타임라인을 matplotlib으로 그린다.

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 결과 파일 로드
with open("./results/pp2_rps5.json") as f:
    data = json.load(f)

# 요청별 시작·종료 시각 재구성
# (완료 시각 = ttft + tpot * output_len)
requests = []
for i, req in enumerate(data.get("individual_request_latencies", [])):
    ttft = req.get("ttft_ms", 0) / 1000
    tpot = req.get("tpot_ms", 0) / 1000
    output_len = req.get("output_len", 1)
    start = req.get("start_time", i * 0.2)
    end = start + ttft + tpot * output_len
    requests.append((start, ttft, end))

# 두 스테이지로 나눠 타임라인 표시
fig, ax = plt.subplots(figsize=(14, 4))
colors = ["steelblue", "tomato"]
stage_labels = ["Stage 0 (layers 0-11)", "Stage 1 (layers 12-23)"]

for i, (start, ttft, end) in enumerate(requests[:20]):  # 상위 20개만
    # Stage 0: prefill 구간
    ax.barh(0, ttft, left=start, height=0.4, color=colors[0], alpha=0.8)
    # Stage 1: prefill 이후 구간 (P2P 지연 포함)
    ax.barh(1, end - (start + ttft), left=start + ttft, height=0.4,
            color=colors[1], alpha=0.8)
    # 버블(idle) 표시: Stage 1이 Stage 0 결과를 기다리는 구간
    ax.barh(1, ttft, left=start, height=0.4, color="lightgray",
            alpha=0.5, hatch="//")

ax.set_yticks([0, 1])
ax.set_yticklabels(stage_labels)
ax.set_xlabel("Time (s)")
ax.set_title("Pipeline Bubble Visualization — PP=2")

bubble_patch = mpatches.Patch(facecolor="lightgray", hatch="//", label="Bubble (idle)")
s0_patch = mpatches.Patch(color=colors[0], label="Stage 0 active")
s1_patch = mpatches.Patch(color=colors[1], label="Stage 1 active")
ax.legend(handles=[s0_patch, s1_patch, bubble_patch], loc="upper right")

plt.tight_layout()
plt.savefig("./results/pipeline_bubble.png", dpi=150)
print("Saved: ./results/pipeline_bubble.png")
```

```bash
.venv/bin/python visualize_bubble.py
```

살펴볼 것:
- 회색 사선(`//`) 구간이 Stage 1의 버블(idle)에 해당함
- `num-scheduler-steps` 값을 바꿔 pp2_steps1.json / pp2_steps8.json으로도 동일하게 시각화
- 버블 비율 = idle 시간 / 전체 시간 — 스텝 수가 클수록 버블 비율이 줄어드는지 확인

---

## 코드 흐름 이해

> `--pipeline-parallel-size N`을 전달하면 vllm은 모델의 레이어를 N개 스테이지로 나눠
> 각 스테이지를 담당하는 GPU 그룹을 구성한다.
> 스테이지 경계에서 활성화 값이 P2P `send`/`recv`로 다음 스테이지에 전달된다.

```
vllm serve --pipeline-parallel-size N
    └─ WorkerGroup: N개 스테이지(각 스테이지당 TP 크기만큼 GPU)
           └─ parallel_state.py  →  initialize_model_parallel()
                  └─ PP 그룹 초기화 (NCCL process group, 스테이지별 P2P)
                         └─ 모델 레이어 로드
                                └─ 레이어를 N등분하여 각 스테이지에 배치
                                     Stage 0: layers[0   : L/N]  → GPU 0
                                     Stage 1: layers[L/N : 2L/N] → GPU 1
                                     ...
                                     Stage N-1: layers[...] → GPU N-1

    Forward pass (마이크로 배치 단위):
        Stage 0 → activation → P2P send →
        Stage 1 → activation → P2P send →
        ...
        Stage N-1 → 출력 토큰 생성
```

소스 읽기 시작점:
- `vllm/distributed/parallel_state.py` — `initialize_model_parallel()`: PP 그룹을 어떻게 구성하는지 (`_PP` GroupCoordinator)
- `vllm/v1/worker/gpu_worker.py` — `init_worker_distributed_environment()`: TP/PP 초기화 진입점
- `vllm/model_executor/models/` — 각 모델의 `forward()`: PP 스테이지에서 어떤 레이어만 실행되는지
- `vllm/core/scheduler.py` — `--num-scheduler-steps`가 반영되는 곳: 한 스텝에 마이크로 배치를 몇 개 묶어 보낼지 결정

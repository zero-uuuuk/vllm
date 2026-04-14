# Task 08: Profiling

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

- [ ] Step 1: `--profiler-config` 옵션으로 서버 기동 + 기본 trace 수집
- [ ] Step 2: `/start_profile` `/stop_profile` API로 특정 구간만 캡처
- [ ] Step 3: `torch_profiler_with_memory=true`로 KV 캐시 메모리 패턴 측정
- [ ] Step 4: TP=1 vs TP=2 — AllReduce 오버헤드 수치화
- [ ] Step 5: trace 파일에서 주요 수치 추출 및 정리
- [ ] `vllm/config/profiler.py`, `vllm/entrypoints/serve/profile/api_router.py` 소스 직접 읽기

---

## 목표

vLLM 내장 `--profiler-config` 인자와 `/start_profile` `/stop_profile` HTTP API를 사용해
추론 내부를 직접 들여다본다.
"TTFT가 느리다"처럼 외부 지표만 보는 게 아니라,
어떤 커널이 얼마나 걸리는지, AllReduce가 전체에서 몇 %인지를 **숫자로** 확인한다.

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **--profiler-config** | vllm serve에 전달하는 프로파일러 설정 인자. `profiler=torch` 또는 `profiler=cuda` 선택 |
| **torch_profiler_dir** | trace 파일 저장 경로. 절대 경로 필수. CPU trace + GPU trace 모두 저장됨 |
| **delay_iterations** | 프로파일 시작 전 건너뛸 iteration 수 — JIT warmup 구간을 제외할 때 사용 |
| **max_iterations** | 캡처할 최대 iteration 수 — trace 파일이 무한정 커지지 않도록 제한 |
| **warmup_iterations** | 스케줄 기반 프로파일링의 warmup 구간 (데이터는 버려짐, 오버헤드 안정화) |
| **active_iterations** | 실제 데이터를 수집하는 iteration 수 (기본값 5) |
| **/start_profile** | 서버가 실행 중일 때 HTTP POST로 프로파일링 시작 |
| **/stop_profile** | HTTP POST로 프로파일링 중단 및 trace 저장 |
| **self_cuda_time** | 해당 연산이 직접 점유한 GPU 시간 (하위 연산 제외) — 병목 탐색의 핵심 수치 |

> `/start_profile` `/stop_profile`은 `--profiler-config.profiler`가 설정된 경우에만 라우터가 등록된다.
> 설정 없이 호출하면 404가 반환된다.

---

## 단계별 실습

### Step 1: --profiler-config 옵션으로 서버 기동

**측정해야 할 값**:
- trace 파일이 `torch_profiler_dir`에 생성되는지
- CPU trace 파일과 GPU trace 파일이 각각 저장되는지
- 서버 로그에서 `"Profiler with mode 'torch' is enabled"` 경고 확인

```bash
mkdir -p /tmp/vllm_prof

# profiler=torch, 5 iteration 캡처
vllm serve facebook/opt-125m \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/vllm_prof \
    --profiler-config.max_iterations=5 \
    --port 8000
```

서버가 뜨면 요청을 보내 trace를 유발한다.

```bash
# 요청 5개 전송
for i in $(seq 1 5); do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"facebook/opt-125m","prompt":"Hello world","max_tokens":32}' > /dev/null
done

# trace 파일 확인
ls -lh /tmp/vllm_prof/
```

---

### Step 2: /start_profile /stop_profile API로 특정 구간 캡처

`delay_iterations`로 warmup을 건너뛰고, bench 실행 중 원하는 구간만 캡처하는 방법이다.

**측정해야 할 값**:
- `/start_profile` 호출 후 응답 코드 200 확인
- 캡처 구간(bench 실행 중)에서만 trace가 기록되는지
- `active_iterations=3`으로 설정 시 정확히 3 iteration만 저장되는지

```bash
mkdir -p /tmp/vllm_prof2

# delay=2: 처음 2 iteration은 건너뜀, active=3: 이후 3 iteration만 수집
vllm serve facebook/opt-125m \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/vllm_prof2 \
    --profiler-config.delay_iterations=2 \
    --profiler-config.active_iterations=3 \
    --port 8000
```

별도 터미널에서 프로파일링 시작 → bench → 중단 순서로 실행한다.

```bash
# 프로파일링 시작
curl -X POST http://localhost:8000/start_profile

# bench 실행 (trace 캡처 구간)
vllm bench serve \
    --model facebook/opt-125m \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 64 \
    --num-prompts 20 \
    --request-rate 5 \
    --port 8000

# 프로파일링 중단 및 저장
curl -X POST http://localhost:8000/stop_profile

ls -lh /tmp/vllm_prof2/
```

---

### Step 3: 메모리 프로파일링 — KV 캐시 할당 패턴

**측정해야 할 값**:
- prefill 직전/직후 GPU 메모리 사용량 변화 (trace의 메모리 타임라인에서 확인)
- 입력 길이 128 vs 256에서 KV 캐시 할당량 차이
- decode 스텝마다 메모리가 얼마나 증가하는지

```bash
mkdir -p /tmp/vllm_prof_mem

vllm serve facebook/opt-125m \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/vllm_prof_mem \
    --profiler-config.torch_profiler_with_memory=true \
    --profiler-config.active_iterations=5 \
    --port 8000
```

```bash
curl -X POST http://localhost:8000/start_profile

# 짧은 입력
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"facebook/opt-125m","prompt":"'"$(python3 -c "print('hello ' * 64)")"'","max_tokens":64}'

# 긴 입력
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"facebook/opt-125m","prompt":"'"$(python3 -c "print('hello ' * 128)")"'","max_tokens":64}'

curl -X POST http://localhost:8000/stop_profile
```

trace를 TensorBoard로 열어 메모리 타임라인을 확인한다.

```bash
.venv/bin/pip install tensorboard
.venv/bin/tensorboard --logdir /tmp/vllm_prof_mem --port 6006
# 브라우저에서 localhost:6006 → "Memory" 탭
```

---

### Step 4: TP=1 vs TP=2 — AllReduce 오버헤드 수치화

**측정해야 할 값**:
- TP=1 trace에서 상위 10개 커널과 각 `self_cuda_time_total`
- TP=2 trace에서 `ncclAllReduce` 커널의 총 시간과 호출 횟수
- 전체 CUDA 시간 중 AllReduce 비율 (%)

```bash
# TP=1 trace
mkdir -p /tmp/vllm_prof_tp1
vllm serve facebook/opt-125m \
    --tensor-parallel-size 1 \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/vllm_prof_tp1 \
    --profiler-config.active_iterations=5 \
    --port 8000 &

sleep 10
curl -X POST http://localhost:8000/start_profile
vllm bench serve --model facebook/opt-125m --dataset-name random \
    --random-input-len 128 --random-output-len 64 --num-prompts 20 --port 8000
curl -X POST http://localhost:8000/stop_profile
kill %1

# TP=2 trace
mkdir -p /tmp/vllm_prof_tp2
vllm serve facebook/opt-125m \
    --tensor-parallel-size 2 \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/vllm_prof_tp2 \
    --profiler-config.active_iterations=5 \
    --port 8000 &

sleep 10
curl -X POST http://localhost:8000/start_profile
vllm bench serve --model facebook/opt-125m --dataset-name random \
    --random-input-len 128 --random-output-len 64 --num-prompts 20 --port 8000
curl -X POST http://localhost:8000/stop_profile
kill %1
```

---

### Step 5: trace 파일에서 주요 수치 추출

TensorBoard 대신 Python으로 직접 파싱해 수치를 뽑는다.

```python
# parse_trace.py
import glob
import json
import gzip

def load_trace(dir_path):
    # torch profiler는 .json.gz 또는 .json으로 저장
    files = glob.glob(f"{dir_path}/**/*.json*", recursive=True)
    events = []
    for f in files:
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt") as fh:
            data = json.load(fh)
        events.extend(data.get("traceEvents", []))
    return events

def top_kernels(events, n=15):
    from collections import defaultdict
    cuda = defaultdict(lambda: {"dur": 0, "count": 0})
    for e in events:
        if e.get("cat") == "kernel":
            name = e.get("name", "unknown")
            cuda[name]["dur"] += e.get("dur", 0)
            cuda[name]["count"] += 1
    total = sum(v["dur"] for v in cuda.values())
    rows = sorted(cuda.items(), key=lambda x: -x[1]["dur"])[:n]
    print(f"{'Kernel':<60} {'dur(ms)':>10} {'count':>8} {'%':>6}")
    print("-" * 90)
    for name, v in rows:
        pct = v["dur"] / total * 100 if total else 0
        print(f"{name:<60} {v['dur']/1000:>10.2f} {v['count']:>8} {pct:>6.1f}%")
    return total

print("=== TP=1 ===")
e1 = load_trace("/tmp/vllm_prof_tp1")
total1 = top_kernels(e1)

print("\n=== TP=2 ===")
e2 = load_trace("/tmp/vllm_prof_tp2")
total2 = top_kernels(e2)

# AllReduce 비율 별도 출력
for label, events, total in [("TP=1", e1, total1), ("TP=2", e2, total2)]:
    ar = sum(e.get("dur", 0) for e in events
             if "allreduce" in e.get("name", "").lower() and e.get("cat") == "kernel")
    print(f"\n{label} AllReduce: {ar/1000:.2f} ms / {total/1000:.2f} ms total = {ar/total*100:.1f}%")
```

```bash
.venv/bin/python parse_trace.py
```

정리 표 (직접 채울 것):

| 구분 | 커널 | self_cuda_time (ms) | 전체 비율 (%) |
|------|------|---------------------|--------------|
| TP=1 | 1위 커널 | | |
| TP=1 | 2위 커널 | | |
| TP=2 | 1위 커널 | | |
| TP=2 | AllReduce | | |

관찰 포인트:
- AllReduce 비율이 클수록 TP 증가의 실효 이득이 줄어드는 이유가 설명됨
- prefill 구간과 decode 구간을 `delay_iterations`로 분리해 캡처하면 병목 위치가 달라지는지 확인

---

## 코드 흐름 이해

```
vllm serve --profiler-config.profiler=torch
    └─ vllm/config/profiler.py  →  ProfilerConfig 파싱 및 검증
           └─ vllm/entrypoints/serve/__init__.py
                  └─ attach_profile_router(app)   ← profiler 설정 시에만 라우터 등록
                         └─ POST /start_profile   →  engine_client.start_profile()
                            POST /stop_profile    →  engine_client.stop_profile()

    프로파일링 범위:
        delay_iterations 동안 skip
        warmup_iterations 동안 오버헤드 안정화
        active_iterations 동안 trace 기록
        → torch_profiler_dir에 .json.gz 저장
```

소스 읽기 시작점:
- `vllm/config/profiler.py` — `ProfilerConfig`: 옵션 전체 목록과 검증 로직
- `vllm/entrypoints/serve/profile/api_router.py` — `/start_profile` `/stop_profile` 라우터 등록 조건
- `vllm/worker/model_runner.py` — 실제 trace가 캡처되는 `execute_model()` 진입점

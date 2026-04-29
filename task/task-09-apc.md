# Task 09: Automatic Prefix Caching (APC)

> **주의**: 이 문서의 내용을 완전히 신뢰하지 말 것.
> 오류가 발생하면 직접 원인을 파악하고 해결하는 것이 학습의 일부다.

## TODO

### Part 1: Mixed Batching + APC (vLLM v1 기본 동작)
- [ ] Step 1: SGLang RadixAttention vs vLLM APC 구조 비교
- [ ] Step 2: APC 비활성 vs 활성 — 동일 prefix 반복 요청으로 TTFT 차이 측정
- [ ] Step 3: 서버 메트릭으로 prefix cache hit rate 확인
- [ ] Step 4: prefix 길이 변화에 따른 히트율과 TTFT 변화 관찰
- [ ] Step 5: `/reset_prefix_cache` API로 캐시 초기화 후 히트율 변화 확인

### Part 2: Chunked Prefill off vs on + APC
- [ ] Step 6: `--no-enable-chunked-prefill` — prefill 전체가 한 번에 들어갈 때의 APC
- [ ] Step 7: chunk 크기 변화에 따른 블록 캐싱 시점과 히트율 비교

### Part 3: P/D disagg with LMCache
- [ ] Step 8: LMCache 커넥터 구조 이해 — vLLM이 서드파티에 위임하는 방식
- [ ] Step 9: LMCache P/D 구성 기동 + 동일 prefix 재요청으로 전송 생략 확인

### 소스 읽기
- [ ] `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/kv_cache_manager.py`
- [ ] `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

---

## 목표

APC를 세 가지 구성에서 직접 실험한다.
vLLM v1 기본값(Mixed Batching + Chunked Prefill 모두 on)에서의 APC,
Chunked Prefill off 시 블록 캐싱 시점 차이,
그리고 P/D가 분리된 환경에서 KV 전송을 생략하는 구조(LMCache)를 순서대로 확인한다.

> **Mixed Batching과 Chunked Prefill은 다른 개념이다.**
> - Mixed Batching: prefill 토큰과 decode 토큰을 같은 배치에 넣어 처리 (여러 요청 간 스케줄링)
> - Chunked Prefill: 하나의 긴 prefill을 여러 스텝으로 나눠 처리 (단일 요청 내부의 분할)
> - Chunked Prefill off 시: prefill이 token_budget 안에 들어오면 전체가 한 번에 스케줄됨.
>   token_budget 초과 시에는 chunking 없이 그 스텝을 통째로 스킵한다.

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **APC (Automatic Prefix Caching)** | 이전 요청에서 계산한 KV 캐시 블록을 해시로 식별해 재사용하는 기능 |
| **블록 해시 (BlockHash)** | `hash(parent_block_hash, token_ids)` — 부모 블록 해시와 현재 블록의 토큰 ID를 연쇄 해싱 |
| **cache hit** | 새 요청의 블록 해시가 `cached_block_hash_to_block`에 존재 — 해당 블록의 prefill을 건너뜀 |
| **delay_cache_blocks** | Chunked Prefill에서 블록이 아직 완성되지 않았을 때 캐싱을 미루는 플래그 |
| **vllm:prefix_cache_hits** | Prometheus 카운터 메트릭. 누적 히트 토큰 수를 추적 |
| **kv_role** | LMCache P/D 구성에서 `kv_producer`(prefill) / `kv_consumer`(decode)를 구분하는 인자 |
| **PYTHONHASHSEED** | prefill·decode 인스턴스 간 블록 해시 일치를 보장하기 위해 동일하게 맞춰야 하는 환경변수 |

> APC는 prefix가 **블록 크기(기본 16토큰) 단위로 정렬**될 때만 히트가 발생한다.
> partial 블록은 해시 계산 자체를 하지 않으므로 캐시 불가 (`"We only hash full blocks"` — `kv_cache_utils.py`).

---

## 단계별 실습

---

## Part 1: Mixed Batching + APC (vLLM v1 기본 동작)

### Step 1: SGLang RadixAttention vs vLLM APC 구조 비교

두 시스템 모두 "공유 prefix의 KV 캐시를 재사용한다"는 목표는 같지만 자료구조와 히트 조건이 다르다.

| 항목 | SGLang RadixAttention | vLLM APC |
|------|----------------------|----------|
| **자료구조** | Radix Tree — 토큰 시퀀스를 트리 노드로 저장 | 해시 테이블 — 블록 해시 → KV 블록 매핑 |
| **히트 단위** | 토큰 단위 최장 일치 | 블록 단위 (기본 16토큰). 블록이 완전히 채워져야 히트 |
| **블록 경계 정렬** | 불필요 | 필요 — prefix 길이가 block_size 배수여야 마지막 블록까지 히트 |
| **eviction 정책** | LRU (트리 노드 단위) | LRU (블록 단위, `BlockPool`) |
| **활성화 방법** | 기본 활성화 | `--enable-prefix-caching` 명시 필요 |

살펴볼 것:
- vLLM이 트리 대신 해시 테이블을 선택한 이유: 블록 단위 KV 캐시 관리와 자연스럽게 맞아떨어짐
- SGLang은 토큰 단위 정밀도가 높은 대신 트리 유지 비용이 있음
- vLLM은 partial 블록(block_size 미만)은 해시 계산 자체를 하지 않으므로 캐시 등록도 히트도 불가능 (`kv_cache_utils.py` — `"We only hash full blocks"` 주석 참고)
- vLLM의 블록 경계 제약이 Step 4 실험에서 직접 확인됨

---

### Step 2: APC 비활성 vs 활성 — TTFT 차이 측정

`--dataset-name random`은 요청마다 토큰을 무작위 생성하므로 prefix 공유가 없다.
고정된 앞부분을 공유하는 요청을 직접 만들어야 APC 효과가 나타난다.

**측정해야 할 값**:
- APC 비활성 vs 활성 각각의 cold/warm TTFT (ms)
- 요청 번호가 늘수록 warm TTFT가 줄어드는 추세

```python
# bench_apc.py
import os, time, json, requests, random, string

URL = "http://localhost:8000/v1/completions"
MODEL = "facebook/opt-125m"
SHARED_PREFIX = "hello world " * 64        # ~128 토큰 고정 prefix
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "apc_result.json")

def rand_suffix():
    return " ".join(random.choices(string.ascii_lowercase, k=32))

ttfts = []
for i in range(100):
    prompt = SHARED_PREFIX + rand_suffix()
    t0 = time.time()
    with requests.post(URL, json={"model": MODEL, "prompt": prompt, "max_tokens": 32, "stream": True}, stream=True) as resp:
        for _ in resp.iter_content(chunk_size=1):
            ttft_ms = (time.time() - t0) * 1000
            break
    ttfts.append(ttft_ms)
    if i < 3 or i % 20 == 0:
        print(f"[{i:3d}] TTFT={ttft_ms:.1f} ms")

result = {"mean_ttft_ms": sum(ttfts)/len(ttfts), "ttfts": ttfts}
with open(f"./results/{OUTPUT_FILE}", "w") as f:
    json.dump(result, f, indent=2)
print(f"mean TTFT = {result['mean_ttft_ms']:.1f} ms")
```

```bash
mkdir -p ./results

# APC 비활성
vllm serve facebook/opt-125m --port 8000
OUTPUT_FILE=apc_off.json .venv/bin/python bench_apc.py

# APC 활성
vllm serve facebook/opt-125m --enable-prefix-caching --port 8000
OUTPUT_FILE=apc_on.json .venv/bin/python bench_apc.py
```

살펴볼 것:
- APC off의 `mean_ttft_ms` vs APC on의 `mean_ttft_ms` 차이
- 서버 로그에서 `num_cached_tokens` 값 확인

---

### Step 3: 서버 메트릭으로 prefix cache hit rate 확인

**측정해야 할 값**:
- `vllm:prefix_cache_hits_total` 요청 전후 증분
- 히트율 = 히트 토큰 수 / 전체 입력 토큰 수
- 첫 번째 요청(cold)과 두 번째 요청(warm)의 카운터 차이

```bash
vllm serve facebook/opt-125m --enable-prefix-caching --port 8000

# 기준값
curl -s http://localhost:8000/metrics | grep prefix_cache

SHARED=$(python3 -c "print('The quick brown fox ' * 20)")
for i in $(seq 1 10); do
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} step ${i}\",\"max_tokens\":16}" \
        > /dev/null
done

# 요청 후
curl -s http://localhost:8000/metrics | grep prefix_cache
```

살펴볼 것:
- `vllm:prefix_cache_hits_total` 증분이 두 번째 요청부터 발생하는지
- 서버 로그의 `Prefix cache hit rate` 출력

---

### Step 4: prefix 길이 변화에 따른 히트율과 TTFT 변화

**측정해야 할 값**:
- prefix 길이 0 / 64 / 128 / 256 토큰 각각의 warm TTFT (ms)
- prefix 길이가 16의 배수일 때와 아닐 때(예: 70토큰)의 히트 토큰 수 차이

```python
# test_prefix_len.py
import time, requests

URL = "http://localhost:8000/v1/completions"
MODEL = "facebook/opt-125m"

def send(prompt):
    t0 = time.time()
    requests.post(URL, json={"model": MODEL, "prompt": prompt, "max_tokens": 16})
    return (time.time() - t0) * 1000

for prefix_len in [0, 64, 70, 128, 256]:
    prompt = "word " * prefix_len + "hello"
    send(prompt)           # cold
    ttft = send(prompt)    # warm
    print(f"prefix_len={prefix_len:4d} | warm TTFT = {ttft:.1f} ms")
```

```bash
.venv/bin/python test_prefix_len.py
```

살펴볼 것:
- prefix가 길수록 warm TTFT 감소 폭이 커지는지
- prefix=70일 때 마지막 불완전 블록(70 % 16 = 6토큰)은 캐시 안 됨 — prefix=64와 TTFT 비교

---

### Step 5: /reset_prefix_cache API로 캐시 초기화

**측정해야 할 값**:
- 초기화 전 warm TTFT vs 초기화 후 cold TTFT 비교
- 초기화 후 `vllm:prefix_cache_hits_total` 증가 멈추는지

```bash
SHARED=$(python3 -c "print('hello world ' * 30)")

# 캐시 채우기
curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} first\",\"max_tokens\":8}" > /dev/null

# warm TTFT
time curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} second\",\"max_tokens\":8}" > /dev/null

# 초기화
curl -X POST http://localhost:8000/reset_prefix_cache

# cold TTFT (초기화 후)
time curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} third\",\"max_tokens\":8}" > /dev/null
```

살펴볼 것:
- `/reset_prefix_cache` 응답 코드 200 확인
- cold TTFT가 Step 2의 APC 비활성 기준값과 유사한지

---

## Part 2: Chunked Prefill off vs on + APC

### Step 6: `--no-enable-chunked-prefill` — prefill 전체가 한 번에 들어갈 때의 APC

vLLM v1 기본값은 `enable_chunked_prefill=True`다.
이 플래그를 끄면 prefill이 절대 잘리지 않는다:
- 토큰 수 ≤ token_budget이면 **한 스텝에 전체 prefill** → 모든 블록이 동시에 캐시 등록
- 토큰 수 > token_budget이면 해당 스텝에서 **통째로 스킵** (chunking 없이 건너뜀)

Chunked Prefill이 켜진 기본 상태에서는 긴 prefill이 여러 스텝에 걸쳐 처리되고,
각 chunk가 완성된 블록부터 순차적으로 캐시에 등록된다 (`delay_cache_blocks` 플래그).

**측정해야 할 값**:
- `--no-enable-chunked-prefill` 서버에서 첫 번째 요청 이후 `prefix_cache_hits` 카운터 증분 확인
- 두 번째 요청의 warm TTFT — 모든 블록이 첫 요청에서 한꺼번에 캐시됐는지 검증
- default(chunked prefill on)와 TTFT 비교

```bash
# Chunked Prefill OFF
vllm serve facebook/opt-125m \
    --enable-prefix-caching \
    --no-enable-chunked-prefill \
    --port 8000

OUTPUT_FILE=apc_no_chunk.json .venv/bin/python bench_apc.py
```

```bash
# 메트릭으로 직접 확인
SHARED=$(python3 -c "print('hello world ' * 64)")

curl -s http://localhost:8000/metrics | grep prefix_cache   # 기준

curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} first\",\"max_tokens\":16}" > /dev/null

curl -s http://localhost:8000/metrics | grep prefix_cache   # 첫 요청 후

curl -s -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"facebook/opt-125m\",\"prompt\":\"${SHARED} second\",\"max_tokens\":16}" > /dev/null

curl -s http://localhost:8000/metrics | grep prefix_cache   # 두 번째 요청 후
```

살펴볼 것:
- 첫 요청 완료 후 `prefix_cache_hits` 증분이 0인지 (자기 자신을 캐시한 것은 hit이 아님)
- 두 번째 요청에서 prefix 블록 전체 hit — `prefix_cache_hits` 증분이 prefix 길이 / block_size 블록 수와 일치하는지
- `vllm/v1/core/sched/scheduler.py:665-673` — `not enable_chunked_prefill and num_new_tokens > token_budget` 시 `break`(스킵) 확인
- `vllm/v1/core/kv_cache_manager.py` — `delay_cache_blocks` 플래그가 chunked prefill off 시 어떻게 처리되는지

---

### Step 7: chunk 크기 변화에 따른 블록 캐싱 시점과 히트율 비교

Chunked Prefill ON 상태에서 `--max-num-batched-tokens`(chunk 크기)를 줄이면
블록 완성 타이밍이 달라진다. chunk 크기 < block_size(16토큰)이면
단일 chunk에서 블록이 완성되지 않아 캐시 등록 자체가 지연된다.

```bash
# 기본(chunked prefill on, max-num-batched-tokens 기본값)
vllm serve facebook/opt-125m \
    --enable-prefix-caching \
    --port 8000

OUTPUT_FILE=apc_on.json .venv/bin/python bench_apc.py

# chunk 크기 = 32 (블록 2개씩 완성)
vllm serve facebook/opt-125m \
    --enable-prefix-caching \
    --max-num-batched-tokens 32 \
    --port 8000

OUTPUT_FILE=apc_chunked32.json .venv/bin/python bench_apc.py

# chunk 크기 = 8 (block_size=16보다 작음 → 단일 chunk에서 블록 완성 불가)
vllm serve facebook/opt-125m \
    --enable-prefix-caching \
    --max-num-batched-tokens 8 \
    --port 8000

OUTPUT_FILE=apc_chunked8.json .venv/bin/python bench_apc.py
```

```bash
for f in apc_no_chunk.json apc_on.json apc_chunked32.json apc_chunked8.json; do
    echo "=== $f ==="
    jq '{mean_ttft_ms}' ./results/$f
done
```

살펴볼 것:
- `max-num-batched-tokens=8`이면 한 chunk 안에서 블록(16토큰)이 완성되지 않음 → `delay_cache_blocks=True` 지속 → APC hit 거의 없는지 확인
- `max-num-batched-tokens=32`이면 블록 2개씩 완성 → 두 번째 요청부터 히트 발생하는지 확인
- `--no-enable-chunked-prefill`(Step 6)과 기본값(chunked on) TTFT 차이 — 전자는 첫 요청 latency가 더 크지만 두 번째 요청 warm TTFT는 비슷해지는지 확인
- `vllm/v1/core/kv_cache_manager.py` — `if not self.enable_caching or delay_cache_blocks: return` 코드 위치 확인

---

## Part 3: P/D disagg with LMCache

### Step 8: LMCache 커넥터 구조 이해

`--enable-prefix-caching`은 단일 인스턴스 내부에서만 동작한다.
P/D 분리 환경에서 "decode가 이미 받은 prefix를 재전송 없이 재사용"하려면
vLLM은 이를 직접 구현하지 않고 **서드파티 KV 커넥터에 위임**한다.

**P2P 커넥터 (기본 P/D disagg) — 캐시 조회 없음:**

```python
# p2p_nccl_connector.py
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    if self.is_producer:
        return 0, False
    # 캐시 조회 없이 "나머지 전부 보내줘"
    num_external_tokens = len(prompt_token_ids) - 1 - num_computed_tokens
    return num_external_tokens, False
```

**LMCache 커넥터 — 로컬 스토어 조회 후 생략 가능:**

```python
# lmcache_integration/vllm_v1_adapter.py
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    num_external_hit_tokens = self.lookup_client.lookup(token_ids)
    # decode 인스턴스 로컬 CPU DRAM에 해당 KV가 있는지 조회
    # hit → P2P 전송 건너뜀, 로컬에서 GPU로 로드
    # miss → prefill 인스턴스에 전송 요청
    return num_external_hit_tokens - num_computed_tokens
```

**전송 경로 비교:**

```
[P2P 커넥터]
    매 요청 → prefill 계산 → NCCL P2P → decode GPU
    (이전에 받은 적 있어도 항상 재전송)

[LMCache 커넥터]
    첫 요청: prefill 계산 → NIXL 전송 → decode CPU DRAM 저장 → decode GPU
    재요청: decode CPU DRAM 조회 hit → decode GPU  (prefill 전혀 관여 안 함)
```

> LMCache 스토어는 **decode 인스턴스의 CPU DRAM**에 위치한다.
> P/D가 다른 머신이라면 전송은 NIXL(RDMA/TCP)로 이루어지고,
> 두 인스턴스가 동일한 `PYTHONHASHSEED`를 써야 블록 해시가 일치한다.

살펴볼 것:
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — `get_num_new_matched_tokens()` 추상 메서드
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` — 캐시 조회 없는 기본 구현
- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py` — `lookup_client.lookup()`
- decode GPU 블록 풀 APC와 LMCache CPU DRAM은 **별개의 시스템**

---

### Step 9: LMCache P/D 구성 기동 + 전송 생략 확인

```bash
pip install lmcache
```

```bash
# configs/lmcache-prefiller.yaml
cat > /tmp/lmcache-prefiller.yaml << 'EOF'
local_cpu: false
max_local_cpu_size: 0
enable_nixl: true
nixl_role: "sender"
nixl_peer_host: "localhost"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
EOF

# configs/lmcache-decoder.yaml
cat > /tmp/lmcache-decoder.yaml << 'EOF'
local_cpu: false
max_local_cpu_size: 0
enable_nixl: true
nixl_role: "receiver"
nixl_peer_host: "localhost"
nixl_peer_port: 55555
nixl_buffer_size: 1073741824
nixl_buffer_device: "cuda"
EOF
```

```bash
# Prefill 인스턴스 (터미널 1)
export PYTHONHASHSEED=123
LMCACHE_CONFIG_FILE=/tmp/lmcache-prefiller.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=0 \
vllm serve facebook/opt-125m \
    --port 8100 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config":{"lmcache_rpc_port":"producer1"}}'
```

```bash
# Decode 인스턴스 (터미널 2)
export PYTHONHASHSEED=123   # prefill과 반드시 동일
LMCACHE_CONFIG_FILE=/tmp/lmcache-decoder.yaml \
LMCACHE_USE_EXPERIMENTAL=True \
CUDA_VISIBLE_DEVICES=1 \
vllm serve facebook/opt-125m \
    --port 8200 \
    --enforce-eager \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config":{"lmcache_rpc_port":"consumer1"}}'
```

```python
# test_lmcache_apc.py
# 동일 prefix로 두 번 요청 → 두 번째는 LMCache hit으로 prefill 전송 생략되는지 확인
import time, requests

DECODE_URL = "http://localhost:8200/v1/completions"
MODEL = "facebook/opt-125m"
SHARED = "hello world " * 64

def send(suffix):
    t0 = time.time()
    requests.post(DECODE_URL, json={
        "model": MODEL,
        "prompt": SHARED + suffix,
        "max_tokens": 16,
    })
    return (time.time() - t0) * 1000

print(f"1st (cold): {send('first'):.1f} ms")
print(f"2nd (warm): {send('second'):.1f} ms")   # LMCache hit → prefill 건너뜀
```

```bash
.venv/bin/python test_lmcache_apc.py
```

살펴볼 것:
- `PYTHONHASHSEED`를 다르게 설정하면 두 번째 요청도 cold처럼 동작하는지 확인
- decode 서버 로그에서 `LMCache hit tokens` 출력 확인
- prefill 서버 로그에서 두 번째 요청 시 prefill 연산이 발생하지 않는지 확인

---

## 코드 흐름 이해

```
[Part 1: Pure APC]
vllm serve --enable-prefix-caching
    └─ BlockPool.cached_block_hash_to_block  ← 해시 → 블록 매핑
새 요청 → hash_block_tokens() → find_longest_cache_hit()
              hit  → 블록 재사용, prefill 건너뜀
              miss → 새 블록 할당 후 prefill 실행

[Part 2: Chunked Prefill + APC]
각 chunk 처리 후 → 완성된 블록만 cache_blocks() 호출
    delay_cache_blocks=True → 아직 완성 안 된 블록은 캐시 등록 보류
    chunk 완성 → delay_cache_blocks=False → 등록

[Part 3: P/D + LMCache]
KVConnectorBase_V1.get_num_new_matched_tokens()
    └─ LMCache: lookup_client.lookup(token_ids)
           hit  → decode CPU DRAM → GPU  (prefill 전혀 관여 안 함)
           miss → prefill 계산 → NIXL 전송 → decode CPU DRAM + GPU
```

소스 읽기 시작점:
- `vllm/v1/core/kv_cache_utils.py` — `hash_block_tokens()`: 블록 해시 생성 방식
- `vllm/v1/core/kv_cache_manager.py` — `allocate_slots()`, `delay_cache_blocks`: Chunked Prefill과 APC 연결 지점
- `vllm/v1/core/block_pool.py` — `BlockPool`, `BlockHashToBlockMap`: 해시 테이블·eviction 관리
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` — `get_num_new_matched_tokens()`: P/D APC 인터페이스
- `vllm/entrypoints/serve/cache/api_router.py` — `POST /reset_prefix_cache`

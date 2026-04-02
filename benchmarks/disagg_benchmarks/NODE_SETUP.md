# Disaggregated Prefill Benchmarks

Prefill과 Decode를 분리하여 KV Cache를 전송하는 벤치마크 도구 모음.

## 구성 파일

| 파일 | 설명 |
|------|------|
| `disagg_prefill_proxy_server.py` | Prefill→Decode 프록시 서버. 요청을 prefill에 보내 KV cache를 생성한 뒤 decode로 스트리밍 |
| `prefill_node_setup.sh` | **분산 환경** — Prefill EC2에서 실행 (Ray Head + 벤치마크) |
| `decode_node_setup.sh` | **분산 환경** — Decode EC2에서 실행 (Ray Worker) |

---

## 분산 환경 실행 (Multi-Node)

서로 다른 EC2 인스턴스(각 1 GPU)에서 Ray Cluster를 통해 KV Cache를 전송함.

### 전제 조건

- 두 노드 간의 **모든 네트워크 통신(TCP/UDP All Traffic)**이 허용되어야 함.
    - AWS 등을 사용하는 경우, 보안 그룹(Security Group) 규칙에서 상대방 노드의 보안 그룹 ID 혹은 IP를 대상으로 모든 트래픽을 허용하는 것을 강력히 권장함.

### Step 1: Prefill Node (Master) — 먼저 실행

```bash
export PREFILL_IP="10.0.x.1"  # 본인 IP
export DECODE_IP="10.0.x.2"   # Decode 노드 IP
export KV_GRANULARITY=1        # 1, 8, 32 중 선택
./benchmarks/disagg_benchmarks/prefill_node_setup.sh
```

`Waiting for Decode Node to join the Ray Cluster...` 메시지가 출력되면 Step 2를 진행.

### Step 2: Decode Node (Worker)

```bash
export PREFILL_IP="10.0.x.1"  # Prefill(Master) 노드의 Private IP
export KV_GRANULARITY=1        # Prefill과 동일한 값
./benchmarks/disagg_benchmarks/decode_node_setup.sh
```

Decode 노드가 Ray에 join하면 Prefill 쪽이 자동으로 진행됨. 프록시 서버 기동과 벤치마크 루프는 `prefill_node_setup.sh`가 자동으로 처리함.

벤치마크 결과는 Prefill 노드의 `results_YYYYMMDD_HHMMSS/` 디렉토리에 저장됨.

---

## Node Setup Scripts 상세

`prefill_node_setup.sh`와 `decode_node_setup.sh`의 내부 동작 설명.

### prefill_node_setup.sh

Prefill 노드(Master)에서 실행. Ray Head 기동 → Decode 노드 대기 → vLLM 서버 기동 → 프록시 기동 → 벤치마크 루프 순으로 진행됨.

#### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PREFILL_IP` | `127.0.0.1` | 이 노드의 Private IP |
| `DECODE_IP` | `127.0.0.1` | Decode 노드의 Private IP |
| `KV_GRANULARITY` | `1` | KV Cache 전송 granularity (1, 8, 32). 몇 개 레이어를 묶어서 한 번에 전송할지 결정 |

#### 실행 흐름

**1. 의존성 설치**

`quart`, `aiohttp`, `ray` 등은 vLLM 기본 설치에 포함되지 않음. 스크립트 상단에서 자동으로 설치함.

**2. Cleanup trap**

`EXIT` 시그널에 `cleanup()`을 등록함. 스크립트 종료(정상/비정상 모두) 시 `vllm serve`, `disagg_prefill_proxy_server.py`, Ray를 kill함.

**3. Ray Head 기동**

Ray는 멀티노드 분산 실행을 위한 클러스터 관리 프레임워크. vLLM은 Ray를 통해 여러 노드에 걸친 프로세스 스케줄링과 통신을 처리함.

Prefill 노드가 Head(Master) 역할을 맡아 클러스터를 초기화함. Decode 노드는 이후 이 주소로 접속(Worker join)함.

```bash
ray start --head --node-ip-address "$PREFILL_IP" --port 6379
```

> [!NOTE]
> Ray는 클러스터 관리 및 프로세스 배치를 담당함. 실제 KV 데이터는 NCCL을 통해 노드 간 직접 전송됨.

**4. Decode 노드 대기 (blocking)**

Ray 클러스터에 노드가 2개 이상 join할 때까지 5초 간격으로 폴링함. `decode_node_setup.sh`가 실행되어야 진행됨.

```bash
until [ "$(ray status | grep -c '^ [0-9]* node_')" -ge 2 ]; do
    sleep 5
done
```

**5. vLLM Prefill 서비스 기동**

`CUDA_VISIBLE_DEVICES=0`으로 포트 8100에 기동. `kv_role=kv_producer`, `kv_rank=0`으로 설정하며 KV를 Decode 노드(`DECODE_IP:14579`)로 전송함.

주요 옵션:
- `--no-enable-chunked-prefill`: chunked prefill이 활성화되면 KV 전송이 마지막 chunk 완료 후에야 시작됨. 비활성화하면 프롬프트 전체를 한 번에 처리하므로 KV 전송이 즉시 시작되고 동작이 단순해짐.
- `--gpu-memory-utilization 0.7`: `recv_store` 버퍼는 vLLM 메모리 계산 밖에서 런타임에 동적으로 GPU 메모리를 추가 점유함. 0.8로 설정하면 vLLM이 80%를 점유한 상태에서 recv_store 버퍼 1.5GB가 추가로 올라오면서 OOM이 발생함. 0.7로 낮춰 여유 공간을 확보함.
- `mem_pool_size_gb: 8`: `recv_store` 누적 크기가 `kv_buffer_size`를 초과하면 GPU 대신 CPU 메모리 풀로 fallback함. 기능은 정상 동작하지만 CPU↔GPU 복사 오버헤드가 발생함. 이 값은 fallback 풀의 최대 크기.

서버 준비 확인은 `curl http://localhost:8100/v1/completions`로 폴링함. Decode 서버(`DECODE_IP:8200`)도 동일하게 폴링하여 양쪽이 모두 준비된 후 진행함.

**6. Disagg Proxy 기동**

```bash
python3 disagg_prefill_proxy_server.py \
    --port 8000 \
    --prefill-url "http://localhost:8100" \
    --decode-url "http://${DECODE_IP}:8200" \
    --prefill-kv-host "$PREFILL_IP" \
    --decode-kv-host "$DECODE_IP" \
    --prefill-kv-port 14579 \
    --decode-kv-port 14580
```

Prefill 서버와 Decode 서버는 각각 독립적인 vLLM 인스턴스로, 서로의 존재를 모름. 클라이언트가 이 둘을 직접 조율하는 것은 불가능하므로, 프록시가 중간에서 두 단계를 순서대로 처리함.

프록시의 핵심 역할은 `X-KV-Target` 헤더 주입. Prefill 요청에 이 헤더를 포함시켜 "KV Cache를 어느 Decode 노드로 보낼지"를 vLLM에게 알림. 이 헤더가 없으면 Prefill 서버는 KV Cache를 어디로 전송해야 할지 알 수 없음.

전체 흐름:
1. 클라이언트 → 프록시(8000): 일반 completion 요청
2. 프록시 → Prefill(8100): `X-KV-Target: {DECODE_IP}:{port}` 헤더를 붙여 전달 → Prefill 연산 후 KV Cache를 NCCL로 Decode 노드에 전송
3. 프록시 → Decode(8200): 동일 요청을 전달 → KV Cache를 받아 토큰 생성 후 스트리밍 응답 반환

클라이언트 입장에서는 단일 엔드포인트처럼 동작함.

**7. 벤치마크 루프**

두 workload case에 대해 `vllm bench serve`를 실행함. lambda=2, 요청 수 500 고정.

| Workload | Input tokens | Output tokens |
|----------|-------------|---------------|
| case1 | 512 | 128 |
| case2 | 128 | 512 |

결과 파일명에 granularity가 포함됨: `case1_g1.json`, `case1_g8.json` 등.

---

### decode_node_setup.sh

Decode 노드(Worker)에서 실행. Ray Worker join → vLLM Decode 서비스 기동 순으로 진행됨.

#### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PREFILL_IP` | `127.0.0.1` | Prefill(Master) 노드의 Private IP |
| `KV_GRANULARITY` | `1` | KV Cache 전송 granularity. Prefill 노드와 동일한 값을 설정해야 함 |

#### 실행 흐름

**1. 의존성 설치**

`ray` 등은 vLLM 기본 설치에 포함되지 않음. 스크립트 상단에서 자동으로 설치함.

**2. Cleanup trap**

`EXIT` 시그널에 `cleanup()`을 등록함. `vllm serve`와 Ray를 kill함.

**3. Ray Worker join**

```bash
ray start --address "$PREFILL_IP:6379"
```

Prefill 노드의 Ray Head에 접속함. 이 시점에 `prefill_node_setup.sh`의 대기 루프(Step 4)가 해제됨.

**4. vLLM Decode 서비스 기동**

`CUDA_VISIBLE_DEVICES=0`으로 포트 8200에 기동. `kv_role=kv_consumer`, `kv_rank=1`로 설정하며 Prefill 노드(`PREFILL_IP:14580`)로부터 KV를 수신함.

주요 옵션 (Prefill과 동일한 이유로 설정):
- `--no-enable-chunked-prefill`
- `--gpu-memory-utilization 0.7`
- `mem_pool_size_gb: 8`

서버 기동 후 `wait`로 스크립트를 유지함. Prefill 노드의 벤치마크가 완료될 때까지 서비스를 유지해야 함.

---
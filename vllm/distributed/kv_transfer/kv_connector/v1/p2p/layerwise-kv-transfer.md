# KV Cache Transfer Granularity Experiment

## 배경

vLLM의 `P2pNcclConnector`는 `maybe_transfer_kv_layer` 데코레이터를 통해 **매 attention 레이어 직후 KV를 즉시 개별 전송**한다 (사실상 granularity=1).

```python
# vllm/model_executor/layers/attention/kv_transfer_utils.py
@maybe_transfer_kv_layer
def forward(...):
    connector.wait_for_layer_load(layer_name)   # no-op (미구현)
    result = func(*args, **kwargs)               # attention 실행
    connector.save_kv_layer(layer_name, ...)     # 즉시 send_tensor()
```

### 현재 동작 요약

**Prefill (Producer)**:
- `save_kv_layer()`가 매 attention 레이어 직후 호출
- PUT_ASYNC 모드: 큐에 넣고 즉시 리턴 → 백그라운드 스레드가 NCCL send
- compute(MLP)와 transfer가 overlap

**Decode (Consumer)**:
- `start_load_kv()`에서 32개 레이어를 **순차 blocking 수신** (전부 완료될 때까지)
- 수신 완료 후 forward 시작
- `wait_for_layer_load()`는 no-op (layer-by-layer pipelining 미구현)

```
Prefill Node                              Decode Node
────────────────                          ──────────
Layer 0 attn → send(KV[0]) ──────────→   start_load_kv() {
Layer 0 mlp    ↕ (overlap)                  recv(KV[0]) ← blocking
Layer 1 attn → send(KV[1]) ──────────→     recv(KV[1]) ← blocking
Layer 1 mlp    ↕ (overlap)                  ...
  ...                                       recv(KV[31]) ← blocking
Layer 31 attn → send(KV[31]) ────────→   }
Layer 31 mlp                              → forward 전체 실행 (decode)
wait_for_save()
```

## 실험 가설

> 현재 granularity=1은 32회 NCCL 호출의 오버헤드가 있다.
> 여러 레이어 KV를 **concat하여 단일 NCCL send**로 보내면 호출 오버헤드가 줄고
> 대역폭 효율이 올라갈 수 있다. 반면 concat은 전송 시작을 지연시키므로
> overlap이 줄어 TTFT가 증가할 수 있다. **최적의 granularity**를 찾는다.

## 조작 변수

`kv_granularity` (전송 단위 레이어 수): **1, 8, 32**

| granularity | 전송 방식 | NCCL 호출 | 특성 |
|---|---|---|---|
| **1** | 레이어별 개별 전송 | 32회 | **현재 동작 (baseline)**. 최대 overlap, 호출 오버헤드 최대 |
| **8** | 8개 레이어 concat → 단일 전송 | **4회** | 균형. overlap 감소 vs 호출 오버헤드 감소 |
| **32** | 32개 레이어 concat → 단일 전송 | **1회** | 구버전 모방. overlap 없음, 호출 오버헤드 최소 |

### 타이밍 다이어그램

```
granularity=1 (baseline, 현재):
  Prefill  ██L0██L1██L2██...██L31██
  NCCL      ↑s0 ↑s1 ↑s2      ↑s31     (32 sends, per-layer)
  Decode   recv 0,1,2,...,31 → forward
           (prefill과 transfer overlap 최대)

granularity=8 (concat):
  Prefill  ██L0~L7██ cat+send ██L8~L15██ cat+send ██L16~L23██ ██L24~L31██
  NCCL              ↑s[0:7]            ↑s[8:15]             ↑s[16:23] ↑s[24:31]
  Decode   recv batch0 → split, recv batch1 → split, ... → forward
           (4 sends, compute와 transfer 부분 overlap)

granularity=32 (구버전 모방):
  Prefill  ██L0██L1██...██L31██ cat+send
  NCCL                                  ↑s[0:31]──────────→
  Decode                                recv batch0 → split → forward
           (1 send, overlap 없음. 전체 prefill 완료 후 전송)
```

---

## 실험 환경

- **인스턴스**: g5.xlarge × 2 (Prefill + Decode)
- **모델**: `Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (32 layers)
- **서빙 프레임워크**: vLLM + Custom `BatchedNcclConnector`
- **Precision**: INT4
- **gpu-memory-utilization**: 0.7 (both nodes)
- **Network**: 동일 AZ, Placement Group 기준 ~50Gbps

---

## 수정 대상 파일 목록

| # | 파일 | 작업 |
|---|---|---|
| ① | `vllm/distributed/kv_transfer/kv_connector/v1/p2p/batched_nccl_connector.py` **(신규)** | `P2pNcclConnector` 상속, concat 기반 batched 전송/수신 |
| ② | `vllm/distributed/kv_transfer/kv_connector/factory.py` | `BatchedNcclConnector` 등록 |

> `llama.py`, `base.py`, `arg_utils.py` 수정 불필요.
> `kv_granularity`는 기존 `kv_connector_extra_config`로 전달.

---

## ① `BatchedNcclConnector` — 신규 파일

**경로**: `vllm/distributed/kv_transfer/kv_connector/v1/p2p/batched_nccl_connector.py`

### 핵심 아이디어

- **granularity=1** → 부모(`P2pNcclConnector`) 동작 그대로 위임
- **granularity>1** → Producer: N개 레이어 KV를 concat하여 단일 `send_tensor()` / Consumer: batch 단위 `recv_tensor()` 후 split하여 inject

### 전송 프로토콜 변경

```
granularity=1 (현재):
  tensor_id = "req_123#model.layers.0.self_attn"    → send(KV[0])
  tensor_id = "req_123#model.layers.1.self_attn"    → send(KV[1])
  ...
  tensor_id = "req_123#model.layers.31.self_attn"   → send(KV[31])
  (32 sends, 32 recvs)

granularity=8 (concat):
  tensor_id = "req_123#batch_0"  → send(cat(KV[0:8]))    1개 텐서
  tensor_id = "req_123#batch_1"  → send(cat(KV[8:16]))   1개 텐서
  tensor_id = "req_123#batch_2"  → send(cat(KV[16:24]))  1개 텐서
  tensor_id = "req_123#batch_3"  → send(cat(KV[24:32]))  1개 텐서
  (4 sends, 4 recvs)
```

### 클래스 구조

```python
import time
from typing import Any, TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    P2pNcclConnector,
    P2pNcclConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class BatchedNcclConnector(P2pNcclConnector):
    """
    P2pNcclConnector with configurable transfer granularity.

    granularity=1  → identical to P2pNcclConnector (per-layer, baseline)
    granularity=8  → concat 8 layers, send as one tensor (4 NCCL calls)
    granularity=32 → concat all layers, send as one tensor (1 NCCL call)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._granularity: int = (
            vllm_config.kv_transfer_config
            .get_from_extra_config("kv_granularity", 1)
        )
        # Producer: buffer for accumulating layer KVs before concat+send
        # request_id → list of (layer_name, kv_tensor)
        self._layer_buffer: dict[str, list[tuple[str, torch.Tensor]]] = {}
        self._layer_count: dict[str, int] = {}
        self._batch_idx: dict[str, int] = {}

        logger.info(
            "BatchedNcclConnector initialized with granularity=%d",
            self._granularity,
        )

    # ════════════════════════════════════
    #  Producer (Prefill) side
    # ════════════════════════════════════

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if not self.is_producer:
            return

        # granularity=1 → 부모 동작 (per-layer send, baseline)
        if self._granularity <= 1:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
            return

        # granularity>1 → 버퍼 축적 후 concat + send
        assert self.p2p_nccl_engine is not None
        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, P2pNcclConnectorMetadata)

        for request in connector_metadata.requests:
            request_id = request.request_id
            kv_cache = self._extract_kv(kv_layer, request.block_ids, attn_metadata)
            if kv_cache is None:
                continue

            if request_id not in self._layer_buffer:
                self._layer_buffer[request_id] = []
                self._layer_count[request_id] = 0
                self._batch_idx[request_id] = 0

            self._layer_buffer[request_id].append((layer_name, kv_cache))
            self._layer_count[request_id] += 1

            if self._layer_count[request_id] % self._granularity == 0:
                self._flush_concat(request_id, request)

    def _flush_concat(self, request_id: str, request) -> None:
        """Concat buffered layer KVs into one tensor and send."""
        ip, port = self.parse_request_id(request_id, True)
        remote_address = f"{ip}:{port + self._rank}"

        if request_id not in self._kv_send_start:
            self._kv_send_start[request_id] = time.perf_counter()

        kvs = [kv for _, kv in self._layer_buffer[request_id]]
        merged = torch.cat(kvs, dim=0)

        batch_idx = self._batch_idx[request_id]
        tensor_id = f"{request_id}#batch_{batch_idx}"
        self.p2p_nccl_engine.send_tensor(tensor_id, merged, remote_address)

        self._batch_idx[request_id] += 1
        self._layer_buffer[request_id].clear()

        logger.debug(
            "Flushed batch_%d for %s, shape=%s",
            batch_idx, request_id, merged.shape,
        )

    def wait_for_save(self):
        if self.is_producer and self._granularity > 1:
            connector_metadata = self._get_connector_metadata()
            assert isinstance(connector_metadata, P2pNcclConnectorMetadata)
            for request in connector_metadata.requests:
                rid = request.request_id
                if rid in self._layer_buffer and self._layer_buffer[rid]:
                    self._flush_concat(rid, request)
            self._layer_count.clear()
            self._batch_idx.clear()
        super().wait_for_save()

    # ════════════════════════════════════
    #  Consumer (Decode) side
    # ════════════════════════════════════

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if self.is_producer:
            return

        # granularity=1 → 부모 동작 (per-layer recv)
        if self._granularity <= 1:
            super().start_load_kv(forward_context, **kwargs)
            return

        # granularity>1 → batch 단위 recv 후 split하여 inject
        assert self.p2p_nccl_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, P2pNcclConnectorMetadata)
        if metadata is None:
            return

        # attention 레이어 이름 목록 수집
        attn_layer_names = []
        for layer_name in forward_context.no_compile_layers:
            layer = forward_context.no_compile_layers[layer_name]
            if getattr(layer, "kv_cache", None) is not None:
                attn_layer_names.append(layer_name)

        num_attn_layers = len(attn_layer_names)
        num_batches = (num_attn_layers + self._granularity - 1) // self._granularity

        for request in metadata.requests:
            request_id = request.request_id
            ip, port = self.parse_request_id(request_id, False)
            remote_address = f"{ip}:{port + self._rank}"

            for batch_idx in range(num_batches):
                start = batch_idx * self._granularity
                end = min(start + self._granularity, num_attn_layers)
                batch_layer_names = attn_layer_names[start:end]

                # 하나의 concat된 텐서 수신
                tensor_id = f"{request_id}#batch_{batch_idx}"
                merged = self.p2p_nccl_engine.recv_tensor(tensor_id, remote_address)

                if merged is None:
                    logger.warning("batch recv failed: %s", tensor_id)
                    continue

                # split하여 각 레이어 KV cache에 inject
                chunks = self._split_kv(merged, len(batch_layer_names), attn_metadata)

                for layer_name, kv_chunk in zip(batch_layer_names, chunks):
                    layer_obj = forward_context.no_compile_layers[layer_name]
                    kv_cache = layer_obj.kv_cache
                    self._inject_kv(kv_cache, kv_chunk, request.block_ids,
                                    request_id, attn_metadata)

    # ════════════════════════════════════
    #  Helpers
    # ════════════════════════════════════

    def _extract_kv(self, kv_layer, block_ids, attn_metadata):
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadata,
        )
        if isinstance(attn_metadata, MLACommonMetadata) or kv_layer.shape[1] == 2:
            return kv_layer[block_ids, ...]
        if kv_layer.shape[0] == 2:
            return kv_layer[:, block_ids, ...]
        return None

    def _split_kv(self, merged, num_layers, attn_metadata):
        """Split a concat'd tensor back into per-layer tensors."""
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadata,
        )
        if isinstance(attn_metadata, MLACommonMetadata) or merged.shape[1] == 2:
            # concat on dim=0 → split on dim=0
            per_layer = merged.shape[0] // num_layers
            return merged.split(per_layer, dim=0)
        if merged.shape[0] == 2:
            # concat on dim=1 → split on dim=1
            per_layer = merged.shape[1] // num_layers
            return merged.split(per_layer, dim=1)
        return [merged]

    def _inject_kv(self, layer, kv_cache, block_ids, request_id, attn_metadata):
        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadata,
        )
        if isinstance(attn_metadata, MLACommonMetadata) or layer.shape[1] == 2:
            num_block = kv_cache.shape[0]
            if len(block_ids) == num_block:
                layer[block_ids, ...] = kv_cache
            else:
                layer[block_ids[:num_block], ...] = kv_cache
        elif layer.shape[0] == 2:
            num_block = kv_cache.shape[1]
            if len(block_ids) == num_block:
                layer[:, block_ids, ...] = kv_cache
            else:
                layer[:, block_ids[:num_block], ...] = kv_cache
```

### concat 시 텐서 형상 예시 (FlashInfer, Llama-8B)

```
단일 레이어 KV: shape = [num_blocks, 2, num_kv_heads, head_dim]
                          e.g. [4, 2, 8, 128]

granularity=8 concat (dim=0):
  cat([4,2,8,128] × 8) → [32, 2, 8, 128]    ← 1회 NCCL send

granularity=32 concat (dim=0):
  cat([4,2,8,128] × 32) → [128, 2, 8, 128]  ← 1회 NCCL send
```

### Producer/Consumer 대칭 요약

| | granularity=1 | granularity=8 | granularity=32 |
|---|---|---|---|
| **Producer send** | `send(KV[i])` × 32 | `send(cat(KV[0:8]))` × 4 | `send(cat(KV[0:32]))` × 1 |
| **Consumer recv** | `recv("req#layer.i")` × 32 | `recv("req#batch_i")` × 4 → split | `recv("req#batch_0")` × 1 → split |
| **tensor_id** | `req#model.layers.N.self_attn` | `req#batch_N` | `req#batch_0` |

---

## ② `factory.py` — 새 커넥터 등록

### 수정 위치

`vllm/distributed/kv_transfer/kv_connector/factory.py`

기존 `P2pNcclConnector` 등록 라인 아래에 추가:

```python
KVConnectorFactory.register_connector(
    "BatchedNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.batched_nccl_connector",
    "BatchedNcclConnector",
)
```

---

## 실행 예시

### granularity=1 (baseline, 현재 P2pNcclConnector와 동일)

```bash
--kv-transfer-config '{
    "kv_connector": "BatchedNcclConnector",
    "kv_role": "kv_producer", "kv_rank": 0, "kv_parallel_size": 2,
    "kv_ip": "10.0.0.1", "kv_port": 14579,
    "kv_connector_extra_config": {
        "kv_granularity": 1,
        "send_type": "PUT_ASYNC", "http_port": 8100
    }
}'
```

### granularity=8 / 32

`"kv_granularity": 8` 또는 `"kv_granularity": 32`로 변경.

---

## 측정 지표

| 지표 | 측정 방법 |
|---|---|
| **Decode TTFT** | 클라이언트 타임스탬프 (요청 전송 → 첫 토큰 수신) |
| **Throughput** (req/s) | 클라이언트 측 또는 vLLM metrics 엔드포인트 |
| **KV Transfer Time** | connector 내 `_kv_send_start` 로그 (기존 instrumentation) |
| **GPU Utilization** | nvtop 스크린샷 (정성적) |

---

## 실험 매트릭스

| 조건 | granularity=1 (baseline) | granularity=8 | granularity=32 |
|---|---|---|---|
| prompt=128 | | | |
| prompt=512 | | | |
| prompt=1024 | | | |
| prompt=2048 | | | |

각 조건에서 `max_tokens=128`, 반복 3회 후 중앙값 사용.

---

## 구현 순서

```
Phase 1: 골격
  ② factory.py에 BatchedNcclConnector 등록
  ① BatchedNcclConnector 기본 골격 (granularity=1 → 부모 위임, 기존 동작 확인)

Phase 2: Producer concat 전송
  ① save_kv_layer() 오버라이드 — 버퍼 축적 + concat + send
  ① wait_for_save() 오버라이드 — 잔여 flush

Phase 3: Consumer split 수신
  ① start_load_kv() 오버라이드 — batch recv + split + inject

Phase 4: 검증
  - granularity=1 → P2pNcclConnector와 동일 결과 확인
  - granularity=8 → 4회 send/recv, concat shape 로그 확인
  - granularity=32 → 1회 send/recv 확인

Phase 5: 벤치마크
  - 실험 매트릭스 전체 실행
  - TTFT, Throughput, KV Transfer Time 수집
```

---

## 리스크 및 주의사항

1. **concat 메모리**: `torch.cat()`이 새 텐서를 할당함. granularity=32일 때 전체 모델 KV의 복사본이 잠시 존재. GPU 메모리 여유 필요.
2. **split 정확성**: 모든 레이어의 KV shape이 동일해야 dim=0 split이 정확함. Llama는 uniform이므로 문제 없지만, MoE 등 이종 구조에서는 검증 필요.
3. **Chunked Prefill 호환**: 기존 `P2pNcclConnector`의 `chunked_prefill` 딕셔너리 로직과의 상호작용 검증 필요.
4. **batch_idx 동기화**: Producer와 Consumer가 동일한 `batch_idx` 순서를 사용해야 함. `attn_layer_names`의 순서가 양쪽에서 동일한지 확인 필요.

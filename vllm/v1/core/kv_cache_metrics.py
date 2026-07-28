# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV cache metrics tracking."""

import random
import time
from collections import deque
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # NOTE(QuotaServe PR 0): hook 시그니처가 Request를 참조하지만, 이 모듈은
    # hot-path에서 import되므로 런타임 순환 import를 피하기 위해 TYPE_CHECKING
    # 블록 안에서만 타입을 들여온다. PR3의 QuotaServeCollector가 이 인자들로
    # owner attribution과 occupancy 상태 전이를 처리한다.
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

from vllm.v1.metrics.stats import KVCacheEvictionEvent


class BlockMetricsState:
    """Tracks lifecycle metrics for a single KV cache block."""

    def __init__(self):
        now_ns = time.monotonic_ns()
        self.birth_time_ns = now_ns
        self.last_access_ns = now_ns
        # Bounded to prevent unbounded growth if a block is accessed many times.
        self.access_history: deque[int] = deque(maxlen=4)

    def record_access(self) -> None:
        now_ns = time.monotonic_ns()
        self.last_access_ns = now_ns
        self.access_history.append(now_ns)

    def get_lifetime_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.birth_time_ns) / 1e9

    def get_idle_time_seconds(self) -> float:
        now_ns = time.monotonic_ns()
        return (now_ns - self.last_access_ns) / 1e9

    def get_reuse_gaps_seconds(self) -> list[float]:
        if len(self.access_history) < 2:
            return []
        history = list(self.access_history)
        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]


class KVCacheMetricsCollector:
    """Collects KV cache residency metrics with sampling.

    QuotaServe hook map (PR 0)
    --------------------------
    이 클래스는 BlockPool이 호출하는 **모든 block lifecycle hook의 단일 정의
    지점**이다(`QUOTASERVE_IMPLEMENTATION_PLAN.md` §4.1). base 구현은 Case 1과
    동일하게 *observation 전용*(샘플링 기반 residency 계측)이며, QuotaServe 정책
    개입은 이후 PR에서 이 클래스를 상속한 전용 collector가 담당한다.

    PR 0에서 hook을 두 종류로 정리했다.

    1) **시그니처 확장 hook** — 기존 observation hook에 trigger/owner request를
       optional 인자로 추가한다. base는 인자를 무시하므로 동작이 바뀌지 않는다.
       PR3의 QuotaServeCollector가 이 인자로 block owner tag를 부여한다.
         - on_block_allocated(block, request)
         - on_block_evicted(block, trigger_request)
         - on_block_accessed(block, request)

    2) **신규 hook** — Case 1 계측에는 없던 진입점. base는 no-op이다.
         - on_block_cached(block, request)         : cache 등록 순간
         - on_block_freed(block, prev_ref, new_ref): ref_cnt 감소(특히 →0) 순간
         - on_eviction_recorded(event)             : eviction event 생성 순간
         - on_block_reused(event, request)         : pending eviction 재사용 확정

    모든 신규/확장 hook은 base에서 부작용이 없으므로, 정책이 꺼진 상태
    (mode=off)에서는 baseline LRU와 완전히 동일하게 동작한다.

    NOTE: victim 선택을 가로채는 policy hook은 PR0에
    포함하지 않는다. "실제로 무엇이 evict됐는지"는 on_block_evicted가 block
    단위로 이미 기록하고, "QuotaServe라면 무엇을 골랐을지"는 owner workload
    와 static quota policy가 갖춰져야 의미가 생기기 때문이다. 따라서 그
    진입점은 실제로 동작을 바꾸는 PR4에서 popleft_n 경로에 추가한다.
    """

    collects_eviction_signals = False

    def __init__(self, sample_rate: float = 0.01):
        assert 0 < sample_rate <= 1.0, (
            f"sample_rate must be in (0, 1.0], got {sample_rate}"
        )
        self.sample_rate = sample_rate

        self.block_metrics: dict[int, BlockMetricsState] = {}

        self._eviction_events: list[KVCacheEvictionEvent] = []

    def should_sample_block(self) -> bool:
        return random.random() < self.sample_rate

    # ------------------------------------------------------------------
    # 시그니처 확장 hook (PR 0): 기존 observation hook + trigger/owner request
    # ------------------------------------------------------------------

    def on_block_allocated(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Block이 free queue에서 빠져나와 새로 할당되는 순간.

        호출 지점: ``BlockPool.get_new_blocks()`` (block_pool.py).
        ``request``는 이 block을 끌어온 trigger 요청이다. base collector는
        residency 샘플링만 하므로 무시한다. PR3의 QuotaServeCollector는
        request의 workload tag를 block의 owner로 부여한다.
        """
        if self.should_sample_block():
            self.block_metrics[block.block_id] = BlockMetricsState()

    def on_block_accessed(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Cache hit으로 block의 ref_cnt가 증가하는 순간(touch).

        호출 지점: ``BlockPool.touch()`` (block_pool.py).
        ``request``는 hit을 일으킨 요청(hit-side workload). owner는 hit으로
        바꾸지 않는다. Future PR에서 필요 시 hit-side workload만 별도 metric으로
        기록한다. base는 access 시각만 갱신한다.
        """
        metrics = self.block_metrics.get(block.block_id)
        if metrics:
            metrics.record_access()

    def on_block_evicted(
        self,
        block: "KVCacheBlock",
        trigger_request: "Request | None" = None,
    ) -> None:
        """Cached block이 prefix cache에서 빠지는(evict) 순간.

        호출 지점: ``BlockPool._maybe_evict_cached_block()`` (block_pool.py).
        ``trigger_request``는 이 eviction을 유발한(= 새 block이 필요했던) 요청
        이다. PR3의 QuotaServeCollector는 여기서 victim owner의 occupancy를
        감소시킨다. Trigger workload의 quota log 활용은 이후 PR에서 연결한다.
        """
        metrics = self.block_metrics.pop(block.block_id, None)
        if not metrics:
            return

        lifetime = metrics.get_lifetime_seconds()
        idle_time = metrics.get_idle_time_seconds()
        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())

        self._eviction_events.append(
            KVCacheEvictionEvent(
                lifetime_seconds=lifetime,
                idle_seconds=idle_time,
                reuse_gaps_seconds=reuse_gaps,
            )
        )

    # ------------------------------------------------------------------
    # 신규 hook (PR 0): base는 no-op. QuotaServe collector가 override한다.
    # ------------------------------------------------------------------

    def on_block_cached(
        self,
        block: "KVCacheBlock",
        request: "Request | None" = None,
    ) -> None:
        """Block이 prefix cache에 **등록**되는 순간(is_cached: False → True).

        호출 지점: ``BlockPool.cache_full_blocks()`` insert 직후 (block_pool.py).
        PR3의 QuotaServeCollector는 cache 등록 순간에 occupancy 포함 여부를
        재평가한다.
        base는 계측 대상이 아니므로 no-op.
        """
        return None

    def on_block_freed(
        self,
        block: "KVCacheBlock",
        prev_ref_cnt: int,
        new_ref_cnt: int,
    ) -> None:
        """Block의 ref_cnt가 감소하는 순간(특히 →0 transition).

        호출 지점: ``BlockPool.free_blocks()`` ``ref_cnt -= 1`` 직후
        (block_pool.py). ref_cnt가 0이 되면 그 block은 free queue로 들어가
        evictable 후보가 된다. PR3의 QuotaServeCollector는 이 transition
        (``new_ref_cnt == 0 and is_cached``)에서 occupancy에 추가한다.
        prev/new ref_cnt를 함께 넘기지만 base는 no-op이다.
        """
        return None

    def on_block_reused(
        self,
        event: Mapping[str, object],
        request: "Request | None" = None,
    ) -> None:
        """Notify the collector when a pending eviction is reused.

        호출 지점: ``BlockPool._complete_pending_reuse()``.
        ``request``는 eviction된 prefix를 실제로 다시 계산한 요청이다.
        PR5-2에서는 callback 경로만 연결하고, QuotaServe의 useful eviction
        집계는 이후 PR에서 이 hook을 override한다. base는 no-op이다.
        """
        del event, request
        return None

    def on_eviction_recorded(self, event: Mapping[str, object]) -> None:
        """Notify the collector when an eviction enters the shadow window."""
        del event
        return None

    def maybe_log_signal(
        self,
        now: float | None = None,
        *,
        force: bool = False,
    ) -> int:
        """Optionally write a QuotaServe signal snapshot."""
        del now, force
        return 0

    def flush_signal_log(self, *, force: bool = True) -> int:
        """Optionally flush a QuotaServe signal snapshot."""
        del force
        return 0

    def reset(self) -> None:
        """Clear all state on cache reset."""
        self.block_metrics.clear()
        self._eviction_events.clear()

    def drain_events(self) -> list[KVCacheEvictionEvent]:
        events = self._eviction_events
        self._eviction_events = []
        return events

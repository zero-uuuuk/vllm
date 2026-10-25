# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UniCache's public SimpleHYBRID policy for multi-turn Chat/Agent traffic.

Ported from xsyslab/UniCache, revision
bc51340e3d8a623ecdfc6372ed8b64ec978f1b15, vllm/core/evictor.py and
vllm/simulator/output_simulator.py (Apache-2.0). The single-turn queues and
future-aware OPT variant are outside this experiment's workload.
"""

from __future__ import annotations

import heapq
import math
import time
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request

UPSTREAM_REVISION = "bc51340e3d8a623ecdfc6372ed8b64ec978f1b15"
TASK_CONFIG = {"chat": (4.77, 1.279), "agent": (1.81, 1.092)}


class UniCacheAdapter:
    """Hide task queues, interval scores and online hit-efficiency weights.

    Only zero-reference blocks are candidates; uncached space wins. Freeing
    the final reference records the last-use timestamp available at this V1
    seam. on_prefix_lookup records each request once, including zero hits.
    """

    tracks_sessions = False

    def __init__(
        self,
        blocks: list[KVCacheBlock],
        free_block_queue: FreeKVCacheBlockQueue,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.blocks = blocks
        self.free_block_queue = free_block_queue
        self.clock = clock
        self.on_reset()

    def on_reset(self) -> None:
        self.uncached = OrderedDict(
            (b.block_id, b) for b in self.free_block_queue.get_all_free_blocks()
        )
        self.metadata: dict[int, tuple[str, int]] = {}
        self.candidates: dict[int, tuple[float, int, int, bytes]] = {}
        self.queues: dict[str, list[tuple[float, int, int, bytes]]] = {
            app: [] for app in TASK_CONFIG
        }
        self.hit_history: dict[str, deque[float]] = {
            app: deque(maxlen=500) for app in TASK_CONFIG
        }
        self.weights = {app: 1.0 for app in TASK_CONFIG}
        self.seen_requests: weakref.WeakSet[Request] = weakref.WeakSet()
        self.step = 0
        self.last_update_step = -1

    def on_cached(
        self, block: KVCacheBlock, request: Request, prefix_tokens: int = 0
    ) -> None:
        app = request.application_id
        if app not in TASK_CONFIG:
            raise ValueError("UniCache requires application_id 'chat' or 'agent'")
        if prefix_tokens <= 0:
            raise ValueError("UniCache requires the block's prefix token length")
        self.metadata[block.block_id] = (app, prefix_tokens)
        self.uncached.pop(block.block_id, None)

    def on_prefix_lookup(self, request: Request, hit_tokens: int) -> None:
        if request in self.seen_requests:
            return
        app = request.application_id
        if app not in TASK_CONFIG:
            raise ValueError("UniCache requires application_id 'chat' or 'agent'")
        self.seen_requests.add(request)
        self.step += 1
        self.hit_history[app].append(float(hit_tokens))
        for other, history in self.hit_history.items():
            if other != app:
                decayed = [value * 0.95 for value in history]
                history.clear()
                history.extend(decayed)

    def on_touch(self, block: KVCacheBlock) -> None:
        self.uncached.pop(block.block_id, None)
        self.candidates.pop(block.block_id, None)

    def on_free(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            self.uncached[block.block_id] = block
            return
        app, prefix_tokens = self.metadata[block.block_id]
        entry = (self.clock(), -prefix_tokens, block.block_id, block.block_hash)
        self.candidates[block.block_id] = entry
        heapq.heappush(self.queues[app], entry)
        # Bound lazy entries using the original evictor's cleanup threshold.
        if sum(map(len, self.queues.values())) > 50 * len(self.candidates):
            for queue in self.queues.values():
                queue.clear()
            for bid, current in self.candidates.items():
                heapq.heappush(self.queues[self.metadata[bid][0]], current)

    def on_evict(self, block: KVCacheBlock) -> None:
        self.uncached.pop(block.block_id, None)
        self.candidates.pop(block.block_id, None)
        self.metadata.pop(block.block_id, None)

    def on_allocate(self, block: KVCacheBlock, app: str | None) -> None:
        pass

    def on_request_start(self, app, session, prompt_blocks=0) -> None:
        pass

    def on_request_finish(self, app, session) -> None:
        pass

    def observe_demand(self, app, num_blocks, session=None, blocks=None) -> None:
        pass

    def _update_weights(self) -> None:
        eps = 1e-3
        hits = {app: sum(history) for app, history in self.hit_history.items()}
        total_hits = sum(hits.values()) + eps
        total_cache = len(self.candidates) + eps
        efficiency = {}
        for app, hit_tokens in hits.items():
            # Match public SimpleHYBRID: queue length after popping candidates.
            occupancy = len(self.queues[app]) / total_cache
            if hit_tokens > 0 and occupancy > 0:
                efficiency[app] = (hit_tokens / total_hits) / (occupancy + eps)
        if not efficiency:
            return
        mean_efficiency = sum(efficiency.values()) / len(efficiency) + eps
        scaled = {
            app: (value / mean_efficiency) ** (1 / 0.2)
            for app, value in efficiency.items()
        }
        mean_scaled = sum(scaled.values()) / len(scaled)
        std_scaled = math.sqrt(
            sum((value - mean_scaled) ** 2 for value in scaled.values()) / len(scaled)
        )
        lower = max(0.001, mean_scaled - 2 * std_scaled)
        upper = min(10.0, mean_scaled + 2 * std_scaled)
        for app, value in scaled.items():
            updated = 0.1 * self.weights[app] + 0.9 * value
            self.weights[app] = min(upper, max(lower, updated))

    def _priority(self, app: str, accessed: float, now: float) -> float:
        elapsed = int(max(0.0, now - accessed))
        if elapsed <= 0:
            return self.weights[app]
        mu, sigma = TASK_CONFIG[app]
        cdf = 0.5 * (1 + math.erf((math.log(elapsed) - mu) / (sigma * math.sqrt(2))))
        return self.weights[app] * (1 - cdf)

    def take_free_block(self, application_id=None):
        if self.uncached:
            _, block = self.uncached.popitem(last=False)
            reason = None
        else:
            heads = {}
            for app, queue in self.queues.items():
                while queue:
                    entry = heapq.heappop(queue)
                    if self.candidates.get(entry[2]) == entry:
                        heads[app] = entry
                        break
            if not heads:
                raise RuntimeError("UniCache candidates and free queue disagree")
            if self.step % 10 == 0 and self.step != self.last_update_step:
                self._update_weights()
                self.last_update_step = self.step
            chosen = None
            best_score = math.inf
            best_offset = 0
            now = self.clock()
            for app, entry in heads.items():
                score = self._priority(app, entry[0], now)
                offset = -entry[1]
                if score < best_score or (
                    abs(score - best_score) <= 1e-9 and offset > best_offset
                ):
                    chosen, best_score, best_offset = app, score, offset
            assert chosen is not None
            entry = heads[chosen]
            block = self.blocks[entry[2]]
            self.candidates.pop(block.block_id)
            for app, head in heads.items():
                if app != chosen:
                    heapq.heappush(self.queues[app], head)
            reason = "unicache"
        assert block.ref_cnt == 0 and not block.is_null
        self.free_block_queue.remove(block)
        return block, reason

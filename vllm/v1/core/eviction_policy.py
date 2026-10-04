# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool의 기존 global LRU 선택을 대체할 정책 어댑터의 연결점."""

from __future__ import annotations

import heapq
import os
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Protocol

from vllm.v1.core.quota_serve import QuotaServeAdapter

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


class BlockEvictionPolicy(Protocol):
    """정책이 구현할 블록 상태 변경 hook. 명시적 상속은 필요하지 않다."""

    tracks_sessions: bool

    def on_cached(self, block: KVCacheBlock, request: Request) -> None: ...
    def take_free_block(
        self, application_id: str | None = None
    ) -> tuple[KVCacheBlock, str | None]: ...
    def on_evict(self, block: KVCacheBlock) -> None: ...
    def on_touch(self, block: KVCacheBlock) -> None: ...
    def on_free(self, block: KVCacheBlock) -> None: ...
    def on_reset(self) -> None: ...
    def on_allocate(self, block: KVCacheBlock, app: str | None) -> None: ...
    def on_request_start(
        self, app: str | None, session: str | None, prompt_blocks: int = 0
    ) -> None: ...
    def on_request_finish(self, app: str | None, session: str | None) -> None: ...
    def observe_demand(
        self,
        app: str | None,
        num_blocks: int,
        session: str | None = None,
        blocks: list[KVCacheBlock] | None = None,
    ) -> None: ...


class RankedEvictionAdapter:
    """FIFO by insertion; LFU by acquisition count with least-recent-use ties.

    Frequencies start at one on cache insertion and reset on eviction. Every
    prefix acquisition counts, including acquisitions of already pinned blocks.
    Only zero-reference blocks are indexed as candidates; uncached space wins.
    """

    tracks_sessions = False

    def __init__(self, blocks, free_block_queue, *, mode):
        self.mode = mode
        self.blocks = blocks
        self.free_block_queue = free_block_queue
        self.on_reset()

    def on_reset(self):
        self.clock = 0
        self.frequency = [0] * len(self.blocks)
        self.order = [0] * len(self.blocks)
        self.uncached = OrderedDict(
            (b.block_id, b) for b in self.free_block_queue.get_all_free_blocks()
        )
        self.candidates = {}
        self.heap = []

    def on_cached(self, block, request):
        self.clock += 1
        self.frequency[block.block_id] = 1
        self.order[block.block_id] = self.clock

    def _remove(self, block):
        self.uncached.pop(block.block_id, None)
        self.candidates.pop(block.block_id, None)
        # Bound lazy heap entries during repeated hits without eviction.
        if len(self.heap) > 2 * len(self.candidates) + 64:
            self.heap = list(self.candidates.values())
            heapq.heapify(self.heap)

    def on_touch(self, block):
        self._remove(block)
        if block.block_hash is not None and self.mode == "lfu":
            self.clock += 1
            self.frequency[block.block_id] += 1
            self.order[block.block_id] = self.clock

    def on_free(self, block):
        if block.block_hash is None:
            self.uncached[block.block_id] = block
        else:
            bid = block.block_id
            key = (
                self.frequency[bid] if self.mode == "lfu" else 0,
                self.order[bid],
                bid,
            )
            self.candidates[bid] = key
            heapq.heappush(self.heap, key)

    def on_evict(self, block):
        self._remove(block)
        self.frequency[block.block_id] = 0
        self.order[block.block_id] = 0

    def on_allocate(self, block, app):
        pass

    def on_request_start(self, app, session, prompt_blocks=0):
        pass

    def on_request_finish(self, app, session):
        pass

    def observe_demand(self, app, num_blocks, session=None, blocks=None):
        pass

    def take_free_block(self, application_id=None):
        if self.uncached:
            block = next(iter(self.uncached.values()))
            reason = None
        else:
            while self.heap:
                key = heapq.heappop(self.heap)
                if self.candidates.get(key[2]) == key:
                    block = self.blocks[key[2]]
                    break
            else:
                raise RuntimeError("Free queue and eviction candidates disagree")
            reason = self.mode
        assert block.ref_cnt == 0 and not block.is_null
        self._remove(block)
        self.free_block_queue.remove(block)
        return block, reason


_POLICY_ADAPTERS: dict[
    str, Callable[[list[KVCacheBlock], FreeKVCacheBlockQueue], BlockEvictionPolicy]
] = {
    "quotaserve": QuotaServeAdapter,
    "fifo": partial(RankedEvictionAdapter, mode="fifo"),
    "lfu": partial(RankedEvictionAdapter, mode="lfu"),
}


def create_block_eviction_policy(
    blocks: list[KVCacheBlock],
    free_block_queue: FreeKVCacheBlockQueue,
    enable_caching: bool,
) -> BlockEvictionPolicy | None:
    policy = os.getenv("EVICTION_POLICY", "lru")
    # LRU는 어댑터 없이 BlockPool의 기존 free queue 경로를 사용한다.
    if policy == "lru":
        return None
    adapter = _POLICY_ADAPTERS.get(policy)
    if adapter is None:
        raise ValueError(f"Unknown EVICTION_POLICY: {policy}")
    if not enable_caching:
        raise ValueError(f"{policy} requires prefix caching")
    return adapter(blocks, free_block_queue)

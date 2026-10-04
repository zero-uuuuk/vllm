# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool의 기존 global LRU 선택을 대체할 정책 어댑터의 연결점."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

from vllm.v1.core.quota_serve import QuotaServeAdapter

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


class BlockEvictionPolicy(Protocol):
    """정책이 구현할 블록 상태 변경 hook. 명시적 상속은 필요하지 않다."""

    def on_cached(self, block: KVCacheBlock, request: Request) -> None: ...
    def take_free_block(
        self, application_id: str | None = None
    ) -> tuple[KVCacheBlock, str | None]: ...
    def on_evict(self, block: KVCacheBlock) -> None: ...
    def on_touch(self, block: KVCacheBlock) -> None: ...
    def on_free(self, block: KVCacheBlock) -> None: ...
    def on_reset(self) -> None: ...
    def on_allocate(self, block: KVCacheBlock, app: str | None) -> None: ...
    def observe_demand(self, app: str | None, num_blocks: int) -> None: ...


_POLICY_ADAPTERS = {"quotaserve": QuotaServeAdapter}


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

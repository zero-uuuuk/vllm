# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise FIFO/LFU through BlockPool's real allocation and reference hooks."""

from unittest.mock import Mock

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pytestmark = pytest.mark.skip_global_cleanup


def cache(pool, block):
    key = make_block_hash_with_group_id(BlockHash(str(block.block_id).encode()), 0)
    block.block_hash = key
    pool.eviction_policy.on_cached(block, Mock())
    pool.cached_block_hash_to_block.insert(key, block)


def make_pool(monkeypatch, policy, count=3):
    monkeypatch.setenv("EVICTION_POLICY", policy)
    return BlockPool(num_gpu_blocks=count + 1, enable_caching=True, hash_block_size=16)


def test_fifo_keeps_insertion_order_across_reuse(monkeypatch):
    pool = make_pool(monkeypatch, "fifo")
    a, b, c = pool.get_new_blocks(3)
    for block in (a, b, c):
        cache(pool, block)
    pool.free_blocks([c, b, a])
    pool.touch([a])
    pool.free_blocks([a])
    assert pool.get_new_blocks(2) == [a, b]
    assert pool.take_eviction_counts() == {"fifo": 2}


def test_lfu_counts_hits_while_pinned_and_uses_recency_ties(monkeypatch):
    pool = make_pool(monkeypatch, "lfu")
    a, b, c = pool.get_new_blocks(3)
    for block in (a, b, c):
        cache(pool, block)
    pool.touch([a, a])  # Concurrent prefix acquisitions, before the first release.
    pool.touch([b])
    pool.free_blocks([a, b, c])
    pool.free_blocks([a, b])
    pool.free_blocks([a])
    assert pool.get_new_blocks(1) == [c]
    # Same frequency: a is older than b, despite b being freed first.
    pool.touch([b])
    pool.free_blocks([b])
    assert pool.get_new_blocks(1) == [a]
    assert pool.take_eviction_counts() == {"lfu": 2}


@pytest.mark.parametrize("policy", ["fifo", "lfu"])
def test_uncached_pinned_invalidation_and_reset(monkeypatch, policy):
    pool = make_pool(monkeypatch, policy)
    a, b, blank = pool.get_new_blocks(3)
    cache(pool, a)
    cache(pool, b)
    pool.free_blocks([a, blank])
    assert pool.get_new_blocks(1) == [blank]
    pool.evict_blocks({a.block_id})  # External invalidation of a free cache entry.
    assert pool.get_new_blocks(1) == [a]
    assert b.ref_cnt == 1 and b.block_hash is not None
    assert pool.take_eviction_counts() == {}
    pool.evict_blocks({b.block_id})  # Invalidation must not make a pinned block free.
    assert pool.get_num_free_blocks() == 0
    pool.free_blocks([a, b, blank])
    assert pool.reset_prefix_cache()
    assert len({x.block_id for x in pool.get_new_blocks(3)}) == 3


@pytest.mark.parametrize("policy", ["fifo", "lfu"])
def test_reinsert_resets_history_and_heap_is_bounded(monkeypatch, policy):
    pool = make_pool(monkeypatch, policy, 2)
    a, b = pool.get_new_blocks(2)
    cache(pool, a)
    cache(pool, b)
    pool.free_blocks([a, b])
    for _ in range(300):
        pool.touch([a])
        pool.free_blocks([a])
    adapter = pool.eviction_policy
    assert len(adapter.heap) <= 2 * len(adapter.candidates) + 65
    pool.evict_blocks({a.block_id})
    assert pool.get_new_blocks(1) == [a]
    cache(pool, a)
    pool.free_blocks([a])
    assert adapter.frequency[a.block_id] == 1
    assert pool.get_new_blocks(1) == [b]

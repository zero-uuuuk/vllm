# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.quota_serve import QuotaServeAdapter, QuotaServeController
from vllm.v1.request import Request

# 이 정책 테스트는 가속기를 쓰지 않으므로 공통 GPU 메모리 정리를 건너뛴다.
pytestmark = pytest.mark.skip_global_cleanup


def _cache(pool: BlockPool, block: KVCacheBlock, owner: str) -> None:
    key = make_block_hash_with_group_id(BlockHash(str(block.block_id).encode()), 0)
    block.block_hash = key
    block.owner = owner
    pool.cached_block_hash_to_block.insert(key, block)


def _adapter(pool: BlockPool) -> QuotaServeAdapter:
    assert isinstance(pool.eviction_policy, QuotaServeAdapter)
    return pool.eviction_policy


def test_application_id_from_sampling_extra_args() -> None:
    request = Request(
        request_id="chat-request",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"application_id": "chat"}
        ),
        pooling_params=None,
    )
    assert request.application_id == "chat"


def test_server_derives_application_cache_salt() -> None:
    def make_request(app: str | None, salt: str | None = None) -> Request:
        return Request(
            request_id="request",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(
                max_tokens=1,
                extra_args={"application_id": app} if app is not None else None,
            ),
            pooling_params=None,
            cache_salt=salt,
        )

    chat = make_request("chat")
    assert chat.cache_salt == make_request("chat").cache_salt
    assert chat.cache_salt != make_request("agent").cache_salt
    assert chat.cache_salt != make_request("chat", "client-salt").cache_salt
    assert make_request("chat", "client-salt").cache_salt != make_request(
        "agent", "client-salt"
    ).cache_salt
    assert make_request(None, "client-salt").cache_salt == "client-salt"


def test_full_block_records_request_owner(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    request = Request(
        request_id="chat-request",
        prompt_token_ids=list(range(16)),
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"application_id": "chat"}
        ),
        pooling_params=None,
        block_hasher=lambda _: [BlockHash(b"chat-prefix")],
    )
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    block = pool.get_new_blocks(1)[0]
    pool.cache_full_blocks(request, [block], 0, 1, 16, 0)
    pool.free_blocks([block])

    assert block.owner == "chat"
    assert list(_adapter(pool)._free_cached["chat"].values()) == [block]


def test_quota_shares_and_decay() -> None:
    controller = QuotaServeController()
    controller.last_tick = 0
    controller.observe("chat", 100, 40, 80, now=0.2)
    controller.observe("agent", 100, 0, 20, now=0.3)

    assert controller.shares() is None
    assert controller.pending == {"chat": [60, 80], "agent": [80, 20]}
    controller.tick(now=1.1)
    shares = controller.shares()
    assert shares is not None
    assert shares["chat"] == pytest.approx(60 / 140)
    assert shares["agent"] == pytest.approx(80 / 140)

    demand = controller.signals["chat"].demand
    controller.tick(now=2.1)
    assert controller.signals["chat"].demand == pytest.approx(
        demand * controller.decay
    )
    controller.tick(now=31.1)
    assert controller.shares() is None


def test_controller_ticks_while_idle() -> None:
    controller = QuotaServeController(interval=0.02)
    controller.observe("chat", 100, 0, 0)
    initial_tick = controller.last_tick
    controller.start()
    try:
        deadline = time.monotonic() + 1
        while controller.shares() is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert controller.shares() == {"chat": 1.0}
        assert controller.last_tick > initial_tick
    finally:
        controller.stop()


def test_uncached_free_block_before_cached_victim(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=16)
    cached, uncached, held = pool.get_new_blocks(3)
    _cache(pool, cached, "chat")
    pool.free_blocks([cached, uncached])

    assert pool.get_new_blocks(1) == [uncached]
    assert pool.take_eviction_counts() == {}
    assert cached.block_hash is not None
    assert held.ref_cnt == 1


def test_default_mode_keeps_global_lru(monkeypatch) -> None:
    monkeypatch.delenv("EVICTION_POLICY", raising=False)
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    cached, uncached = pool.get_new_blocks(2)
    _cache(pool, cached, "chat")
    pool.free_blocks([cached, uncached])

    assert pool.eviction_policy is None
    assert pool.get_new_blocks(1) == [cached]
    assert pool.take_eviction_counts() == {"lru": 1}


def test_explicit_lru_keeps_global_lru(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "lru")
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    assert pool.eviction_policy is None


def test_eviction_count_accepts_another_policy(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "lru")
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    block = pool.get_new_blocks(1)[0]
    _cache(pool, block, "chat")
    pool.free_blocks([block])

    policy = Mock()
    policy.take_free_block.side_effect = lambda: (
        pool.free_block_queue.popleft(),
        "fifo",
    )
    pool.eviction_policy = policy

    assert pool.get_new_blocks(1) == [block]
    assert pool.take_eviction_counts() == {"fifo": 1}


def test_unknown_policy_fails_before_experiment(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "invalid")
    with pytest.raises(ValueError, match="Unknown EVICTION_POLICY"):
        BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)


def test_quotaserve_requires_prefix_caching(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    with pytest.raises(ValueError, match="requires prefix caching"):
        BlockPool(num_gpu_blocks=2, enable_caching=False, hash_block_size=16)


def test_over_quota_uses_application_local_lru(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    chat, agent_old, agent_mid, agent_new = pool.get_new_blocks(4)
    for block, owner in (
        (chat, "chat"),
        (agent_old, "agent"),
        (agent_mid, "agent"),
        (agent_new, "agent"),
    ):
        _cache(pool, block, owner)
    pool.free_blocks([chat, agent_old, agent_mid, agent_new])

    controller = _adapter(pool).controller
    now = time.monotonic()
    controller.last_tick = now - 2
    controller.observe("chat", 900, 100, 0, now=now - 1.5)
    controller.observe("agent", 100, 0, 0, now=now - 1.5)
    controller.tick(now=now - 0.5)

    assert pool.get_new_blocks(1) == [agent_old]
    assert chat.block_hash is not None
    assert agent_old.block_hash is None
    assert agent_old.owner is None
    assert pool.get_new_blocks(1) == [agent_mid]
    assert pool.take_eviction_counts() == {"quotaserve": 2}
    assert pool.take_eviction_counts() == {}


def test_inactive_and_global_lru_fallback(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    old, new = pool.get_new_blocks(2)
    _cache(pool, old, "inactive")
    _cache(pool, new, "active")
    pool.free_blocks([new, old])  # 전역 LRU라면 "active"를 선택한다.

    controller = _adapter(pool).controller
    now = time.monotonic()
    controller.last_tick = now - 2
    controller.observe("active", 100, 0, 0, now=now - 1.5)
    controller.tick(now=now - 0.5)
    assert pool.get_new_blocks(1) == [old]
    assert pool.take_eviction_counts() == {"quotaserve": 1}

    controller.signals.clear()
    controller.tick(now=now + 0.5)
    assert pool.get_new_blocks(1) == [new]
    assert pool.take_eviction_counts() == {"lru": 1}


def test_touch_and_external_eviction_update_free_indexes(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    block = pool.get_new_blocks(1)[0]
    _cache(pool, block, "chat")
    pool.free_blocks([block])

    pool.touch([block])
    assert pool.get_num_free_blocks() == 0
    pool.free_blocks([block])
    pool.evict_blocks({block.block_id})
    assert pool.take_eviction_counts() == {}
    assert block.block_hash is None
    assert block.owner is None
    assert pool.get_new_blocks(1) == [block]


def test_reset_clears_ownership_and_free_indexes(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    old, new = pool.get_new_blocks(2)
    _cache(pool, old, "chat")
    _cache(pool, new, "agent")
    pool.free_blocks([old, new])

    assert pool.reset_prefix_cache()
    assert all(block.owner is None and block.block_hash is None for block in (old, new))
    assert pool.get_new_blocks(2) == [old, new]

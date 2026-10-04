# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.quota_serve import QuotaServeAdapter
from vllm.v1.request import Request

# 이 정책 테스트는 가속기를 쓰지 않으므로 공통 GPU 메모리 정리를 건너뛴다.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(autouse=True)
def registered_apps(monkeypatch):
    monkeypatch.setenv("QUOTASERVE_APPS", '["chat", "agent"]')


def _cache(pool: BlockPool, block: KVCacheBlock, owner: str) -> None:
    key = make_block_hash_with_group_id(BlockHash(str(block.block_id).encode()), 0)
    block.block_hash = key
    block.owner = owner
    if pool.eviction_policy is not None:
        pool.eviction_policy.on_cached(block, Mock(application_id=owner))
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
    assert (
        make_request("chat", "client-salt").cache_salt
        != make_request("agent", "client-salt").cache_salt
    )
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
    policy.take_free_block.side_effect = lambda app: (
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


def test_exact_quota_incoming_agent_cannot_evict_chat(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=7, enable_caching=True, hash_block_size=16)
    chat = pool.get_new_blocks(3, "chat")
    agent = pool.get_new_blocks(3, "agent")
    for blocks, app in ((chat, "chat"), (agent, "agent")):
        for block in blocks:
            _cache(pool, block, app)
        pool.free_blocks(blocks)
    assert _adapter(pool).quotas == {"chat": 3, "agent": 3}
    # Even a multi-block allocation must charge each new block before choosing again.
    assert pool.get_new_blocks(3, "agent") == agent
    assert all(block.block_hash is not None for block in chat)
    assert _adapter(pool).resident == {"chat": 3, "agent": 3}
    assert pool.take_eviction_counts() == {"quotaserve": 3}


def test_borrow_idle_capacity_and_reclaim_for_returning_app(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=7, enable_caching=True, hash_block_size=16)
    agent = pool.get_new_blocks(6, "agent")
    for block in agent:
        _cache(pool, block, "agent")
    pool.free_blocks(agent)
    assert _adapter(pool).resident == {"agent": 6}
    assert pool.get_new_blocks(3, "chat") == agent[:3]
    assert _adapter(pool).resident == {"agent": 3, "chat": 3}


def test_pinned_excess_reports_pressure_fallback(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    chat = pool.get_new_blocks(2, "chat")
    for block in chat:
        _cache(pool, block, "chat")
    pool.free_blocks(chat)
    pinned = pool.get_new_blocks(2, "agent")
    assert pool.get_new_blocks(1, "agent") == chat[:1]
    assert all(block.ref_cnt == 1 for block in pinned)
    assert pool.take_eviction_counts() == {"quotaserve_pressure": 1}


def test_pinning_cached_blocks_keeps_resident_charge(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=3, enable_caching=True, hash_block_size=16)
    block = pool.get_new_blocks(1, "chat")[0]
    _cache(pool, block, "chat")
    pool.free_blocks([block])
    pool.touch([block])
    assert _adapter(pool).resident == {"chat": 1}
    pool.evict_blocks({block.block_id})
    assert _adapter(pool).resident == {"chat": 1}
    pool.free_blocks([block])
    assert _adapter(pool).resident == {}


@pytest.mark.parametrize(
    "raw", ["", "[]", "{}", '["chat", "chat"]', '[""]', "[1]", "[null]"]
)
def test_invalid_app_config(monkeypatch, raw) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    monkeypatch.setenv("QUOTASERVE_APPS", raw)
    with pytest.raises(ValueError, match="QUOTASERVE_APPS"):
        BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)


def test_integer_budget_and_unconfigured_app(monkeypatch) -> None:
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=6, enable_caching=True, hash_block_size=16)
    _adapter(pool).observe_demand("chat", 2)
    _adapter(pool).observe_demand("agent", 1)
    assert _adapter(pool).quotas == {"chat": 2, "agent": 1}
    unknown = pool.get_new_blocks(5, "unknown")
    for block in unknown:
        _cache(pool, block, "unknown")
    pool.free_blocks(unknown)
    assert pool.get_new_blocks(1, "chat") == unknown[:1]


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

    quotas = _adapter(pool).quotas.copy()
    assert pool.reset_prefix_cache()
    assert _adapter(pool).quotas == quotas
    assert _adapter(pool).resident == {}
    assert all(block.owner is None and block.block_hash is None for block in (old, new))
    assert pool.get_new_blocks(2, "chat") == [old, new]


def test_mean_demand_ignores_request_frequency_and_adapts_to_size(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=97, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    assert policy.quotas == {"chat": 48, "agent": 48}

    policy.observe_demand("chat", 57)
    # Unseen apps get a neutral prior; no zero-budget cold start.
    assert policy.quotas == {"chat": 48, "agent": 48}
    policy.observe_demand("agent", 33)
    assert policy.quotas == {"chat": 57, "agent": 33}
    for _ in range(100):
        policy.observe_demand("agent", 33)
    assert policy.quotas == {"chat": 57, "agent": 33}
    policy.observe_demand("chat", 17)
    assert policy.mean_demand["chat"] == pytest.approx(49)
    assert policy.quotas == {"chat": 49, "agent": 33}
    policy.observe_demand("chat", 0)
    assert policy.mean_demand["chat"] == pytest.approx(49)
    assert pool.reset_prefix_cache()
    assert policy.mean_demand == {}
    assert policy.quotas == {"chat": 48, "agent": 48}


def test_shared_capacity_uses_oldest_eligible_owner(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 1)
    policy.observe_demand("agent", 1)
    chat = pool.get_new_blocks(2, "chat")
    agent = pool.get_new_blocks(6, "agent")
    for blocks, app in ((chat, "chat"), (agent, "agent")):
        for block in blocks:
            _cache(pool, block, app)
        pool.free_blocks(blocks)
    # Both apps exceed their minimum. Preserve global recency in shared space.
    assert pool.get_new_blocks(1, "agent") == chat[:1]
    # Chat has reached its guarantee; Agent must now replace its own cache.
    assert pool.get_new_blocks(1, "agent") == agent[:1]
    assert chat[1].block_hash is not None


def test_guarantees_scale_down_only_when_demand_exceeds_capacity(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=81, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 57)
    policy.observe_demand("agent", 33)
    assert policy.quotas == {"chat": 51, "agent": 29}


def test_null_only_pool_has_zero_guarantees(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=1, enable_caching=True, hash_block_size=16)
    assert _adapter(pool).quotas == {"chat": 0, "agent": 0}

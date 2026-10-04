# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
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


def test_quota_reclaim_respects_observed_return_interval(monkeypatch):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=7, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    chat = pool.get_new_blocks(3, "chat")
    agent = pool.get_new_blocks(3, "agent")
    for block in chat:
        _cache(pool, block, "chat")
    pool.free_blocks(chat)
    clock[0] = 10
    for block in agent:
        _cache(pool, block, "agent")
    pool.free_blocks(agent)
    policy.mean_gap["agent"] = 1
    # Agent is at quota, but its prefix has not had a chance to return yet.
    clock[0] = 10.5
    assert pool.get_new_blocks(1, "agent") == chat[:1]
    assert pool.take_eviction_counts() == {"quotaserve_reuse": 1}
    # Once the observed interval passes, ordinary quota enforcement resumes.
    clock[0] = 11.1
    assert pool.get_new_blocks(1, "agent") == agent[:1]
    assert pool.take_eviction_counts() == {"quotaserve": 1}


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


def test_waiting_sessions_protect_multiple_prefixes(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=9, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 2)
    policy.observe_demand("agent", 2)
    for session in ("a", "b"):
        policy.on_request_start("chat", session)
        policy.on_request_finish("chat", session)
    assert policy.quotas == {"chat": 4, "agent": 2}
    chat = pool.get_new_blocks(4, "chat")
    agent = pool.get_new_blocks(4, "agent")
    for app, blocks in (("chat", chat), ("agent", agent)):
        for block in blocks:
            _cache(pool, block, app)
        pool.free_blocks(blocks)
    assert pool.get_new_blocks(2, "agent") == agent[:2]
    assert all(block.block_hash is not None for block in chat)


def test_session_expiration_does_not_drop_queued_or_parallel_requests(monkeypatch):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    monkeypatch.setenv("QUOTASERVE_SESSION_IDLE_TTL_S", "60")
    pool = BlockPool(num_gpu_blocks=101, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 10)
    policy.observe_demand("agent", 10)
    for i in range(5):
        policy.on_request_start("chat", str(i))
        policy.on_request_finish("chat", str(i))
    policy.on_request_start("agent", "queued")
    policy.on_request_start("agent", "queued")
    clock[0] = 20
    policy._refresh_quotas()
    assert policy.quotas == {"chat": 50, "agent": 10}
    assert policy.session_counts == {"chat": 5, "agent": 1}
    policy.on_request_finish("agent", "queued")
    clock[0] = 61
    policy._refresh_quotas()
    assert policy.session_counts == {"agent": 1}
    assert pool.reset_prefix_cache()
    assert policy.session_counts == {"agent": 1}
    policy.on_request_finish("agent", "queued")
    clock[0] = 122
    policy._refresh_quotas()
    assert not policy.session_counts


def test_session_demands_share_one_capacity_budget(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=101, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 10)
    policy.observe_demand("agent", 10)
    for app, count in (("chat", 5), ("agent", 10)):
        for i in range(count):
            policy.on_request_start(app, str(i))
    assert policy.session_counts == {"chat": 5, "agent": 10}
    assert policy.quotas == {"chat": 33, "agent": 67}


def test_heterogeneous_sessions_keep_their_own_demand(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=201, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("agent", 10)
    for i, footprint in enumerate((5, 5, 80)):
        policy.on_request_start("chat", str(i))
        policy.observe_demand("chat", footprint, str(i))
        policy.on_request_finish("chat", str(i))
    assert policy.quotas == {"chat": 90, "agent": 10}
    policy.on_request_start("chat", "0")
    policy.observe_demand("chat", 10, "0")
    policy.on_request_finish("chat", "0")
    assert policy.quotas == {"chat": 95, "agent": 10}


def test_arrived_prompt_updates_growing_and_unobserved_sessions(monkeypatch):
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=201, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.observe_demand("chat", 10)
    policy.observe_demand("agent", 10)
    policy.on_request_start("agent", "growing", 40)
    assert policy.quotas == {"chat": 10, "agent": 40}
    assert policy.mean_demand["agent"] == 10  # Arrival is not a completion sample.
    policy.observe_demand("agent", 45, "growing")
    policy.on_request_finish("agent", "growing")
    policy.on_request_start("agent", "growing", 80)
    assert policy.quotas == {"chat": 10, "agent": 80}
    policy.observe_demand("agent", 90, "growing")
    policy.on_request_finish("agent", "growing")
    assert policy.quotas == {"chat": 10, "agent": 90}
    assert not policy._prompt_demands


def test_idle_window_adapts_to_observed_return_gap(monkeypatch):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    monkeypatch.setenv("QUOTASERVE_SESSION_IDLE_TTL_S", "60")
    pool = BlockPool(num_gpu_blocks=101, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    for app in ("chat", "agent"):
        policy.on_request_start(app, "x")
        policy.observe_demand(app, 10, "x")
        policy.on_request_finish(app, "x")
    clock[0] = 1
    policy.on_request_start("agent", "x")
    policy.on_request_finish("agent", "x")
    clock[0] = 20
    policy.on_request_start("chat", "x")
    policy.on_request_finish("chat", "x")
    assert policy._session_timeout("chat") == 60
    assert policy._session_timeout("agent") == 5
    assert policy.session_counts == {"chat": 1}
    assert policy._session_demands == {("chat", "x"): 10}
    policy.on_request_start("agent", "queued")
    clock[0] = 70
    policy._refresh_quotas()
    assert policy.session_counts == {"chat": 1, "agent": 1}
    clock[0] = 81
    policy._refresh_quotas()
    assert policy.session_counts == {"agent": 1}
    assert pool.reset_prefix_cache()
    assert not policy.mean_gap


def test_expired_session_cache_is_reclaimed_before_live_prefix(monkeypatch):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    saved = {}
    for app in ("chat", "agent"):
        policy.on_request_start(app, "a")
        blocks = pool.get_new_blocks(2, app)
        for block in blocks:
            _cache(pool, block, app)
        policy.observe_demand(app, 2, "a", blocks)
        policy.on_request_finish(app, "a")
        pool.free_blocks(reversed(blocks))
        saved[app] = blocks
    policy.mean_gap["agent"] = 1
    clock[0] = 6
    policy._refresh_quotas()
    assert pool.get_new_blocks(1, "chat") == saved["agent"][-1:]
    assert all(b.block_hash is not None for b in saved["chat"])
    assert pool.take_eviction_counts() == {"quotaserve_expired": 1}


@pytest.mark.parametrize("pinned_at_expiry", [False, True])
def test_expiry_preserves_blocks_shared_with_a_live_session(
    monkeypatch, pinned_at_expiry
):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    monkeypatch.setenv("QUOTASERVE_SESSION_IDLE_TTL_S", "60")
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    block = pool.get_new_blocks(1, "chat")[0]
    _cache(pool, block, "chat")
    for session, at in (("a", 0), ("b", 10)):
        clock[0] = at
        if at:
            pool.touch([block])
        policy.on_request_start("chat", session)
        policy.observe_demand("chat", 1, session, [block])
        policy.on_request_finish("chat", session)
        pool.free_blocks([block])
    clock[0] = 61
    policy._refresh_quotas()
    assert not policy._expired_free
    if pinned_at_expiry:
        pool.touch([block])
    clock[0] = 71
    policy._refresh_quotas()
    if pinned_at_expiry:
        assert not policy._expired_free
    else:
        assert list(policy._expired_free.values()) == [block]
        pool.touch([block])
    assert not policy._expired_free
    pool.free_blocks([block])
    assert not policy._expired_free  # A new use refreshes recency.


def test_expiry_ignores_recycled_physical_block(monkeypatch):
    import vllm.v1.core.quota_serve as quota_module

    clock = [0.0]
    monkeypatch.setattr(
        quota_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setenv("EVICTION_POLICY", "quotaserve")
    monkeypatch.setenv("QUOTASERVE_SESSION_IDLE_TTL_S", "60")
    pool = BlockPool(num_gpu_blocks=2, enable_caching=True, hash_block_size=16)
    policy = _adapter(pool)
    policy.on_request_start("chat", "old")
    block = pool.get_new_blocks(1, "chat")[0]
    _cache(pool, block, "chat")
    policy.observe_demand("chat", 1, "old", [block])
    policy.on_request_finish("chat", "old")
    pool.free_blocks([block])
    pool.evict_blocks({block.block_id})
    assert pool.get_new_blocks(1, "agent") == [block]
    key = make_block_hash_with_group_id(BlockHash(b"different-prefix"), 0)
    block.block_hash = key
    block.owner = "agent"
    policy.on_cached(block, Mock(application_id="agent"))
    pool.cached_block_hash_to_block.insert(key, block)
    pool.free_blocks([block])
    clock[0] = 61
    policy._refresh_quotas()
    assert not policy._expired_free
    assert block.block_hash == key


@pytest.mark.parametrize("session", ["session-a", "", 3, None])
def test_cache_session_identity_from_extra_args(session):
    request = Request(
        request_id="request",
        prompt_token_ids=[1],
        sampling_params=SamplingParams(
            max_tokens=1, extra_args={"application_id": "chat", "session_id": session}
        ),
        pooling_params=None,
    )
    assert request.cache_session_id == (
        session if isinstance(session, str) and session else None
    )

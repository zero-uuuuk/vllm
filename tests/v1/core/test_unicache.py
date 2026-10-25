# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the V1 seam and optionally compare pinned upstream source."""

import ast
import json
import logging
import math
import os
import random
from collections import deque
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Union
from unittest.mock import Mock

import numpy as np
import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.unicache import UniCacheAdapter

pytestmark = pytest.mark.skip_global_cleanup


def make_pool(monkeypatch, size=6):
    monkeypatch.setenv("EVICTION_POLICY", "unicache")
    pool = BlockPool(num_gpu_blocks=size + 1, enable_caching=True, hash_block_size=16)
    assert isinstance(pool.eviction_policy, UniCacheAdapter)
    clock = [0.0]
    pool.eviction_policy.clock = lambda: clock[0]
    return pool, clock


def cache(pool, block, app, offset=16, generation=0):
    key = make_block_hash_with_group_id(
        BlockHash(f"{block.block_id}-{generation}".encode()), 0
    )
    block.block_hash = key
    pool.eviction_policy.on_cached(block, Mock(application_id=app), offset)
    pool.cached_block_hash_to_block.insert(key, block)


def test_uncached_space_and_referenced_blocks_are_preserved(monkeypatch):
    pool, _ = make_pool(monkeypatch, 3)
    a, b, blank = pool.get_new_blocks(3)
    cache(pool, a, "chat")
    cache(pool, b, "agent")
    pool.free_blocks([a, blank])
    assert pool.get_new_blocks(1) == [blank]
    assert pool.get_new_blocks(1) == [a]
    assert b.ref_cnt == 1 and b.block_hash is not None
    assert pool.take_eviction_counts() == {"unicache": 1}


def test_interval_cdf_and_position_ties_change_victim(monkeypatch):
    pool, clock = make_pool(monkeypatch, 2)
    chat, agent = pool.get_new_blocks(2)
    cache(pool, chat, "chat", 32)
    cache(pool, agent, "agent", 16)
    pool.free_blocks([chat, agent])
    # Equal age has a different survival probability for each task.
    clock[0] = 20
    assert pool.get_new_blocks(1) == [agent]
    cache(pool, agent, "agent", 64, 1)
    pool.touch([chat])
    pool.free_blocks([chat, agent])
    # At age zero both scores are one; preserve the shorter prefix.
    assert pool.get_new_blocks(1) == [agent]


def test_lookup_is_counted_once_including_misses_and_reset(monkeypatch):
    pool, _ = make_pool(monkeypatch)
    policy = pool.eviction_policy
    chat, agent = Mock(application_id="chat"), Mock(application_id="agent")
    policy.on_request_scheduled(chat, 32)
    policy.on_request_scheduled(chat, 64)
    policy.on_request_scheduled(agent, 0)
    assert policy.step == 2
    assert list(policy.hit_history["chat"]) == [32 * 0.95]
    assert list(policy.hit_history["agent"]) == [0]
    assert pool.reset_prefix_cache()
    assert policy.step == 0 and not policy.seen_requests
    assert policy.weights == {"chat": 1.0, "agent": 1.0}


def test_external_invalidation_and_reused_physical_ids(monkeypatch):
    pool, _ = make_pool(monkeypatch, 2)
    a, b = pool.get_new_blocks(2)
    cache(pool, a, "chat")
    cache(pool, b, "agent")
    pool.free_blocks([a])
    pool.evict_blocks({a.block_id})
    assert pool.get_new_blocks(1) == [a]
    cache(pool, a, "agent", 32, 1)
    pool.free_blocks([a])
    pool.evict_blocks({b.block_id})
    assert pool.get_num_free_blocks() == 1
    assert pool.get_new_blocks(1) == [a]
    pool.free_blocks([b])
    assert pool.get_new_blocks(1) == [b]


def test_unknown_task_fails_instead_of_silently_using_lru(monkeypatch):
    pool, _ = make_pool(monkeypatch, 1)
    with pytest.raises(ValueError, match="application_id"):
        pool.eviction_policy.on_request_scheduled(Mock(application_id="other"), 0)


def test_waiting_lookup_does_not_update_admitted_request_statistics(
    monkeypatch, tmp_path
):
    from vllm.platforms import current_platform

    if not current_platform.device_type:
        pytest.skip("Scheduler integration needs a configured vLLM device platform")
    from tests.v1.core.utils import create_requests, create_scheduler

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 128,
            }
        )
    )
    monkeypatch.setenv("EVICTION_POLICY", "unicache")
    scheduler = create_scheduler(
        model=str(tmp_path),
        num_blocks=4,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        max_model_len=128,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
    )
    requests = create_requests(2, num_tokens=32)
    for request in requests:
        request.application_id = "chat"
        scheduler.add_request(request)
    policy = scheduler.kv_cache_manager.block_pool.eviction_policy
    assert policy.step == 0
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == 1
    assert len(scheduler.waiting) == 1
    assert policy.step == 1
    assert list(policy.hit_history["chat"]) == [0]


@pytest.mark.skipif(
    not os.getenv("UNICACHE_REFERENCE_DIR"),
    reason="Set UNICACHE_REFERENCE_DIR to pinned upstream evictor/simulator sources",
)
def test_matches_upstream_victims_weights_and_hit_history(monkeypatch):
    """Run the actual upstream methods, rather than a second handwritten oracle."""
    root = Path(os.environ["UNICACHE_REFERENCE_DIR"])
    pool, clock = make_pool(monkeypatch)
    policy = pool.eviction_policy
    types = SimpleNamespace(
        MULTI_TURN_TEXT="text",
        MULTI_TURN_AGENTIC="agentic",
        MULTI_TURN_IMAGE="image",
        MULTI_TURN_SEARCH="search",
        MULTI_TURN_FILE="file",
        SINGLE_TURN="single",
        SINGLE_TURN_RANDOM="random",
    )
    simulator = SimpleNamespace(
        tracked_hit_stats={
            q: deque(maxlen=500)
            for q in ("single", "text", "file", "image", "search", "agentic")
        },
        step_track=0,
        global_hit_tokens_num=0,
        global_tokens=0,
        request_id_2_hit_rate_map={},
        request_id_2_hit_count_map={},
        request_id_2_conv_key_map={},
        parser_conv_key=lambda value: value,
    )
    namespace = dict(
        math=math,
        np=np,
        os=os,
        random=random,
        time=SimpleNamespace(time=lambda: clock[0]),
        lru_cache=lru_cache,
        Dict=dict,
        List=list,
        Tuple=tuple,
        Union=Union,
        deque=deque,
        heapq=__import__("heapq"),
        TaskType=types,
        Evictor=object,
        logger=logging.getLogger("unicache_reference"),
        GLOBAL_SIMULATOR_CONFIG=SimpleNamespace(
            eviction_policy="SimpleHYBRID", simulator_enable_virtual_time=False
        ),
        get_global_output_simulator=lambda: simulator,
    )
    wanted = {
        "_cached_lognorm_params",
        "_cached_lognorm_cdf",
        "TASK_CONFIG",
        "_INV_HALF_LIFE",
        "_PRECOMPUTED_DECAY",
        "SimpleHYBRIDlockMetaData",
        "SimpleHYBRIDEvictor",
    }
    tree = ast.parse((root / "evictor.py").read_text())
    nodes = [
        node
        for node in tree.body
        if getattr(node, "name", None) in wanted
        or isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in wanted for t in node.targets)
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "upstream_evictor", "exec"),
        namespace,
    )
    tree = ast.parse((root / "output_simulator.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OutputSimulator"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_hit_rates"
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "upstream_stats", "exec"),
        namespace,
    )
    reference = namespace["SimpleHYBRIDEvictor"]()
    requests = []
    rng = random.Random(17)

    def release(block, app, generation):
        cache(pool, block, app, 16 * (1 + generation % 5), generation)
        pool.free_blocks([block])
        reference.add(
            block.block_id,
            block.block_hash,
            16 * (1 + generation % 5),
            clock[0],
            1,
            types.MULTI_TURN_TEXT if app == "chat" else types.MULTI_TURN_AGENTIC,
            False,
        )

    for i, block in enumerate(pool.get_new_blocks(6)):
        release(block, "chat" if i < 3 else "agent", i)
    for i in range(80):
        clock[0] += rng.choice([0.0, 0.3, 1.0, 20.0])
        app = rng.choice(["chat", "agent"])
        request = Mock(application_id=app)
        requests.append(request)
        hits = rng.choice([0, 16, 32, 64])
        name = f"request-{i}"
        simulator.request_id_2_conv_key_map[name] = (
            ("text_ShareGPT-multi-turn" if app == "chat" else "alfred_") + name,
            i,
        )
        namespace["update_hit_rates"](simulator, name, hits, 128)
        policy.on_request_scheduled(request, hits)
        expected, _ = reference.evict()
        block = pool.get_new_blocks(1)[0]
        assert block.block_id == expected
        for local, upstream in (("chat", "text"), ("agent", "agentic")):
            assert policy.weights[local] == pytest.approx(
                reference.alpha_balance[upstream], rel=1e-12
            )
            assert list(policy.hit_history[local]) == pytest.approx(
                list(simulator.tracked_hit_stats[upstream])
            )
        release(block, app, i + 6)

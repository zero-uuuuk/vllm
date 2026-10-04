# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session-aware mean-footprint KV guarantees with a shared LRU borrowing pool."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, OrderedDict
from typing import TYPE_CHECKING

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


class QuotaServeAdapter:
    """Reclaim borrowed capacity before evicting another app's entitlement.

    Count both pinned and cached blocks. Only unreferenced cached blocks can
    be victims. Mean demand persists across think time; cache reset clears it.
    """

    tracks_sessions = True

    def __init__(
        self, blocks: list[KVCacheBlock], free_block_queue: FreeKVCacheBlockQueue
    ) -> None:
        try:
            apps = json.loads(os.getenv("QUOTASERVE_APPS", ""))
        except (ValueError, TypeError) as exc:
            raise ValueError("QUOTASERVE_APPS must be a JSON list of app IDs") from exc
        if (
            not isinstance(apps, list)
            or not apps
            or any(not isinstance(app, str) or not app for app in apps)
            or len(set(apps)) != len(apps)
        ):
            raise ValueError("QUOTASERVE_APPS requires distinct nonempty app IDs")
        self.apps = tuple(apps)
        self.capacity = sum(not block.is_null for block in blocks)
        self.session_idle_ttl = float(os.getenv("QUOTASERVE_SESSION_IDLE_TTL_S", "60"))
        if not math.isfinite(self.session_idle_ttl) or self.session_idle_ttl <= 0:
            raise ValueError(
                "QUOTASERVE_SESSION_IDLE_TTL_S must be finite and positive"
            )
        self._active_sessions: Counter[tuple[str, str]] = Counter()
        self._waiting_sessions: OrderedDict[tuple[str, str], float] = OrderedDict()
        self.session_counts: Counter[str] = Counter()
        self._session_demands: dict[tuple[str, str], int] = {}
        self._prompt_demands: dict[tuple[str, str], int] = {}
        self._blocks = blocks
        self._session_blocks: dict[tuple[str, str], set[tuple[int, bytes]]] = {}
        self._session_block_refs: Counter[tuple[int, bytes]] = Counter()
        self._expired_free: OrderedDict[int, KVCacheBlock] = OrderedDict()
        self.mean_gap: dict[str, float] = {}
        self.mean_demand: dict[str, float] = {}
        self.quotas: dict[str, int] = {}
        self._refresh_quotas()
        self.free_block_queue = free_block_queue
        self.resident: Counter[str | None] = Counter()
        self._owners: dict[int, str | None] = {}
        self._free_uncached: OrderedDict[int, KVCacheBlock] = OrderedDict(
            (block.block_id, block) for block in blocks if not block.is_null
        )
        self._free_cached: dict[str | None, OrderedDict[int, KVCacheBlock]] = {}
        self._free_order: dict[int, int] = {}
        self._freed_at: dict[int, float] = {}
        self._order = 0

    def on_request_start(
        self, app: str | None, session: str | None, prompt_blocks: int = 0
    ) -> None:
        """Register queued requests too; multiple requests share one session charge."""
        if app not in self.apps or not session:
            return
        assert app is not None
        key = (app, session)
        if key in self._waiting_sessions:
            gap = time.monotonic() - self._waiting_sessions[key]
            previous = self.mean_gap.get(app, gap)
            self.mean_gap[app] = 0.8 * previous + 0.2 * gap
        self._expire_sessions()
        if key not in self._active_sessions and key not in self._waiting_sessions:
            self.session_counts[app] += 1
        self._waiting_sessions.pop(key, None)
        self._active_sessions[key] += 1
        if prompt_blocks > 0:
            self._prompt_demands[key] = max(
                prompt_blocks, self._prompt_demands.get(key, 0)
            )
        self._refresh_quotas()

    def on_request_finish(self, app: str | None, session: str | None) -> None:
        """Keep a completed request's session charged through its think time."""
        if app not in self.apps or not session:
            return
        assert app is not None
        key = (app, session)
        if key not in self._active_sessions:
            return
        self._active_sessions[key] -= 1
        if not self._active_sessions[key]:
            del self._active_sessions[key]
            self._prompt_demands.pop(key, None)
            self._waiting_sessions[key] = time.monotonic()
        self._refresh_quotas()

    def _session_timeout(self, app: str) -> float:
        if app not in self.mean_gap:
            return self.session_idle_ttl
        return max(5.0, 3.0 * self.mean_gap[app])

    def _expire_sessions(self) -> None:
        now = time.monotonic()
        changed = False
        # ponytail: scan waiting sessions; use per-app queues for larger populations.
        for key, completed in tuple(self._waiting_sessions.items()):
            if completed + self._session_timeout(key[0]) > now:
                continue
            del self._waiting_sessions[key]
            self._session_demands.pop(key, None)
            for token in self._session_blocks.pop(key, ()):
                self._session_block_refs[token] -= 1
                if self._session_block_refs[token]:
                    continue
                del self._session_block_refs[token]
                block = self._blocks[token[0]]
                if block.block_hash == token[1] and block.block_id in self._free_order:
                    self._expired_free[block.block_id] = block
                    changed = True
            self.session_counts[key[0]] -= 1
            if not self.session_counts[key[0]]:
                del self.session_counts[key[0]]
        if changed:
            # Sort when sessions expire; allocation and removal stay O(1).
            self._expired_free = OrderedDict(
                sorted(
                    self._expired_free.items(),
                    key=lambda item: self._free_order[item[0]],
                )
            )

    def _remember_session_blocks(
        self, key: tuple[str, str], blocks: list[KVCacheBlock]
    ) -> None:
        previous = self._session_blocks.get(key, set())
        current = {
            token for token in previous if self._blocks[token[0]].block_hash == token[1]
        }
        for token in previous - current:
            self._session_block_refs[token] -= 1
            if not self._session_block_refs[token]:
                del self._session_block_refs[token]
        # Keep valid historical blocks until session expiry. This does not infer
        # the final turn or discard generated text based on the next prompt.
        current.update(
            (b.block_id, h) for b in blocks if (h := b.block_hash) is not None
        )
        for token in current - previous:
            self._session_block_refs[token] += 1
            self._expired_free.pop(token[0], None)
        self._session_blocks[key] = current

    def observe_demand(
        self,
        app: str | None,
        num_blocks: int,
        session: str | None = None,
        blocks: list[KVCacheBlock] | None = None,
    ) -> None:
        """Observe a normally completed request's physical KV footprint.

        Cached hits count toward footprint. Arrival frequency and recomputed
        token rate do not enter the score. A quiet app retains its last mean.
        """
        if app not in self.apps or num_blocks <= 0:
            return
        assert app is not None
        previous = self.mean_demand.get(app, float(num_blocks))
        self.mean_demand[app] = 0.8 * previous + 0.2 * num_blocks
        if session:
            key = (app, session)
            if key in self._active_sessions or key in self._waiting_sessions:
                self._session_demands[key] = num_blocks
                if blocks is not None:
                    self._remember_session_blocks(key, blocks)
        self._refresh_quotas()

    def _refresh_quotas(self) -> None:
        self._expire_sessions()
        # An unseen app inherits the observed apps' mean, avoiding a zero-quota
        # cold start. Before any observations, equal scores give equal quotas.
        prior = (
            sum(self.mean_demand.values()) / len(self.mean_demand)
            if self.mean_demand
            else self.capacity / len(self.apps)
        )
        # Use each live session's last observation, so a long completed request
        # cannot inflate the estimated footprint of every shorter session.
        # ponytail: sum over live sessions; maintain running sums at larger scale.
        observed: Counter[str] = Counter()
        known: Counter[str] = Counter()
        for key in self._session_demands.keys() | self._prompt_demands.keys():
            app, _ = key
            # The arrived prompt is already known; a growing context must not
            # wait until completion to correct its previous-turn estimate.
            footprint = max(
                self._session_demands.get(key, 0), self._prompt_demands.get(key, 0)
            )
            observed[app] += footprint
            known[app] += 1
        scores = {}
        for app in self.apps:
            unknown = max(1, self.session_counts[app]) - known[app]
            scores[app] = max(
                1, math.ceil(observed[app] + unknown * self.mean_demand.get(app, prior))
            )
        total = sum(scores.values())
        budget = min(self.capacity, total)
        exact = {app: budget * score / total for app, score in scores.items()}
        quotas = {app: math.floor(value) for app, value in exact.items()}
        order = sorted(exact, key=lambda app: (-(exact[app] - quotas[app]), app))
        for app in order[: budget - sum(quotas.values())]:
            quotas[app] += 1
        if quotas != self.quotas:
            self.quotas = quotas
            logger.info(
                "QuotaServe session-demand quotas: demand=%s scores=%s "
                "idle_window_s=%s sessions=%s blocks=%s",
                self.mean_demand,
                scores,
                {app: self._session_timeout(app) for app in self.apps},
                dict(self.session_counts),
                self.quotas,
            )

    def _release(self, block: KVCacheBlock) -> None:
        if block.block_id in self._owners:
            owner = self._owners.pop(block.block_id)
            self.resident[owner] -= 1
            if not self.resident[owner]:
                del self.resident[owner]

    def on_allocate(self, block: KVCacheBlock, app: str | None) -> None:
        self._release(block)
        self._owners[block.block_id] = app
        self.resident[app] += 1

    def on_cached(self, block: KVCacheBlock, request: Request) -> None:
        block.owner = request.application_id
        self.on_allocate(block, request.application_id)

    def on_touch(self, block: KVCacheBlock) -> None:
        self._forget_free(block)
        # Ownership/occupancy does not disappear when a cache hit pins a block.

    def on_free(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            self._release(block)
            self._free_uncached[block.block_id] = block
        else:
            self._free_cached.setdefault(block.owner, OrderedDict())[block.block_id] = (
                block
            )
            self._order += 1
            self._free_order[block.block_id] = self._order
            self._freed_at[block.block_id] = time.monotonic()

    def on_evict(self, block: KVCacheBlock) -> None:
        if block.ref_cnt == 0:
            self._forget_free(block)
            self._release(block)
        # An externally invalidated pinned block still consumes physical memory.

    def on_reset(self) -> None:
        # A cache reset may leave queued requests registered in the scheduler.
        self._waiting_sessions.clear()
        self.session_counts = Counter(app for app, _ in self._active_sessions)
        self._session_demands.clear()
        self._session_blocks.clear()
        self._session_block_refs.clear()
        self._expired_free.clear()
        self.mean_gap.clear()
        self.mean_demand.clear()
        self._refresh_quotas()
        self.resident.clear()
        self._owners.clear()
        self._free_cached.clear()
        self._free_order.clear()
        self._freed_at.clear()
        self._order = 0
        self._free_uncached = OrderedDict(
            (block.block_id, block)
            for block in self.free_block_queue.get_all_free_blocks()
        )

    def _forget_free(self, block: KVCacheBlock) -> None:
        self._expired_free.pop(block.block_id, None)
        self._free_uncached.pop(block.block_id, None)
        owner_blocks = self._free_cached.get(block.owner)
        if owner_blocks is not None:
            owner_blocks.pop(block.block_id, None)
            if not owner_blocks:
                del self._free_cached[block.owner]
        self._free_order.pop(block.block_id, None)
        self._freed_at.pop(block.block_id, None)

    def take_free_block(
        self, application_id: str | None = None
    ) -> tuple[KVCacheBlock, str | None]:
        if self._free_uncached:
            block = next(iter(self._free_uncached.values()))
            reason = None
        elif self._expired_free:
            block = next(iter(self._expired_free.values()))
            reason = "quotaserve_expired"
        else:
            # Reclaim the oldest borrowed block. Include the incoming allocation
            # so an app at quota can replace its own blocks without stealing.
            # ponytail: O(apps) scan; use a heap if app count warrants it.
            candidates = []
            for app, blocks in self._free_cached.items():
                quota = self.quotas.get(app, 0) if app is not None else 0
                excess = self.resident[app] + (app == application_id) - quota
                if excess > 0:
                    head_id = next(iter(blocks))
                    candidates.append((self._free_order[head_id], head_id))
            if candidates:
                _, head_id = min(candidates)
                # The queue map avoids a scan over blocks within the chosen app.
                owner = self._owners[head_id]
                block = self._free_cached[owner][head_id]
                reason = "quotaserve"
                # A quota violation alone should not displace a prefix still
                # inside its observed return interval ahead of older cache.
                # No gap estimate means the original quota rule applies.
                gap = self.mean_gap.get(owner, 0.0) if owner is not None else 0.0
                if time.monotonic() - self._freed_at[head_id] < gap:
                    block = self.free_block_queue.fake_free_list_head.next_free_block
                    assert block is not None
                    reason = "quotaserve_reuse"
            else:
                # Excess owners can have all their blocks pinned. Preserve progress
                # and expose this exception rather than claiming strict isolation.
                block = self.free_block_queue.fake_free_list_head.next_free_block
                reason = "quotaserve_pressure"
                assert block is not None
        self._forget_free(block)
        self.free_block_queue.remove(block)
        return block, reason

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-based soft quotas for QuotaServe's prefix-cache eviction."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
    from vllm.v1.request import Request


@dataclass
class _Signals:
    # 1초 구간값을 반영한 EWMA 상태와 마지막 정상 완료 시각.
    demand: float = 0.0
    cached: float = 0.0
    last_completed: float = float("-inf")


# ############ Quota Controller ############
# Signal Observer의 구간 합계를 1초마다 EWMA와 앱별 목표 비율로 바꾼다.
class QuotaServeController:
    def __init__(
        self,
        interval: float = 1.0,
        half_life: float = 10.0,
        active_timeout: float = 30.0,
    ) -> None:
        self.interval = interval
        self.active_timeout = active_timeout
        self.decay = 2 ** (-interval / half_life)
        self.last_tick = time.monotonic()
        self.signals: dict[str, _Signals] = {}
        # 앱별 현재 구간 합계: [미캐시 입력+출력, 내부 캐시 재사용 입력].
        self.pending: dict[str, list[int]] = {}
        # 최근 갱신에서 계산한 soft quota 비율. 유효한 수요가 없으면 None.
        self._shares: dict[str, float] | None = None
        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="quota-serve", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        # 요청이 없는 동안에도 매 구간 EWMA에 0을 반영한다.
        while True:
            with self._lock:
                delay = max(0.0, self.last_tick + self.interval - time.monotonic())
            if self._stop.wait(delay):
                return
            self.tick()

    def _advance(self, now: float) -> bool:
        steps = int((now - self.last_tick) // self.interval)
        if steps <= 0:
            return False
        for app, signals in self.signals.items():
            pending = self.pending.get(app, [0, 0])
            # 밀린 첫 구간에 pending을 반영하고, 나머지 빈 구간만큼 0으로 감쇠한다.
            factor = self.decay ** (steps - 1)
            signals.demand = factor * (
                self.decay * signals.demand
                + (1 - self.decay) * pending[0] / self.interval
            )
            signals.cached = factor * (
                self.decay * signals.cached
                + (1 - self.decay) * pending[1] / self.interval
            )
        self.pending.clear()
        self.last_tick += steps * self.interval
        return True

    def _refresh(self, now: float) -> None:
        if self._advance(now):
            self._shares = self._calculate_shares(self.last_tick)

    def tick(self, now: float | None = None) -> None:
        with self._lock:
            self._refresh(time.monotonic() if now is None else now)

    # ############ Signal Observer: 완료 요청 집계 ############
    def observe(
        self,
        app: str,
        prompt_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            # 타이머보다 완료 요청이 먼저 구간 경계를 지나면 이전 구간을 먼저 확정한다.
            self._refresh(now)
            signals = self.signals.setdefault(app, _Signals())
            signals.last_completed = now
            pending = self.pending.setdefault(app, [0, 0])
            cached_input = min(cached_tokens, prompt_tokens)
            pending[0] += prompt_tokens - cached_input + output_tokens
            pending[1] += cached_input

    def shares(self) -> dict[str, float] | None:
        with self._lock:
            return self._shares.copy() if self._shares is not None else None

    def _calculate_shares(self, now: float) -> dict[str, float] | None:
        # EWMA가 남아 있어도 최근 완료 요청이 없으면 quota 대상에서 제외한다.
        active = {
            app: signal
            for app, signal in self.signals.items()
            if now - signal.last_completed <= self.active_timeout
        }
        demand_total = sum(signal.demand for signal in active.values())
        if demand_total <= 0:
            return None

        return {
            app: signal.demand / demand_total
            for app, signal in active.items()
        }


# ############ Hierarchical Eviction Policy ############
# 앱별 free block과 local LRU를 추적하고, cache pressure 시 회수 대상을 고른다.
class QuotaServeAdapter:
    """Track QuotaServe state and override the pool's free-block choice."""

    def __init__(
        self, blocks: list[KVCacheBlock], free_block_queue: FreeKVCacheBlockQueue
    ) -> None:
        self.controller = QuotaServeController()
        self.free_block_queue = free_block_queue
        # 캐시되지 않은 free block을 먼저 쓰면 기존 캐시를 회수하지 않아도 된다.
        self._free_uncached: OrderedDict[int, KVCacheBlock] = OrderedDict(
            (block.block_id, block) for block in blocks if not block.is_null
        )
        # ref_cnt=0인 cached block만 owner별로 free 진입 순서대로 보관한다.
        self._free_cached: dict[str | None, OrderedDict[int, KVCacheBlock]] = {}
        # 서로 다른 앱의 local LRU head를 비교할 때 사용할 전역 순서.
        self._free_order: dict[int, int] = {}
        self._order = 0

    def start(self) -> None:
        self.controller.start()

    def stop(self) -> None:
        self.controller.stop()

    def on_cached(self, block: KVCacheBlock, request: Request) -> None:
        # owner는 물리 블록이 아닌 현재 등록된 full prefix의 소유자다.
        block.owner = request.application_id

    def on_touch(self, block: KVCacheBlock) -> None:
        self._forget_free(block)

    def on_free(self, block: KVCacheBlock) -> None:
        self._track_free(block)

    def on_evict(self, block: KVCacheBlock) -> None:
        if block.ref_cnt == 0 and block.prev_free_block is not None:
            self._forget_free(block)

    def on_reset(self) -> None:
        self._free_cached.clear()
        self._free_order.clear()
        self._free_uncached = OrderedDict(
            (block.block_id, block)
            for block in self.free_block_queue.get_all_free_blocks()
        )

    def observe_request(self, request: Request) -> None:
        if request.application_id is not None:
            # 완료 요청 한 건을 현재 1초 구간 합계에 더한다.
            self.controller.observe(
                request.application_id,
                request.num_prompt_tokens,
                request.num_output_tokens,
                request.quota_cached_tokens,
            )

    def _forget_free(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            self._free_uncached.pop(block.block_id, None)
        else:
            owner_blocks = self._free_cached.get(block.owner)
            if owner_blocks is not None:
                owner_blocks.pop(block.block_id, None)
                if not owner_blocks:
                    del self._free_cached[block.owner]
        self._free_order.pop(block.block_id, None)

    def _track_free(self, block: KVCacheBlock) -> None:
        if block.block_hash is None:
            self._free_uncached[block.block_id] = block
        else:
            self._free_cached.setdefault(block.owner, OrderedDict())[
                block.block_id
            ] = block
            self._order += 1
            self._free_order[block.block_id] = self._order

    def take_free_block(self) -> tuple[KVCacheBlock, str | None]:
        if self._free_uncached:
            block = next(iter(self._free_uncached.values()))
            reason = None
        else:
            block = None
            reason = "quotaserve"
            shares = self.controller.shares()
            if shares is not None:
                # 1순위: 최근 요청이 없는 앱의 가장 오래된 캐시 블록.
                inactive = [app for app in self._free_cached if app not in shares]
                if inactive:
                    app = min(
                        inactive,
                        key=lambda app: self._free_order[
                            next(iter(self._free_cached[app]))
                        ],
                    )
                    block = next(iter(self._free_cached[app].values()))
                else:
                    # 2순위: 목표 점유량을 상대적으로 가장 많이 초과한 앱.
                    total = sum(len(blocks) for blocks in self._free_cached.values())
                    over = [
                        app
                        for app, blocks in self._free_cached.items()
                        if app is not None and len(blocks) > shares[app] * total
                    ]
                    if over:
                        app = max(
                            over,
                            key=lambda app: (
                                (len(self._free_cached[app]) - shares[app] * total)
                                / max(shares[app] * total, 1),
                                -self._free_order[
                                    next(iter(self._free_cached[app]))
                                ],
                            ),
                        )
                        block = next(iter(self._free_cached[app].values()))
            if block is None:
                # 유효한 quota나 초과 앱이 없으면 기존 global LRU 순서를 따른다.
                block = self.free_block_queue.fake_free_list_head.next_free_block
                reason = "lru"
            assert block is not None
        self._forget_free(block)
        self.free_block_queue.remove(block)
        return block, reason

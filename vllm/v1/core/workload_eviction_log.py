# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write on-demand workload eviction reports for benchmark runs."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from vllm.v1.core.block_pool import BlockPool


def build_workload_eviction_report(block_pool: BlockPool) -> dict[str, Any]:
    """Build summary statistics from the current cache epoch without draining it."""
    stats = []
    for trigger, victim in sorted(block_pool._workload_eviction_counts):
        snapshot = block_pool.get_workload_eviction_stats(trigger, victim)
        stats.append(
            {
                "trigger_workload": trigger,
                "victim_workload": victim,
                **asdict(snapshot),
                "useful_ratio": snapshot.useful_ratio,
            }
        )
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "stats": stats,
    }


def write_workload_eviction_report(path: str, block_pool: BlockPool) -> dict[str, Any]:
    """Write summary and event records as JSONL; return the summary after closing."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = build_workload_eviction_report(block_pool)
    report["evictions_path"] = str(destination)
    with destination.open("w", encoding="utf-8") as output:
        # Keep the summary on disk as well, before the client resets the cache.
        output.write(json.dumps({"type": "summary", **report}, allow_nan=False) + "\n")
        for event in block_pool.get_workload_eviction_events():
            record = {
                "type": "eviction",
                **asdict(event),
                "block_hash": event.block_hash.hex(),
            }
            output.write(json.dumps(record, allow_nan=False) + "\n")
    return report

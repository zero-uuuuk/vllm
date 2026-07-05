# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workload tag inference for QuotaServe PR0 eviction attribution."""

import re

_ENGINE_PREFIXES = ("chatcmpl-", "cmpl-")
_SPLIT_RE = re.compile(r"[-_]")

_TOKEN_TO_WORKLOAD = {
    "agent": "agent",
    "chat": "chat",
    "hotpotqa": "longctx",
    "longctx": "longctx",
    "msmarco": "rag",
    "rag": "rag",
}


def infer_workload(request_id: str | None) -> str:
    """Infer the workload tag from an OpenAI request id."""
    if not request_id:
        return "unknown"

    rid = request_id
    for prefix in _ENGINE_PREFIXES:
        if rid.startswith(prefix):
            rid = rid[len(prefix) :]
            break

    token = _SPLIT_RE.split(rid, maxsplit=1)[0].lower()
    return _TOKEN_TO_WORKLOAD.get(token, "unknown")

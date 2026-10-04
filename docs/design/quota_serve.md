# QuotaServe: session demand and shared KV cache quotas

```bash
EVICTION_POLICY=quotaserve QUOTASERVE_APPS='["chat","agent"]' \
  QUOTASERVE_SESSION_IDLE_TTL_S=60 \
  vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details
```

`EVICTION_POLICY` selects `lru` (default), `lfu`, `fifo`, or `quotaserve`.
LRU uses the native free queue. FIFO orders cached blocks by insertion; LFU
counts prefix acquisitions and breaks frequency ties by recency. All policies
reclaim only blocks with no active request references and prefer uncached space.

QuotaServe requires prefix caching and a JSON list of unique, nonempty app IDs.
Requests supply `application_id` and `session_id` through `vllm_xargs`.
Unregistered apps can borrow capacity but have no protected quota. App cache
salts apply to every policy in the comparison.

## Session demand

The scheduler registers sessions when requests arrive, including queued
requests. On normal completion, it counts the request's unique, non-null
physical KV blocks before release. Cached hits and partially filled blocks
count toward this footprint. Aborted requests do not update the footprint mean.
For sliding-window and hybrid models, the observation is the final resident
footprint, not necessarily the peak.

Each session retains its latest completed footprint while waiting for the next
turn. For full-attention cache groups, the scheduler also estimates blocks from
the arrived prompt, accounting for context-parallel block size. The larger of
that estimate and the latest completed footprint enters the session's demand.
This uses an already received request, without future-turn information. Other
cache types use the completed-footprint estimate.

Apps maintain a request-footprint EWMA with weight 0.2 for new samples. Sessions
without a footprint or arrived-prompt estimate use this mean. An unobserved app
uses the mean of observed apps; before any observations the prior is usable
capacity divided by registered app count. The app score sums known session
demands and prior estimates for unknown sessions, with at least one prior share
for an app that has no tracked sessions. Scores round up to whole blocks.

If the total score fits in capacity, quotas equal scores and unused capacity
remains shared. Otherwise, scores scale proportionally and use largest-remainder
rounding. These are soft quotas, not physical reservations or guarantees for
individual session prefixes.

## Think time and session retirement

A session becomes idle when its last unfinished request completes. A subsequent
request from a still-tracked idle session updates its app's think-time EWMA and
EWMA variance. With weight `alpha = 0.2`, old mean `G`, variance `V`, and new gap
`g`, the updates are:

```text
V <- (1 - alpha) * (V + alpha * (g - G)^2)
G <- (1 - alpha) * G + alpha * g
```

The first gap initializes the mean and zero variance. Idle timeout is
`max(5 seconds, 3 * G)` once a gap is observed; before that it uses
`QUOTASERVE_SESSION_IDLE_TTL_S`. Queued or running sessions do not expire.
Expiration is checked during policy updates, without a background timer.

The policy retains block references from each session's latest completed
context. A timeout or a newly completed context can retire older references.
Only valid, unreferenced cached blocks no longer retained by another tracked
session enter the retired-block LRU queue. A new acquisition removes the block
from eviction candidates.

## Block allocation and eviction

Resident occupancy includes both pinned blocks and unreferenced cached blocks.
A cache hit does not reduce occupancy. Uncached blocks stop counting when their
last request reference is released. Allocation proceeds one block at a time:

1. Use uncached free space.
2. Reclaim the oldest retired cached block, if available
   (`selection="quotaserve_retired"`).
3. Find owners with evictable cache and
   `resident[app] + (app == requester) > quota[app]`. Choose the oldest LRU head
   across these owners, independently of the amount of quota excess.
4. Before reclaiming that candidate, compare its time since release with
   `min(idle_timeout, G + 2 * sqrt(V))` for its owner. Within this reuse interval,
   use global LRU instead (`quotaserve_reuse`). Otherwise reclaim the quota
   candidate (`quotaserve`). Without a gap observation the interval is zero.
5. If no owner has an eligible quota-excess block, use global LRU
   (`quotaserve_pressure`) so requests can continue.

The reuse interval is a heuristic based on observed gaps, not a statistical
coverage guarantee. Both global-LRU paths can reclaim cache within another
app's quota. Neither path changes request references or permits reclaiming
pinned blocks. No app-specific think time is hardcoded.

## Validation and GPU replay

```bash
.venv/bin/python -m pytest tests/v1/core/test_quota_serve.py \
  tests/v1/core/test_ranked_eviction.py -q
```

The sibling research repository contains
`004_serving_experiments/runner.py`. Its GPU configuration uses
Llama-3.2-3B-Instruct on an A10G, fixed KV memory, and a 64-request execution
limit. The saved input has 60 Chat sessions and 120 Agent sessions, each with
10 original turns. Chat waits 20 seconds and Agent waits 1 second after each
response. Both reuse original contexts and generate the original response's
token length. See that directory's README and `results/gpu_comparison.md` for
conditions, raw measurements, repeat counts, and limitations.

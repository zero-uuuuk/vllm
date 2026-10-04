# QuotaServe: dynamic mean-demand guarantees

```bash
EVICTION_POLICY=quotaserve QUOTASERVE_APPS='["chat","agent"]' \
  vllm serve <model> --enable-prefix-caching --enable-prompt-tokens-details
```

The default `lru` retains the native free queue. QuotaServe requires prefix
caching and an explicit JSON list of registered applications. App IDs must be
unique nonempty strings. Unregistered apps can borrow free capacity but have no
protected quota.

## One demand signal

On normal request completion, before freeing its blocks, the scheduler counts
unique non-null physical block IDs still attached to the request. This includes
cached hits and partial blocks. For full attention this is the final context's
footprint; for sliding-window/hybrid models it is the final resident footprint,
not necessarily peak usage.

The first sample initializes the app mean. Subsequent samples use
`D[app] = 0.8 * D[app] + 0.2 * request_blocks`. Aborted/error requests and zero
footprints are excluded. No wall-clock decay, request rate, or cache-miss rate
enters the score. Repeating the same-sized Agent request does not increase its
mean simply because it arrives more often.

Unobserved apps use the observed-app mean as a prior. Before any observations,
each app receives `capacity / app_count` as its bootstrap demand. Round each
mean up to blocks. If their sum fits, these demands are the minimum guarantees;
the remaining capacity is shared. If the sum exceeds capacity, scale demands
proportionally and use largest-remainder rounding. At 96 usable blocks, demands
Chat=57 and Agent=33 give guarantees 57/33 and 6 shared blocks. At capacity 80,
they give guarantees 51/29. Integer changes are logged as
`QuotaServe mean-demand quotas`.

## Physical accounting and allocation

Resident occupancy includes pinned allocations and unreferenced full cached
blocks. Pinning a hit does not reduce occupancy. Uncached blocks stop counting
when their final reference is released. App cache salts apply in both policies.

`KVCacheManager.allocate_slots` passes the app ID through the coordinator and
single-type managers to `BlockPool.get_new_blocks`, including external-computed
and Mamba paths. The pool accounts each block before choosing the next victim.

1. Use uncached free capacity.
2. For owners with evictable cache, compute
   `resident[app] + (app == requester) - quota[app]`.
   Owners with positive excess are eligible.
3. Choose the globally oldest app-local LRU head among eligible owners.
   This preserves recency in shared space while respecting minimum guarantees.
4. If no evictable excess exists, use global LRU and count
   `selection="quotaserve_pressure"`. Normal reclamation is `quotaserve`.

With an excess victim available, another app at or below its current quota
cannot be chosen. Idle capacity is borrowable. Fully pinned excess may require
breaking protection; this is not a hard reservation. Mean demand survives think
time. Guarantees scale down proportionally only when their total exceeds
capacity. Cache reset clears observations and occupancy and restores equal bootstrap quotas.

Selection costs O(apps) per allocated block. Quota updates cost O(apps log apps)
per completion. There is no background timer or controller lock. The footprint
mean does not model concurrent session count or a full reusable working set.
Inactive registered apps retain their last mean; lifecycle retirement is not
implemented.

## Validation

```bash
.venv/bin/python -m pytest tests/v1/core/test_quota_serve.py \
  tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_single_type_kv_cache_manager.py -q
```

The sibling research repo's `002_quotaserve_experiment/cpu_workload_compare.py` runs real
HTTP CPU inference. Its default has **both apps reusing prefixes**, with Chat
20 seconds and Agent 1 second between completion and next request. See
`DIALOGUE_REPORT.md` for original conversation replay conditions and results.

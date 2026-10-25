# UniCache baseline and QuotaServe ablations

`EVICTION_POLICY=unicache` ports the multi-turn Chat/Agent paths of public
`SimpleHYBRID` from [xsyslab/UniCache, bc51340](https://github.com/xsyslab/UniCache/tree/bc51340e3d8a623ecdfc6372ed8b64ec978f1b15).
Chat maps to `MULTI_TURN_TEXT`; Agent maps to `MULTI_TURN_AGENTIC`. Other task
types fail explicitly. Single-turn and future-aware OPT policies are outside
this experiment's scope.

The adapter retains the published CDF parameters, integer elapsed seconds,
within-queue recency/position ordering, cross-queue score/tie rules and weight
update formula. Admitted-request statistics retain 500 observations per task, decay
other tasks' histories by 0.95 on each first-scheduled request, and update
weights at eviction once per multiple of ten such requests. The public
implementation's queue-length occupancy calculation is retained.

The V1 interface supplies the prefix length on cache insertion and the hit
token count after scheduling admission. Repeated lookups on the same live request are
deduplicated. The final reference release is the last-use timestamp available
at this seam; no simulator globals or future request information are used.
Only unreferenced blocks are candidates, and uncached free space takes priority.
Reset clears statistics, queue weights and candidate metadata.

Three registered policies use the same `QuotaServeAdapter`:

| Policy | Removed behavior | Preserved behavior |
| --- | --- | --- |
| `quotaserve_active_only` | Waiting sessions in demand aggregation | Lifecycle, gaps, fallback demand, block links |
| `quotaserve_no_retired` | Retired-block priority | Retirement tracking, ordinary candidate eligibility |
| `quotaserve_no_reuse` | Return-gap guard | Quotas, retirement, gap tracking, pressure fallback |

`quotaserve` retains the existing full policy. No combinations of ablations
are exposed. The same application cache salts and BlockPool allocation rules
apply to every policy.

Policy tests run without GPU inference. For upstream parity, download
`vllm/core/evictor.py` and `vllm/simulator/output_simulator.py` from the pinned
revision into one directory, then run:

```sh
UNICACHE_REFERENCE_DIR=/path/to/reference .venv/bin/python -m pytest \
  tests/v1/core/test_quota_serve.py tests/v1/core/test_ranked_eviction.py \
  tests/v1/core/test_unicache.py -q
```

The parity test executes the actual upstream methods and compares victims,
weights and hit histories over a deterministic event sequence.

# n-autoresearch

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Same idea — agent modifies train.py, trains for 5 minutes, keeps or discards, repeats — but with structured experiment state, multi-GPU parallelism, adaptive search, and crash recovery via [iii-engine](https://github.com/iii-hq/iii-engine) (Worker/Function/Trigger).

The agent is still external. Claude, Codex, whatever you want. This repo is the infrastructure that replaces the bash loop, git-as-state, and flat TSV with queryable experiment tracking across N GPUs.

## How it works

Two workers talk to iii-engine:

    Orchestrator (Python) — 26 functions for experiment tracking, search strategy, guidance memory, GPU pool, reporting
    GPU Worker (Rust) — one per GPU, executes uv run train.py, parses metrics, handles timeouts

The external agent calls the same uv run train.py but wraps it with REST API calls:

    POST /api/guidance/brief        — get full briefing before starting (memory + state + strategy)
    POST /api/experiment/register   — record hypothesis before training
    POST /api/experiment/complete   — record metrics after, auto-decides keep/discard
    POST /api/guidance/record       — record an insight the agent discovered
    POST /api/report/summary        — full stats for a run tag

Everything else stays the same. train.py is the only file agents modify. prepare.py is read-only. 5-minute fixed time budget. val_bpb is the metric.

## Quick start

Requirements: NVIDIA GPU(s), Python 3.10+, uv, Rust 1.82+.

    # 1. Install iii-engine
    curl -fsSL https://install.iii.dev | sh

    # 2. Clone
    git clone https://github.com/iii-hq/n-autoresearch.git
    cd n-autoresearch

    # 3. Install Python dependencies
    uv sync

    # 4. Start iii-engine
    iii --config iii-config.yaml

    # 5. Start orchestrator (new terminal)
    uv run python workers/orchestrator/orchestrator.py

    # 6. Start GPU worker (new terminal, one per GPU)
    cd workers/gpu
    GPU_INDEX=0 REPO_DIR=/path/to/n-autoresearch cargo run --release

    # For multiple GPUs:
    GPU_INDEX=1 REPO_DIR=/path/to/n-autoresearch cargo run --release

    # 7. Download data and train tokenizer (one-time)
    uv run prepare.py

    # 8. Point your agent at program.md and go
    # e.g. in Claude Code: "read program.md and kick off a new experiment"

## What the agent does

Same loop as autoresearch, but with API calls for tracking:

    1. curl POST /api/experiment/setup          — init run tag
    2. curl POST /api/guidance/brief            — get briefing (memory + strategy + warnings)
    3. edit train.py                            — the experiment
    4. git commit
    5. curl POST /api/experiment/register       — record hypothesis
    6. uv run train.py > run.log 2>&1           — train (5 min)
    7. curl POST /api/experiment/complete       — record results
       response: { improved: true, action: "keep_commit" }
       or:       { improved: false, action: "git_reset" }
    8. repeat from 2

If training crashes:

    curl POST /api/experiment/crash             — tracks consecutive crashes
    response: { consecutive_crashes: 2, should_abort: false }

## Functions (28)

    experiment::setup           init tag + branch + strategy
    experiment::register        record hypothesis before training
    experiment::complete        record metrics, auto keep/discard, detect near-misses
    experiment::crash           track consecutive crashes, abort after 3
    experiment::history         query by tag/status/limit
    experiment::best            current best for a tag
    experiment::near_misses     experiments within 0.002 BPB of best

    search::strategy            get current mode (explore/exploit/combine/ablation)
    search::set_strategy        manual override
    search::adapt               auto-adapt from experiment history
    search::suggest_direction   category stats, underexplored areas, concrete suggestions

    guidance::synthesize        auto-extract patterns from experiment history into memory
    guidance::memory            read accumulated insights for a tag
    guidance::record            agent records an insight (dead end, observation, hardware limit)
    guidance::delete            remove an outdated insight by index
    guidance::brief             full briefing: memory + current best + strategy + recent experiments

    pool::register_gpu          GPU worker self-registers on startup
    pool::heartbeat             30s heartbeat, offline after 60s stale
    pool::list                  all GPUs with status
    pool::acquire               atomic claim of idle GPU
    pool::release               return GPU to pool
    pool::deregister            remove on shutdown

    report::summary             full stats, BPB progression, category breakdown
    report::tsv                 export in original autoresearch TSV format
    report::diff                compare two experiments
    report::tags                list all run tags

    gpu::train                  execute training, parse metrics, enforce timeout
    gpu::health                 nvidia-smi temperature/memory/utilization

## State (KV)

    experiments:{id}    full experiment object (hypothesis, metrics, status, diff)
    lineage:{tag}       ordered array of experiment IDs
    best:{tag}          current best val_bpb + commit + experiment_id
    near_misses:{id}    experiments that almost improved (delta < 0.002)
    gpu_pool:{gpu_id}   GPU worker status (idle/training/offline)
    strategy:{tag}      search mode + temperature + reason
    tags:{name}         run metadata (total/kept experiments, best BPB)
    crashes:{tag}       consecutive crash count
    guidance:{tag}      accumulated insights (dead ends, patterns, hardware limits)

## Search adaptation

Strategy auto-adapts after each experiment based on recent history:

    explore     (default) broad random changes, try underexplored categories
    exploit     refine around best config, small incremental tweaks
    combine     merge two near-miss experiments that improved different aspects
    ablation    systematically remove components to find what matters

Transitions:
    crash rate > 50%                    -> exploit (conservative)
    plateau + near-misses available     -> combine
    plateau + no near-misses            -> ablation
    keep rate > 30%                     -> exploit
    default                             -> explore

## Guidance agent

Long-running experiment loops get stuck in local optima. The agent keeps refining a narrow approach instead of stepping back and trying something better. Guidance solves this with persistent cross-session memory.

    curl -X POST localhost:3111/api/guidance/brief -d '{"tag":"run-001"}'
    {
      "best": { "val_bpb": 1.0412 },
      "strategy": "exploit",
      "guidance": {
        "dead_ends": ["attention changes never improved BPB across 8 attempts"],
        "high_yield": ["optimizer changes are productive (5/9 kept)"],
        "warnings": ["OOM crashes observed — models above ~124M params may not fit"],
        "observations": ["BPB plateau detected: last 4 improvements within 0.001"]
      }
    }

How it works:
- `guidance::synthesize` runs automatically every 5 experiments, extracting patterns from history
- The agent can also manually record insights via `guidance::record`
- Before each experiment, the agent calls `guidance::brief` to get a full briefing
- Dead ends, hardware limits, and crash patterns persist across sessions

Insight types:
    dead_end         category that never improves (>= 5 attempts, 0 kept)
    high_yield       category with > 40% keep rate
    unstable         category that crashes > 50% of the time
    recurring_crash  same error message appears 2+ times
    hardware_limit   OOM pattern with param count threshold
    plateau          recent improvements within 0.001 BPB
    observation      manually recorded by the agent

## Multi-GPU

N GPU workers = N parallel experiments on the same tag. Each agent acquires a GPU, trains, records results, releases. Search strategy adapts globally.

    curl localhost:3111/api/pool/list
    { "total": 8, "idle": 5, "training": 3 }

    curl -X POST localhost:3111/api/pool/acquire -d '{"experiment_id":"exp-xxx"}'
    { "acquired": true, "gpu_id": "gpu-3", "gpu_index": 3 }

    CUDA_VISIBLE_DEVICES=3 uv run train.py > run.log 2>&1

    curl -X POST localhost:3111/api/pool/release -d '{"gpu_id":"gpu-3"}'

## Project structure

    iii-config.yaml                         iii-engine runtime config
    program.md                              agent instructions
    prepare.py                              data prep + eval (read-only)
    train.py                                model + optimizer + loop (agent modifies)
    workers/
      orchestrator/
        orchestrator.py                     Python worker — 26 functions, 26 triggers
      gpu/                                  Rust worker (one per GPU)
        src/
          main.rs                           init, GPU detection, pool registration
          config.rs                         env config
          state.rs                          StateKV wrapper
          functions/train.rs                gpu::train + gpu::health
          triggers/mod.rs                   HTTP + cron triggers

## What stays the same vs autoresearch

Same:
    - train.py is the only file agents modify
    - prepare.py is read-only
    - 5-minute fixed time budget
    - val_bpb as the single metric
    - git branches (autoresearch/<tag>)
    - external agents drive the loop
    - program.md as agent instructions

Different:
    - structured KV state instead of results.tsv
    - multi-GPU parallel experiments
    - adaptive search strategy
    - crash recovery with consecutive tracking
    - near-miss detection for combination strategies
    - queryable experiment history with category analysis
    - guidance agent with persistent cross-session memory
    - TSV export for backwards compatibility

## License

Apache-2.0

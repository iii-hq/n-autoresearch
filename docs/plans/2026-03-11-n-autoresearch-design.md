# n-autoresearch Design

## Overview

Autonomous ML research infrastructure built on iii-engine (Worker/Function/Trigger).
Same philosophy as Karpathy's autoresearch — agent modifies train.py, trains for
5 min, keep or discard — but with structured state, multi-GPU parallelism, adaptive
search strategy, and crash recovery.

## Architecture

```
                  External Agents (Claude Code, Codex, etc.)
                              │
                              │ HTTP REST API
                              ▼
┌────────────────────────────────────────────────────────┐
│                     iii-engine                          │
│                                                        │
│  ┌─────────────────────┐  ┌──────────────────────────┐ │
│  │  Orchestrator (TS)  │  │     GPU Workers (Rust)   │ │
│  │                     │  │                          │ │
│  │  experiment::*  (7) │  │  gpu::train              │ │
│  │  search::*      (4) │  │  gpu::health             │ │
│  │  pool::*        (6) │  │                          │ │
│  │  report::*      (4) │  │  × N GPUs               │ │
│  └─────────┬───────────┘  └────────────┬─────────────┘ │
│            │                           │               │
│            └───────────┬───────────────┘               │
│                        │                               │
│  ┌─────────────────────┴──────────────────────────┐   │
│  │              KV State Store                     │   │
│  │                                                 │   │
│  │  experiments:{id}  lineage:{tag}  best:{tag}   │   │
│  │  near_misses:{id}  gpu_pool:{id}  strategy:{t} │   │
│  │  tags:{name}       crashes:{tag}               │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

## Workers

### Orchestrator (TypeScript)
- 21 functions across 4 domains
- 20 HTTP triggers (REST API)
- 2 event triggers (experiment.completed, experiment.crashed)
- 1 cron trigger (GPU pool health check)

### GPU Worker (Rust)
- Per-GPU process, one per NVIDIA GPU
- Spawns `uv run train.py` with CUDA_VISIBLE_DEVICES isolation
- Parses stdout metrics (val_bpb, peak_vram_mb, etc.)
- Timeout + crash handling
- Heartbeat for pool liveness

## Functions

| ID | Domain | Description |
|---|---|---|
| experiment::setup | Experiment | Initialize run tag + branch |
| experiment::register | Experiment | Register experiment before training |
| experiment::complete | Experiment | Record results, decide keep/discard |
| experiment::crash | Experiment | Record crash, track consecutive |
| experiment::history | Experiment | Query experiment history |
| experiment::best | Experiment | Get current best result |
| experiment::near_misses | Experiment | Get near-miss experiments |
| search::strategy | Search | Get current search strategy |
| search::set_strategy | Search | Override strategy |
| search::adapt | Search | Auto-adapt based on history |
| search::suggest_direction | Search | Suggest what to try next |
| pool::register_gpu | Pool | Register GPU worker |
| pool::heartbeat | Pool | Worker heartbeat |
| pool::list | Pool | List all GPUs with status |
| pool::acquire | Pool | Acquire idle GPU |
| pool::release | Pool | Release GPU to pool |
| pool::deregister | Pool | Remove GPU from pool |
| report::summary | Report | Full run summary |
| report::tsv | Report | Export TSV (autoresearch compat) |
| report::diff | Report | Compare two experiments |
| report::tags | Report | List all run tags |
| gpu::train | GPU | Execute training run |
| gpu::health | GPU | GPU health check |

## Search Strategy

Auto-adaptive based on experiment history:

- **explore** (default): Broad random changes, try underexplored categories
- **exploit**: Refine around best-known configuration, small tweaks
- **combine**: Merge two near-miss experiments that improved different aspects
- **ablation**: Systematically remove components to find what matters

Transitions:
- High crash rate (>50%) → exploit (conservative)
- Plateau (0 keeps in 10+) with near-misses → combine
- Plateau without near-misses → ablation
- Good keep rate (>30%) → exploit
- Default → explore

## KV Scopes

| Scope | Key | Value |
|---|---|---|
| experiments | experiment ID | Full Experiment object |
| lineage | tag name | Ordered array of experiment IDs |
| best | tag name | BestResult (val_bpb, commit, id) |
| near_misses | experiment ID | NearMiss (bpb, delta, hypothesis) |
| gpu_pool | gpu ID | GpuWorker status object |
| strategy | tag name | SearchStrategy (mode, temp, reason) |
| tags | tag name | ExperimentTag metadata |
| crashes | tag name | Consecutive crash count |

## Compatibility

- train.py and prepare.py are identical to original autoresearch
- TSV export matches original format
- Git branch naming matches (`autoresearch/<tag>`)
- External agent workflow is the same loop, just with API calls for tracking

# n-autoresearch

Autonomous ML research on iii-engine. Same philosophy as autoresearch — you modify `train.py`, train for 5 minutes, keep or discard — but with structured experiment tracking, multi-GPU parallelism, and adaptive search via iii functions.

## Setup

1. **Agree on a run tag** with the user (e.g. `mar11`).
2. **Initialize the tag**:
   ```bash
   curl -X POST http://localhost:3111/api/experiment/setup -d '{"tag":"mar11"}'
   ```
3. **Create the git branch**: `git checkout -b autoresearch/<tag>` from current master.
4. **Read the in-scope files**:
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
5. **Verify data exists**: Check `~/.cache/autoresearch/` for data shards.
6. **Run baseline**: Register and run the first experiment unchanged.

## The Experiment Loop

The iii-engine REST API at `http://localhost:3111` tracks everything. The loop:

### 1. Get search guidance

```bash
curl -X POST http://localhost:3111/api/search/suggest -d '{"tag":"mar11"}'
```

Returns: current strategy (explore/exploit/combine/ablation), underexplored categories, high-yield categories, near-misses to combine, and concrete suggestions.

### 2. Modify train.py

Edit `train.py` with your experimental idea. Classify the change:
- `architecture` — model structure, layer types, attention mechanism
- `optimizer` — optimizer params, new optimizer, scheduling
- `hyperparams` — learning rates, batch sizes, warmup/cooldown
- `activation` — activation functions (ReLU, GELU, SiLU, etc.)
- `attention` — attention patterns, window sizes, head counts
- `embedding` — token embeddings, value embeddings, positional encoding
- `normalization` — RMSNorm, LayerNorm, placement
- `regularization` — dropout, weight decay, gradient clipping
- `scheduling` — LR scheduling, warmup, cooldown ratios
- `initialization` — weight init strategies
- `simplification` — removing components, reducing complexity
- `combination` — merging ideas from near-misses
- `ablation` — systematically removing one component

### 3. Git commit + Register

```bash
git add train.py && git commit -m "experiment: <description>"
COMMIT=$(git rev-parse --short HEAD)
```

```bash
curl -X POST http://localhost:3111/api/experiment/register -d '{
  "tag": "mar11",
  "hypothesis": "Increasing depth from 8 to 12 should lower BPB due to more representational capacity",
  "description": "increase depth 8->12",
  "category": "architecture",
  "commit_sha": "'$COMMIT'",
  "diff_summary": "DEPTH=8 -> DEPTH=12",
  "parent_id": null
}'
```

Save the returned `experiment_id`.

### 4. Run training

```bash
uv run train.py > run.log 2>&1
```

### 5. Extract results

```bash
grep "^val_bpb:\|^peak_vram_mb:\|^training_seconds:\|^total_tokens_M:\|^mfu_percent:\|^num_steps:\|^num_params_M:\|^depth:" run.log
```

### 6. Record completion

If training succeeded:
```bash
curl -X POST http://localhost:3111/api/experiment/complete -d '{
  "experiment_id": "<id>",
  "val_bpb": 0.997900,
  "peak_vram_mb": 45060.2,
  "training_seconds": 300.1,
  "total_tokens_m": 499.6,
  "mfu_percent": 39.80,
  "num_steps": 953,
  "num_params_m": 50.3,
  "depth": 8
}'
```

The response tells you: `improved: true/false`, `action: "keep_commit"` or `"git_reset"`, and the current `best_val_bpb`.

If training crashed:
```bash
curl -X POST http://localhost:3111/api/experiment/crash -d '{
  "experiment_id": "<id>",
  "error": "RuntimeError: CUDA out of memory..."
}'
```

### 7. Act on the decision

- If `action: "keep_commit"` — keep the git commit, advance.
- If `action: "git_reset"` — `git reset --hard HEAD~1` to revert.
- If `should_abort: true` (3+ consecutive crashes) — stop and rethink.

### 8. Repeat

Go back to step 1. The search strategy auto-adapts after each experiment.

## Multi-GPU Mode

If multiple GPU workers are running:

```bash
# Check available GPUs
curl http://localhost:3111/api/pool/list

# Acquire a GPU
curl -X POST http://localhost:3111/api/pool/acquire -d '{"experiment_id":"<id>"}'

# Run on specific GPU
CUDA_VISIBLE_DEVICES=<gpu_index> uv run train.py > run.log 2>&1

# Release GPU when done
curl -X POST http://localhost:3111/api/pool/release -d '{"gpu_id":"<id>"}'
```

Multiple agents can run in parallel, each on a different GPU.

## Monitoring

```bash
# Full summary
curl -X POST http://localhost:3111/api/report/summary -d '{"tag":"mar11"}'

# TSV export (original autoresearch format)
curl -X POST http://localhost:3111/api/report/tsv -d '{"tag":"mar11"}'

# Compare two experiments
curl -X POST http://localhost:3111/api/report/diff -d '{"experiment_a":"exp-xxx","experiment_b":"exp-yyy"}'

# Best result
curl -X POST http://localhost:3111/api/experiment/best -d '{"tag":"mar11"}'

# Near-misses (for combination strategy)
curl -X POST http://localhost:3111/api/experiment/near-misses -d '{"tag":"mar11"}'
```

## Rules

**What you CAN do:**
- Modify `train.py` — everything is fair game.
- Use the iii API to register, complete, and query experiments.
- Run multiple experiments in parallel on different GPUs.

**What you CANNOT do:**
- Modify `prepare.py`. Read-only.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**The goal is simple: get the lowest val_bpb.**

**Simplicity criterion**: A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**NEVER STOP**: Once the loop begins, do not pause to ask the human. You are autonomous. If you run out of ideas, call `search::suggest_direction` for guidance. The loop runs until the human interrupts you.

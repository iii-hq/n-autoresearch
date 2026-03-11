import os
import time
import asyncio
import signal
from datetime import datetime, timezone
from iii import register_worker, InitOptions, OtelConfig, Logger

logger = Logger("orchestrator")

VERSION = "0.1.0"
WS_URL = os.environ.get("III_WS_URL", "ws://localhost:49134")
WORKER_NAME = "n-autoresearch-orchestrator"
MAX_CONSECUTIVE_CRASHES = int(os.environ.get("MAX_CONSECUTIVE_CRASHES", "3"))
NEAR_MISS_THRESHOLD = float(os.environ.get("NEAR_MISS_THRESHOLD", "0.002"))

SCOPES = {
    "experiments": "experiments",
    "lineage": "lineage",
    "best": "best",
    "near_misses": "near_misses",
    "gpu_pool": "gpu_pool",
    "strategy": "strategy",
    "tags": "tags",
    "crashes": "crashes",
}

ALL_CATEGORIES = [
    "architecture", "optimizer", "hyperparams", "activation", "attention",
    "embedding", "normalization", "regularization", "scheduling",
    "initialization", "simplification", "combination", "ablation", "other",
]


def experiment_id():
    t = int(time.time() * 1000)
    r = os.urandom(4).hex()[:6]
    return f"exp-{t:x}-{r}"


class StateKV:
    def __init__(self, sdk):
        self.sdk = sdk

    async def get(self, scope, key):
        try:
            return await self.sdk.trigger("state::get", {"scope": scope, "key": key})
        except KeyError:
            return None
        except Exception:
            logger.error("state::get failed for %s:%s", scope, key, exc_info=True)
            raise

    async def set(self, scope, key, value):
        await self.sdk.trigger("state::set", {"scope": scope, "key": key, "value": value})

    async def list(self, scope):
        try:
            return await self.sdk.trigger("state::list", {"scope": scope})
        except KeyError:
            return []
        except Exception:
            logger.error("state::list failed for %s", scope, exc_info=True)
            raise

    async def delete(self, scope, key):
        await self.sdk.trigger("state::delete", {"scope": scope, "key": key})


_tag_locks = {}

def _tag_lock(tag):
    if tag not in _tag_locks:
        _tag_locks[tag] = asyncio.Lock()
    return _tag_locks[tag]


def register_experiment_functions(sdk, kv):
    @sdk.register_function({"id": "experiment::setup", "description": "Initialize a new experiment run tag."})
    async def setup(input):
        tag = input["tag"]
        existing = await kv.get(SCOPES["tags"], tag)
        if existing:
            return {"error": f"Tag '{tag}' already exists", "existing": existing}

        tag_data = {
            "name": tag,
            "branch": f"autoresearch/{tag}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "best_val_bpb": float("inf"),
            "total_experiments": 0,
            "kept_experiments": 0,
        }
        await kv.set(SCOPES["tags"], tag, tag_data)

        strategy = {
            "mode": "explore",
            "explore_ratio": 0.7,
            "temperature": 1.0,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": "initial exploration phase",
        }
        await kv.set(SCOPES["strategy"], tag, strategy)

        return {"tag": tag_data, "branch": tag_data["branch"]}

    @sdk.register_function({"id": "experiment::register", "description": "Register a new experiment before training starts."})
    async def register(input):
        eid = experiment_id()
        experiment = {
            "id": eid,
            "tag": input["tag"],
            "parent_id": input.get("parent_id"),
            "commit_sha": input["commit_sha"],
            "description": input["description"],
            "hypothesis": input["hypothesis"],
            "category": input["category"],
            "val_bpb": 0,
            "peak_vram_mb": 0,
            "training_seconds": 0,
            "total_tokens_m": 0,
            "mfu_percent": 0,
            "num_steps": 0,
            "num_params_m": 0,
            "depth": 0,
            "status": "running",
            "gpu_id": input.get("gpu_id", "gpu-0"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "diff_summary": input["diff_summary"],
            "error": None,
        }
        await kv.set(SCOPES["experiments"], eid, experiment)

        async with _tag_lock(input["tag"]):
            lineage = await kv.get(SCOPES["lineage"], input["tag"]) or []
            lineage.append(eid)
            await kv.set(SCOPES["lineage"], input["tag"], lineage)

        return {"experiment_id": eid, "status": "registered"}

    @sdk.register_function({"id": "experiment::complete", "description": "Record experiment results. Decides keep/discard automatically."})
    async def complete(input):
        exp = await kv.get(SCOPES["experiments"], input["experiment_id"])
        if not exp:
            return {"error": f"Experiment {input['experiment_id']} not found"}

        async with _tag_lock(exp["tag"]):
            best = await kv.get(SCOPES["best"], exp["tag"])
            improved = not best or input["val_bpb"] < best["val_bpb"]
            delta = best["val_bpb"] - input["val_bpb"] if best else 0

            for field in ["val_bpb", "peak_vram_mb", "training_seconds", "total_tokens_m", "mfu_percent", "num_steps", "num_params_m", "depth"]:
                exp[field] = input[field]
            exp["status"] = "keep" if improved else "discard"
            exp["finished_at"] = datetime.now(timezone.utc).isoformat()
            await kv.set(SCOPES["experiments"], exp["id"], exp)

            if improved:
                await kv.set(SCOPES["best"], exp["tag"], {
                    "experiment_id": exp["id"],
                    "val_bpb": input["val_bpb"],
                    "commit_sha": exp["commit_sha"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })

            if not improved and best and delta > -NEAR_MISS_THRESHOLD:
                await kv.set(SCOPES["near_misses"], exp["id"], {
                    "experiment_id": exp["id"],
                    "tag": exp["tag"],
                    "val_bpb": input["val_bpb"],
                    "delta": abs(delta),
                    "hypothesis": exp["hypothesis"],
                    "category": exp["category"],
                    "diff_summary": exp["diff_summary"],
                })

            tag = await kv.get(SCOPES["tags"], exp["tag"])
            if tag:
                tag["total_experiments"] += 1
                if improved:
                    tag["kept_experiments"] += 1
                    tag["best_val_bpb"] = input["val_bpb"]
                await kv.set(SCOPES["tags"], exp["tag"], tag)

            await kv.delete(SCOPES["crashes"], exp["tag"])

        sdk.trigger_void("search::adapt", {"tag": exp["tag"]})

        return {
            "experiment_id": exp["id"],
            "status": exp["status"],
            "val_bpb": input["val_bpb"],
            "improved": improved,
            "delta": delta,
            "best_val_bpb": input["val_bpb"] if improved else (best["val_bpb"] if best else None),
            "action": "keep_commit" if improved else "git_reset",
        }

    @sdk.register_function({"id": "experiment::crash", "description": "Record a crashed experiment."})
    async def crash(input):
        exp = await kv.get(SCOPES["experiments"], input["experiment_id"])
        if not exp:
            return {"error": f"Experiment {input['experiment_id']} not found"}

        exp["status"] = "crash"
        exp["error"] = input["error"]
        exp["finished_at"] = datetime.now(timezone.utc).isoformat()
        await kv.set(SCOPES["experiments"], exp["id"], exp)

        async with _tag_lock(exp["tag"]):
            crashes = await kv.get(SCOPES["crashes"], exp["tag"]) or 0
            consecutive = crashes + 1
            await kv.set(SCOPES["crashes"], exp["tag"], consecutive)

            tag = await kv.get(SCOPES["tags"], exp["tag"])
            if tag:
                tag["total_experiments"] += 1
                await kv.set(SCOPES["tags"], exp["tag"], tag)

        sdk.trigger_void("search::adapt", {"tag": exp["tag"]})

        return {
            "experiment_id": exp["id"],
            "status": "crash",
            "consecutive_crashes": consecutive,
            "should_abort": consecutive >= MAX_CONSECUTIVE_CRASHES,
            "action": "git_reset",
        }

    @sdk.register_function({"id": "experiment::history", "description": "Get experiment history for a tag."})
    async def history(input):
        all_exps = await kv.list(SCOPES["experiments"])
        filtered = [e for e in all_exps if e.get("tag") == input["tag"]]
        if input.get("status"):
            filtered = [e for e in filtered if e.get("status") == input["status"]]
        filtered.sort(key=lambda e: e.get("started_at", ""))
        if input.get("limit"):
            filtered = filtered[-input["limit"]:]
        return {"experiments": filtered, "total": len(filtered)}

    @sdk.register_function({"id": "experiment::best", "description": "Get current best result for a tag."})
    async def best(input):
        b = await kv.get(SCOPES["best"], input["tag"])
        if not b:
            return {"error": "No results yet", "tag": input["tag"]}
        exp = await kv.get(SCOPES["experiments"], b["experiment_id"])
        return {"best": b, "experiment": exp}

    @sdk.register_function({"id": "experiment::near_misses", "description": "Get near-miss experiments."})
    async def near_misses(input):
        all_nm = await kv.list(SCOPES["near_misses"])
        tag_nm = [n for n in all_nm if n.get("tag") == input["tag"]]
        filtered = sorted(tag_nm, key=lambda n: n.get("delta", 0))
        limit = input.get("limit", 20)
        return {"near_misses": filtered[:limit], "total": len(filtered)}


def register_search_functions(sdk, kv):
    @sdk.register_function({"id": "search::strategy", "description": "Get current search strategy for a tag."})
    async def strategy(input):
        s = await kv.get(SCOPES["strategy"], input["tag"])
        return s or {"mode": "explore", "explore_ratio": 0.7, "temperature": 1.0}

    @sdk.register_function({"id": "search::set_strategy", "description": "Override search strategy for a tag."})
    async def set_strategy(input):
        s = {
            "mode": input["mode"],
            "explore_ratio": input.get("explore_ratio", 0.7),
            "temperature": input.get("temperature", 1.0),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": input["reason"],
        }
        await kv.set(SCOPES["strategy"], input["tag"], s)
        return s

    @sdk.register_function({"id": "search::adapt", "description": "Auto-adapt search strategy based on experiment history."})
    async def adapt(input):
        all_exps = await kv.list(SCOPES["experiments"])
        tag_exps = [e for e in all_exps if e.get("tag") == input["tag"] and e.get("status") != "running"]

        if len(tag_exps) < 5:
            return {"mode": "explore", "reason": "too few experiments to adapt"}

        recent = tag_exps[-10:]
        keep_rate = sum(1 for e in recent if e.get("status") == "keep") / len(recent)
        crash_rate = sum(1 for e in recent if e.get("status") == "crash") / len(recent)
        all_nm = await kv.list(SCOPES["near_misses"])
        near_misses = [n for n in all_nm if n.get("tag") == input["tag"]]

        if crash_rate > 0.5:
            mode, temperature = "exploit", 0.3
            reason = f"high crash rate ({crash_rate*100:.0f}%), switching to conservative tweaks"
        elif keep_rate == 0 and len(tag_exps) > 20:
            if len(near_misses) >= 2:
                mode, temperature = "combine", 0.5
                reason = f"plateau with {len(near_misses)} near-misses, trying combinations"
            else:
                mode, temperature = "ablation", 0.3
                reason = "long plateau, switching to ablation to identify essential components"
        elif keep_rate > 0.3:
            mode, temperature = "exploit", 0.5
            reason = f"good keep rate ({keep_rate*100:.0f}%), exploiting current direction"
        else:
            mode, temperature = "explore", 0.8
            reason = "default exploration"

        s = {
            "mode": mode,
            "explore_ratio": 0.8 if mode == "explore" else 0.3,
            "temperature": temperature,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }
        await kv.set(SCOPES["strategy"], input["tag"], s)
        return s

    @sdk.register_function({"id": "search::suggest_direction", "description": "Suggest what to try next based on experiment history."})
    async def suggest(input):
        all_exps = await kv.list(SCOPES["experiments"])
        tag_exps = sorted(
            [e for e in all_exps if e.get("tag") == input["tag"] and e.get("status") != "running"],
            key=lambda e: e.get("started_at", ""),
        )

        strategy = await kv.get(SCOPES["strategy"], input["tag"])
        all_nm = await kv.list(SCOPES["near_misses"])
        near_misses = [n for n in all_nm if n.get("tag") == input["tag"]]

        category_counts = {}
        for exp in tag_exps:
            cat = exp.get("category", "other")
            if cat not in category_counts:
                category_counts[cat] = {"total": 0, "kept": 0}
            category_counts[cat]["total"] += 1
            if exp.get("status") == "keep":
                category_counts[cat]["kept"] += 1

        underexplored = [c for c in ALL_CATEGORIES[:11] if c not in category_counts or category_counts[c]["total"] < 3]
        high_yield = [c for c, v in category_counts.items() if v["total"] >= 3 and v["kept"] / v["total"] > 0.3]

        kept = [e for e in tag_exps if e.get("status") == "keep"]
        bpb_trend = [e["val_bpb"] for e in kept[-5:]]

        mode = strategy.get("mode", "explore") if strategy else "explore"
        suggestions = _build_suggestions(mode, underexplored, high_yield, near_misses, bpb_trend)

        return {
            "strategy": mode,
            "total_experiments": len(tag_exps),
            "category_stats": category_counts,
            "underexplored_categories": underexplored,
            "high_yield_categories": high_yield,
            "near_misses_available": len(near_misses),
            "near_miss_categories": list(set(n.get("category") for n in near_misses)),
            "recent_bpb_trend": bpb_trend,
            "suggestions": suggestions,
        }


def _build_suggestions(mode, underexplored, high_yield, near_misses, trend):
    suggestions = []
    if mode == "explore":
        if underexplored:
            suggestions.append(f"Try changes in underexplored categories: {', '.join(underexplored[:3])}")
        suggestions.append("Try a radical architectural change")
    elif mode == "exploit":
        if high_yield:
            suggestions.append(f"Double down on high-yield categories: {', '.join(high_yield)}")
        suggestions.append("Make small incremental tweaks to the current best config")
    elif mode == "combine":
        if len(near_misses) >= 2:
            pair = near_misses[:2]
            suggestions.append(f'Combine near-misses: "{pair[0].get("hypothesis")}" + "{pair[1].get("hypothesis")}"')
    elif mode == "ablation":
        suggestions.append("Remove one component at a time to identify what actually matters")
        suggestions.append("Try simplifying: fewer layers, simpler activations, remove value embeddings")

    if len(trend) >= 3 and trend[-1] >= trend[0]:
        suggestions.append("BPB trend is flat/worsening. Consider a strategy change.")
    return suggestions


def register_pool_functions(sdk, kv):
    acquire_lock = asyncio.Lock()

    @sdk.register_function({"id": "pool::register_gpu", "description": "Register a GPU worker in the pool."})
    async def register_gpu(input):
        worker = {
            "id": input["gpu_id"],
            "name": f"gpu-{input['gpu_index']}",
            "gpu_index": input["gpu_index"],
            "gpu_name": input["gpu_name"],
            "vram_mb": input["vram_mb"],
            "status": "idle",
            "current_experiment_id": None,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        }
        await kv.set(SCOPES["gpu_pool"], input["gpu_id"], worker)
        return {"registered": True, "worker": worker}

    @sdk.register_function({"id": "pool::heartbeat", "description": "GPU worker heartbeat."})
    async def heartbeat(input):
        worker = await kv.get(SCOPES["gpu_pool"], input["gpu_id"])
        if not worker:
            return {"error": "GPU worker not found"}
        worker["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        if worker["status"] == "offline":
            worker["status"] = "training" if worker.get("current_experiment_id") else "idle"
        await kv.set(SCOPES["gpu_pool"], input["gpu_id"], worker)
        return {"ok": True}

    @sdk.register_function({"id": "pool::list", "description": "List all GPU workers."})
    async def list_gpus(input):
        workers = await kv.list(SCOPES["gpu_pool"])
        now = time.time() * 1000
        stale_ms = 60_000
        for w in workers:
            elapsed = now - datetime.fromisoformat(w["last_heartbeat"]).timestamp() * 1000
            if elapsed > stale_ms and w["status"] != "offline":
                w["status"] = "offline"
                await kv.set(SCOPES["gpu_pool"], w["id"], w)
        return {
            "workers": workers,
            "total": len(workers),
            "idle": sum(1 for w in workers if w["status"] == "idle"),
            "training": sum(1 for w in workers if w["status"] == "training"),
            "offline": sum(1 for w in workers if w["status"] == "offline"),
        }

    @sdk.register_function({"id": "pool::acquire", "description": "Acquire an idle GPU for an experiment."})
    async def acquire(input):
        async with acquire_lock:
            workers = await kv.list(SCOPES["gpu_pool"])
            idle = next((w for w in workers if w["status"] == "idle"), None)
            if not idle:
                return {"acquired": False, "gpu_id": None, "reason": "no idle GPUs"}
            idle["status"] = "training"
            idle["current_experiment_id"] = input["experiment_id"]
            await kv.set(SCOPES["gpu_pool"], idle["id"], idle)
            return {"acquired": True, "gpu_id": idle["id"], "gpu_index": idle["gpu_index"]}

    @sdk.register_function({"id": "pool::release", "description": "Release a GPU back to idle."})
    async def release(input):
        worker = await kv.get(SCOPES["gpu_pool"], input["gpu_id"])
        if not worker:
            return {"error": "GPU worker not found"}
        if worker.get("current_experiment_id") != input.get("experiment_id"):
            return {"error": "experiment_id mismatch or GPU not leased by caller"}
        worker["status"] = "idle"
        worker["current_experiment_id"] = None
        await kv.set(SCOPES["gpu_pool"], input["gpu_id"], worker)
        return {"released": True}

    @sdk.register_function({"id": "pool::deregister", "description": "Remove a GPU worker from the pool."})
    async def deregister(input):
        await kv.delete(SCOPES["gpu_pool"], input["gpu_id"])
        return {"deregistered": True}


def register_report_functions(sdk, kv):
    @sdk.register_function({"id": "report::summary", "description": "Generate a summary report for a tag."})
    async def summary(input):
        tag = await kv.get(SCOPES["tags"], input["tag"])
        if not tag:
            return {"error": f"Tag '{input['tag']}' not found"}

        best = await kv.get(SCOPES["best"], input["tag"])
        strategy = await kv.get(SCOPES["strategy"], input["tag"])
        workers = await kv.list(SCOPES["gpu_pool"])

        all_exps = await kv.list(SCOPES["experiments"])
        tag_exps = sorted(
            [e for e in all_exps if e.get("tag") == input["tag"]],
            key=lambda e: e.get("started_at", ""),
        )

        status_counts = {"keep": 0, "discard": 0, "crash": 0, "running": 0}
        category_counts = {}
        for e in tag_exps:
            status_counts[e.get("status", "running")] += 1
            cat = e.get("category", "other")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        kept = [e for e in tag_exps if e.get("status") == "keep"]
        bpb_history = [
            {"id": e["id"], "val_bpb": e["val_bpb"], "description": e["description"], "category": e["category"], "at": e.get("finished_at")}
            for e in kept
        ]
        total_training_min = sum(e.get("training_seconds", 0) for e in tag_exps) / 60

        return {
            "tag": input["tag"],
            "branch": tag["branch"],
            "best": {"val_bpb": best["val_bpb"], "commit": best["commit_sha"], "experiment_id": best["experiment_id"]} if best else None,
            "stats": {
                "total": tag["total_experiments"],
                "kept": status_counts["keep"],
                "discarded": status_counts["discard"],
                "crashed": status_counts["crash"],
                "running": status_counts["running"],
                "keep_rate": status_counts["keep"] / tag["total_experiments"] if tag["total_experiments"] > 0 else 0,
            },
            "categories": category_counts,
            "bpb_progression": bpb_history,
            "total_training_minutes": round(total_training_min, 1),
            "strategy": strategy.get("mode", "unknown") if strategy else "unknown",
            "gpu_pool": {
                "total": len(workers),
                "idle": sum(1 for w in workers if w["status"] == "idle"),
                "training": sum(1 for w in workers if w["status"] == "training"),
            },
        }

    @sdk.register_function({"id": "report::tsv", "description": "Export experiment history as TSV."})
    async def tsv(input):
        all_exps = await kv.list(SCOPES["experiments"])
        tag_exps = sorted(
            [e for e in all_exps if e.get("tag") == input["tag"] and e.get("status") != "running"],
            key=lambda e: e.get("started_at", ""),
        )
        header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
        rows = []
        for e in tag_exps:
            sha = e["commit_sha"][:7]
            bpb = "0.000000" if e["status"] == "crash" else f"{e['val_bpb']:.6f}"
            mem = "0.0" if e["status"] == "crash" else f"{e['peak_vram_mb'] / 1024:.1f}"
            rows.append(f"{sha}\t{bpb}\t{mem}\t{e['status']}\t{e['description']}")
        return {"tsv": "\n".join([header] + rows), "count": len(rows)}

    @sdk.register_function({"id": "report::diff", "description": "Compare two experiments."})
    async def diff(input):
        a = await kv.get(SCOPES["experiments"], input["experiment_a"])
        b = await kv.get(SCOPES["experiments"], input["experiment_b"])
        if not a or not b:
            return {"error": "One or both experiments not found"}
        return {
            "a": {"id": a["id"], "val_bpb": a["val_bpb"], "description": a["description"], "category": a["category"], "num_params_m": a["num_params_m"], "peak_vram_mb": a["peak_vram_mb"]},
            "b": {"id": b["id"], "val_bpb": b["val_bpb"], "description": b["description"], "category": b["category"], "num_params_m": b["num_params_m"], "peak_vram_mb": b["peak_vram_mb"]},
            "delta_bpb": b["val_bpb"] - a["val_bpb"],
            "delta_params_m": b["num_params_m"] - a["num_params_m"],
            "delta_vram_mb": b["peak_vram_mb"] - a["peak_vram_mb"],
        }

    @sdk.register_function({"id": "report::tags", "description": "List all experiment run tags."})
    async def tags(input):
        all_tags = await kv.list(SCOPES["tags"])
        return {"tags": sorted(all_tags, key=lambda t: t.get("created_at", ""), reverse=True)}


def register_triggers(sdk):
    http_triggers = [
        ("/api/experiment/setup", "POST", "experiment::setup"),
        ("/api/experiment/register", "POST", "experiment::register"),
        ("/api/experiment/complete", "POST", "experiment::complete"),
        ("/api/experiment/crash", "POST", "experiment::crash"),
        ("/api/experiment/history", "POST", "experiment::history"),
        ("/api/experiment/best", "POST", "experiment::best"),
        ("/api/experiment/near-misses", "POST", "experiment::near_misses"),
        ("/api/search/strategy", "POST", "search::strategy"),
        ("/api/search/set-strategy", "POST", "search::set_strategy"),
        ("/api/search/adapt", "POST", "search::adapt"),
        ("/api/search/suggest", "POST", "search::suggest_direction"),
        ("/api/pool/register", "POST", "pool::register_gpu"),
        ("/api/pool/heartbeat", "POST", "pool::heartbeat"),
        ("/api/pool/list", "GET", "pool::list"),
        ("/api/pool/acquire", "POST", "pool::acquire"),
        ("/api/pool/release", "POST", "pool::release"),
        ("/api/report/summary", "POST", "report::summary"),
        ("/api/report/tsv", "POST", "report::tsv"),
        ("/api/report/diff", "POST", "report::diff"),
        ("/api/report/tags", "GET", "report::tags"),
    ]
    for path, method, fn in http_triggers:
        sdk.register_trigger({
            "trigger_type": "http",
            "function_id": fn,
            "config": {"api_path": path, "http_method": method},
        })

    sdk.register_trigger({
        "trigger_type": "cron",
        "function_id": "pool::list",
        "config": {"expression": "*/30 * * * * *"},
    })


def main():
    sdk = register_worker(WS_URL, InitOptions(
        worker_name=WORKER_NAME,
        otel=OtelConfig(
            enabled=True,
            service_name="n-autoresearch",
            service_version=VERSION,
            metrics_enabled=True,
        ),
    ))

    kv = StateKV(sdk)

    register_experiment_functions(sdk, kv)
    register_search_functions(sdk, kv)
    register_pool_functions(sdk, kv)
    register_report_functions(sdk, kv)
    register_triggers(sdk)

    print(f"n-autoresearch orchestrator v{VERSION}")
    print(f"Connected to iii-engine at {WS_URL}")
    print(f"REST API at http://localhost:{os.environ.get('III_REST_PORT', '3111')}")
    print("Functions: 21 | Triggers: 21 | Ready.")

    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def shutdown(*_):
        print("Shutting down orchestrator...")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    loop.run_until_complete(stop.wait())
    loop.run_until_complete(sdk.shutdown())


if __name__ == "__main__":
    main()

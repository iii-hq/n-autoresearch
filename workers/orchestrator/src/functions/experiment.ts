import { StateKV } from "../state/kv.js";
import {
  SCOPES,
  experimentId,
  type Experiment,
  type BestResult,
  type NearMiss,
  type ExperimentCategory,
  type ExperimentTag,
} from "../state/schema.js";
import { CONFIG } from "../config.js";

type SDK = {
  registerFunction(
    opts: { id: string; description: string },
    handler: (input: any) => Promise<any>
  ): void;
  invokeFunction<I, O>(fn: string, input: I, timeout?: number): Promise<O>;
};

export function registerExperimentFunctions(sdk: SDK, kv: StateKV) {
  sdk.registerFunction(
    {
      id: "experiment::setup",
      description:
        "Initialize a new experiment run tag. Creates branch, results tracking, baseline.",
    },
    async (input: { tag: string }) => {
      const { tag } = input;
      const existing = await kv.get<ExperimentTag>(SCOPES.TAGS, tag);
      if (existing) {
        return { error: `Tag '${tag}' already exists`, existing };
      }

      const tagData: ExperimentTag = {
        name: tag,
        branch: `autoresearch/${tag}`,
        created_at: new Date().toISOString(),
        best_val_bpb: Infinity,
        total_experiments: 0,
        kept_experiments: 0,
      };
      await kv.set(SCOPES.TAGS, tag, tagData);

      await kv.set<SearchStrategy>(SCOPES.STRATEGY, tag, {
        mode: "explore",
        explore_ratio: 0.7,
        temperature: 1.0,
        updated_at: new Date().toISOString(),
        reason: "initial exploration phase",
      });

      return { tag: tagData, branch: tagData.branch };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::register",
      description:
        "Register a new experiment before training starts. Returns experiment ID for tracking.",
    },
    async (input: {
      tag: string;
      hypothesis: string;
      description: string;
      category: ExperimentCategory;
      commit_sha: string;
      diff_summary: string;
      parent_id?: string;
      gpu_id?: string;
    }) => {
      const id = experimentId();
      const experiment: Experiment = {
        id,
        tag: input.tag,
        parent_id: input.parent_id ?? null,
        commit_sha: input.commit_sha,
        description: input.description,
        hypothesis: input.hypothesis,
        category: input.category,
        val_bpb: 0,
        peak_vram_mb: 0,
        training_seconds: 0,
        total_tokens_m: 0,
        mfu_percent: 0,
        num_steps: 0,
        num_params_m: 0,
        depth: 0,
        status: "running",
        gpu_id: input.gpu_id ?? "gpu-0",
        started_at: new Date().toISOString(),
        finished_at: null,
        diff_summary: input.diff_summary,
        error: null,
      };

      await kv.set(SCOPES.EXPERIMENTS, id, experiment);

      const lineage = (await kv.get<string[]>(SCOPES.LINEAGE, input.tag)) ?? [];
      lineage.push(id);
      await kv.set(SCOPES.LINEAGE, input.tag, lineage);

      return { experiment_id: id, status: "registered" };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::complete",
      description:
        "Record experiment results after training finishes. Decides keep/discard automatically.",
    },
    async (input: {
      experiment_id: string;
      val_bpb: number;
      peak_vram_mb: number;
      training_seconds: number;
      total_tokens_m: number;
      mfu_percent: number;
      num_steps: number;
      num_params_m: number;
      depth: number;
    }) => {
      const exp = await kv.get<Experiment>(
        SCOPES.EXPERIMENTS,
        input.experiment_id
      );
      if (!exp) {
        return { error: `Experiment ${input.experiment_id} not found` };
      }

      const best = await kv.get<BestResult>(SCOPES.BEST, exp.tag);
      const improved = !best || input.val_bpb < best.val_bpb;
      const delta = best ? best.val_bpb - input.val_bpb : 0;

      exp.val_bpb = input.val_bpb;
      exp.peak_vram_mb = input.peak_vram_mb;
      exp.training_seconds = input.training_seconds;
      exp.total_tokens_m = input.total_tokens_m;
      exp.mfu_percent = input.mfu_percent;
      exp.num_steps = input.num_steps;
      exp.num_params_m = input.num_params_m;
      exp.depth = input.depth;
      exp.status = improved ? "keep" : "discard";
      exp.finished_at = new Date().toISOString();

      await kv.set(SCOPES.EXPERIMENTS, exp.id, exp);

      if (improved) {
        await kv.set<BestResult>(SCOPES.BEST, exp.tag, {
          experiment_id: exp.id,
          val_bpb: input.val_bpb,
          commit_sha: exp.commit_sha,
          updated_at: new Date().toISOString(),
        });
      }

      if (
        !improved &&
        best &&
        delta > -CONFIG.NEAR_MISS_THRESHOLD
      ) {
        await kv.set<NearMiss>(SCOPES.NEAR_MISSES, exp.id, {
          experiment_id: exp.id,
          val_bpb: input.val_bpb,
          delta: Math.abs(delta),
          hypothesis: exp.hypothesis,
          category: exp.category,
          diff_summary: exp.diff_summary,
        });
      }

      const tag = await kv.get<ExperimentTag>(SCOPES.TAGS, exp.tag);
      if (tag) {
        tag.total_experiments += 1;
        if (improved) {
          tag.kept_experiments += 1;
          tag.best_val_bpb = input.val_bpb;
        }
        await kv.set(SCOPES.TAGS, exp.tag, tag);
      }

      await kv.delete(SCOPES.CRASHES, exp.tag);

      return {
        experiment_id: exp.id,
        status: exp.status,
        val_bpb: input.val_bpb,
        improved,
        delta,
        best_val_bpb: improved ? input.val_bpb : best?.val_bpb,
        action: improved ? "keep_commit" : "git_reset",
      };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::crash",
      description:
        "Record a crashed experiment. Tracks consecutive crashes per tag.",
    },
    async (input: {
      experiment_id: string;
      error: string;
    }) => {
      const exp = await kv.get<Experiment>(
        SCOPES.EXPERIMENTS,
        input.experiment_id
      );
      if (!exp) {
        return { error: `Experiment ${input.experiment_id} not found` };
      }

      exp.status = "crash";
      exp.error = input.error;
      exp.finished_at = new Date().toISOString();
      await kv.set(SCOPES.EXPERIMENTS, exp.id, exp);

      const crashes =
        (await kv.get<number>(SCOPES.CRASHES, exp.tag)) ?? 0;
      const consecutive = crashes + 1;
      await kv.set(SCOPES.CRASHES, exp.tag, consecutive);

      const tag = await kv.get<ExperimentTag>(SCOPES.TAGS, exp.tag);
      if (tag) {
        tag.total_experiments += 1;
        await kv.set(SCOPES.TAGS, exp.tag, tag);
      }

      return {
        experiment_id: exp.id,
        status: "crash",
        consecutive_crashes: consecutive,
        should_abort:
          consecutive >= CONFIG.MAX_CONSECUTIVE_CRASHES,
        action: "git_reset",
      };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::history",
      description:
        "Get full experiment history for a tag, ordered by start time.",
    },
    async (input: { tag: string; limit?: number; status?: string }) => {
      const all = await kv.list<Experiment>(SCOPES.EXPERIMENTS);
      let filtered = all.filter((e) => e.tag === input.tag);
      if (input.status) {
        filtered = filtered.filter((e) => e.status === input.status);
      }
      filtered.sort(
        (a, b) =>
          new Date(a.started_at).getTime() - new Date(b.started_at).getTime()
      );
      if (input.limit) {
        filtered = filtered.slice(-input.limit);
      }
      return { experiments: filtered, total: filtered.length };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::best",
      description: "Get current best result for a tag.",
    },
    async (input: { tag: string }) => {
      const best = await kv.get<BestResult>(SCOPES.BEST, input.tag);
      if (!best) {
        return { error: "No results yet", tag: input.tag };
      }
      const exp = await kv.get<Experiment>(
        SCOPES.EXPERIMENTS,
        best.experiment_id
      );
      return { best, experiment: exp };
    }
  );

  sdk.registerFunction(
    {
      id: "experiment::near_misses",
      description:
        "Get near-miss experiments that almost improved. Useful for combination strategies.",
    },
    async (input: { tag: string; limit?: number }) => {
      const all = await kv.list<NearMiss>(SCOPES.NEAR_MISSES);
      const filtered = all
        .sort((a, b) => a.delta - b.delta)
        .slice(0, input.limit ?? 20);
      return { near_misses: filtered, total: filtered.length };
    }
  );
}

type SearchStrategy = {
  mode: string;
  explore_ratio: number;
  temperature: number;
  updated_at: string;
  reason: string;
};

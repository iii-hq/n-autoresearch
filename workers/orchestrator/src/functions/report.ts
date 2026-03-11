import { StateKV } from "../state/kv.js";
import {
  SCOPES,
  type Experiment,
  type ExperimentTag,
  type BestResult,
  type GpuWorker,
  type SearchStrategy,
} from "../state/schema.js";

type SDK = {
  registerFunction(
    opts: { id: string; description: string },
    handler: (input: any) => Promise<any>
  ): void;
};

export function registerReportFunctions(sdk: SDK, kv: StateKV) {
  sdk.registerFunction(
    {
      id: "report::summary",
      description:
        "Generate a summary report for a tag. Includes best result, experiment stats, trends.",
    },
    async (input: { tag: string }) => {
      const tag = await kv.get<ExperimentTag>(SCOPES.TAGS, input.tag);
      if (!tag) return { error: `Tag '${input.tag}' not found` };

      const best = await kv.get<BestResult>(SCOPES.BEST, input.tag);
      const strategy = await kv.get<SearchStrategy>(SCOPES.STRATEGY, input.tag);
      const workers = await kv.list<GpuWorker>(SCOPES.GPU_POOL);

      const all = await kv.list<Experiment>(SCOPES.EXPERIMENTS);
      const tagExps = all
        .filter((e) => e.tag === input.tag)
        .sort(
          (a, b) =>
            new Date(a.started_at).getTime() - new Date(b.started_at).getTime()
        );

      const statusCounts = { keep: 0, discard: 0, crash: 0, running: 0 };
      for (const e of tagExps) {
        statusCounts[e.status] += 1;
      }

      const categoryCounts: Record<string, number> = {};
      for (const e of tagExps) {
        categoryCounts[e.category] = (categoryCounts[e.category] ?? 0) + 1;
      }

      const kept = tagExps.filter((e) => e.status === "keep");
      const bpbHistory = kept.map((e) => ({
        id: e.id,
        val_bpb: e.val_bpb,
        description: e.description,
        category: e.category,
        at: e.finished_at,
      }));

      const totalTrainingMin =
        tagExps.reduce((sum, e) => sum + e.training_seconds, 0) / 60;

      return {
        tag: input.tag,
        branch: tag.branch,
        best: best
          ? {
              val_bpb: best.val_bpb,
              commit: best.commit_sha,
              experiment_id: best.experiment_id,
            }
          : null,
        stats: {
          total: tag.total_experiments,
          kept: statusCounts.keep,
          discarded: statusCounts.discard,
          crashed: statusCounts.crash,
          running: statusCounts.running,
          keep_rate:
            tag.total_experiments > 0
              ? statusCounts.keep / tag.total_experiments
              : 0,
        },
        categories: categoryCounts,
        bpb_progression: bpbHistory,
        total_training_minutes: Math.round(totalTrainingMin * 10) / 10,
        strategy: strategy?.mode ?? "unknown",
        gpu_pool: {
          total: workers.length,
          idle: workers.filter((w) => w.status === "idle").length,
          training: workers.filter((w) => w.status === "training").length,
        },
      };
    }
  );

  sdk.registerFunction(
    {
      id: "report::tsv",
      description:
        "Export experiment history as TSV (compatible with original autoresearch format).",
    },
    async (input: { tag: string }) => {
      const all = await kv.list<Experiment>(SCOPES.EXPERIMENTS);
      const tagExps = all
        .filter((e) => e.tag === input.tag && e.status !== "running")
        .sort(
          (a, b) =>
            new Date(a.started_at).getTime() - new Date(b.started_at).getTime()
        );

      const header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription";
      const rows = tagExps.map((e) => {
        const sha = e.commit_sha.slice(0, 7);
        const bpb = e.status === "crash" ? "0.000000" : e.val_bpb.toFixed(6);
        const mem =
          e.status === "crash"
            ? "0.0"
            : (e.peak_vram_mb / 1024).toFixed(1);
        return `${sha}\t${bpb}\t${mem}\t${e.status}\t${e.description}`;
      });

      return { tsv: [header, ...rows].join("\n"), count: rows.length };
    }
  );

  sdk.registerFunction(
    {
      id: "report::diff",
      description:
        "Compare two experiments. Shows what changed and the BPB delta.",
    },
    async (input: { experiment_a: string; experiment_b: string }) => {
      const a = await kv.get<Experiment>(SCOPES.EXPERIMENTS, input.experiment_a);
      const b = await kv.get<Experiment>(SCOPES.EXPERIMENTS, input.experiment_b);
      if (!a || !b) return { error: "One or both experiments not found" };

      return {
        a: {
          id: a.id,
          val_bpb: a.val_bpb,
          description: a.description,
          category: a.category,
          num_params_m: a.num_params_m,
          peak_vram_mb: a.peak_vram_mb,
        },
        b: {
          id: b.id,
          val_bpb: b.val_bpb,
          description: b.description,
          category: b.category,
          num_params_m: b.num_params_m,
          peak_vram_mb: b.peak_vram_mb,
        },
        delta_bpb: b.val_bpb - a.val_bpb,
        delta_params_m: b.num_params_m - a.num_params_m,
        delta_vram_mb: b.peak_vram_mb - a.peak_vram_mb,
      };
    }
  );

  sdk.registerFunction(
    {
      id: "report::tags",
      description: "List all experiment run tags.",
    },
    async () => {
      const tags = await kv.list<ExperimentTag>(SCOPES.TAGS);
      return {
        tags: tags.sort(
          (a, b) =>
            new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
        ),
      };
    }
  );
}

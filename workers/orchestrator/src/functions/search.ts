import { StateKV } from "../state/kv.js";
import {
  SCOPES,
  type Experiment,
  type ExperimentCategory,
  type NearMiss,
  type SearchStrategy,
} from "../state/schema.js";

type SDK = {
  registerFunction(
    opts: { id: string; description: string },
    handler: (input: any) => Promise<any>
  ): void;
};

export function registerSearchFunctions(sdk: SDK, kv: StateKV) {
  sdk.registerFunction(
    {
      id: "search::strategy",
      description:
        "Get current search strategy for a tag. Returns mode, temperature, and reasoning.",
    },
    async (input: { tag: string }) => {
      const strategy = await kv.get<SearchStrategy>(
        SCOPES.STRATEGY,
        input.tag
      );
      return strategy ?? { mode: "explore", explore_ratio: 0.7, temperature: 1.0 };
    }
  );

  sdk.registerFunction(
    {
      id: "search::set_strategy",
      description: "Override search strategy for a tag.",
    },
    async (input: {
      tag: string;
      mode: SearchStrategy["mode"];
      reason: string;
      explore_ratio?: number;
      temperature?: number;
    }) => {
      const strategy: SearchStrategy = {
        mode: input.mode,
        explore_ratio: input.explore_ratio ?? 0.7,
        temperature: input.temperature ?? 1.0,
        updated_at: new Date().toISOString(),
        reason: input.reason,
      };
      await kv.set(SCOPES.STRATEGY, input.tag, strategy);
      return strategy;
    }
  );

  sdk.registerFunction(
    {
      id: "search::adapt",
      description:
        "Auto-adapt search strategy based on experiment history. Call after each experiment.",
    },
    async (input: { tag: string }) => {
      const all = await kv.list<Experiment>(SCOPES.EXPERIMENTS);
      const tagExps = all.filter((e) => e.tag === input.tag && e.status !== "running");

      if (tagExps.length < 5) {
        return { mode: "explore", reason: "too few experiments to adapt" };
      }

      const recent = tagExps.slice(-10);
      const recentKeeps = recent.filter((e) => e.status === "keep").length;
      const recentCrashes = recent.filter((e) => e.status === "crash").length;
      const keepRate = recentKeeps / recent.length;
      const crashRate = recentCrashes / recent.length;

      const nearMisses = await kv.list<NearMiss>(SCOPES.NEAR_MISSES);

      let mode: SearchStrategy["mode"];
      let reason: string;
      let temperature: number;

      if (crashRate > 0.5) {
        mode = "exploit";
        temperature = 0.3;
        reason = `high crash rate (${(crashRate * 100).toFixed(0)}%), switching to conservative tweaks`;
      } else if (keepRate === 0 && tagExps.length > 20) {
        if (nearMisses.length >= 2) {
          mode = "combine";
          temperature = 0.5;
          reason = `plateau with ${nearMisses.length} near-misses, trying combinations`;
        } else {
          mode = "ablation";
          temperature = 0.3;
          reason = "long plateau, switching to ablation to identify essential components";
        }
      } else if (keepRate > 0.3) {
        mode = "exploit";
        temperature = 0.5;
        reason = `good keep rate (${(keepRate * 100).toFixed(0)}%), exploiting current direction`;
      } else {
        mode = "explore";
        temperature = 0.8;
        reason = "default exploration";
      }

      const strategy: SearchStrategy = {
        mode,
        explore_ratio: mode === "explore" ? 0.8 : 0.3,
        temperature,
        updated_at: new Date().toISOString(),
        reason,
      };
      await kv.set(SCOPES.STRATEGY, input.tag, strategy);

      return strategy;
    }
  );

  sdk.registerFunction(
    {
      id: "search::suggest_direction",
      description:
        "Analyze experiment history and suggest what to try next. Returns structured hints for the external agent.",
    },
    async (input: { tag: string }) => {
      const all = await kv.list<Experiment>(SCOPES.EXPERIMENTS);
      const tagExps = all
        .filter((e) => e.tag === input.tag && e.status !== "running")
        .sort(
          (a, b) =>
            new Date(a.started_at).getTime() - new Date(b.started_at).getTime()
        );

      const strategy = await kv.get<SearchStrategy>(SCOPES.STRATEGY, input.tag);
      const nearMisses = await kv.list<NearMiss>(SCOPES.NEAR_MISSES);

      const categoryCounts: Record<string, { total: number; kept: number }> = {};
      for (const exp of tagExps) {
        if (!categoryCounts[exp.category]) {
          categoryCounts[exp.category] = { total: 0, kept: 0 };
        }
        categoryCounts[exp.category].total += 1;
        if (exp.status === "keep") {
          categoryCounts[exp.category].kept += 1;
        }
      }

      const underexplored: ExperimentCategory[] = [];
      const allCategories: ExperimentCategory[] = [
        "architecture",
        "optimizer",
        "hyperparams",
        "activation",
        "attention",
        "embedding",
        "normalization",
        "regularization",
        "scheduling",
        "initialization",
        "simplification",
      ];
      for (const cat of allCategories) {
        if (!categoryCounts[cat] || categoryCounts[cat].total < 3) {
          underexplored.push(cat);
        }
      }

      const highYield = Object.entries(categoryCounts)
        .filter(([, v]) => v.total >= 3 && v.kept / v.total > 0.3)
        .map(([k]) => k as ExperimentCategory);

      const kept = tagExps.filter((e) => e.status === "keep");
      const bpbTrend = kept.slice(-5).map((e) => e.val_bpb);

      return {
        strategy: strategy?.mode ?? "explore",
        total_experiments: tagExps.length,
        category_stats: categoryCounts,
        underexplored_categories: underexplored,
        high_yield_categories: highYield,
        near_misses_available: nearMisses.length,
        near_miss_categories: [...new Set(nearMisses.map((n) => n.category))],
        recent_bpb_trend: bpbTrend,
        suggestions: buildSuggestions(
          strategy?.mode ?? "explore",
          underexplored,
          highYield,
          nearMisses,
          bpbTrend
        ),
      };
    }
  );
}

function buildSuggestions(
  mode: string,
  underexplored: ExperimentCategory[],
  highYield: ExperimentCategory[],
  nearMisses: NearMiss[],
  trend: number[]
): string[] {
  const suggestions: string[] = [];

  switch (mode) {
    case "explore":
      if (underexplored.length > 0) {
        suggestions.push(
          `Try changes in underexplored categories: ${underexplored.slice(0, 3).join(", ")}`
        );
      }
      suggestions.push("Try a radical architectural change");
      break;

    case "exploit":
      if (highYield.length > 0) {
        suggestions.push(
          `Double down on high-yield categories: ${highYield.join(", ")}`
        );
      }
      suggestions.push("Make small incremental tweaks to the current best config");
      break;

    case "combine":
      if (nearMisses.length >= 2) {
        const pair = nearMisses.slice(0, 2);
        suggestions.push(
          `Combine near-misses: "${pair[0].hypothesis}" + "${pair[1].hypothesis}"`
        );
      }
      break;

    case "ablation":
      suggestions.push(
        "Remove one component at a time to identify what actually matters"
      );
      suggestions.push(
        "Try simplifying: fewer layers, simpler activations, remove value embeddings"
      );
      break;
  }

  if (trend.length >= 3) {
    const improving = trend[trend.length - 1] < trend[0];
    if (!improving) {
      suggestions.push("BPB trend is flat/worsening. Consider a strategy change.");
    }
  }

  return suggestions;
}

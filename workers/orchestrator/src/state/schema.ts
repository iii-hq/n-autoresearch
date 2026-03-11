export const SCOPES = {
  EXPERIMENTS: "experiments",
  LINEAGE: "lineage",
  BEST: "best",
  NEAR_MISSES: "near_misses",
  GPU_POOL: "gpu_pool",
  STRATEGY: "strategy",
  TAGS: "tags",
  CRASHES: "crashes",
} as const;

export interface Experiment {
  id: string;
  tag: string;
  parent_id: string | null;
  commit_sha: string;
  description: string;
  hypothesis: string;
  category: ExperimentCategory;
  val_bpb: number;
  peak_vram_mb: number;
  training_seconds: number;
  total_tokens_m: number;
  mfu_percent: number;
  num_steps: number;
  num_params_m: number;
  depth: number;
  status: "running" | "keep" | "discard" | "crash";
  gpu_id: string;
  started_at: string;
  finished_at: string | null;
  diff_summary: string;
  error: string | null;
}

export type ExperimentCategory =
  | "architecture"
  | "optimizer"
  | "hyperparams"
  | "activation"
  | "attention"
  | "embedding"
  | "normalization"
  | "regularization"
  | "scheduling"
  | "initialization"
  | "simplification"
  | "combination"
  | "ablation"
  | "other";

export interface GpuWorker {
  id: string;
  name: string;
  gpu_index: number;
  gpu_name: string;
  vram_mb: number;
  status: "idle" | "training" | "error" | "offline";
  current_experiment_id: string | null;
  registered_at: string;
  last_heartbeat: string;
}

export interface BestResult {
  experiment_id: string;
  val_bpb: number;
  commit_sha: string;
  updated_at: string;
}

export interface SearchStrategy {
  mode: "explore" | "exploit" | "combine" | "ablation" | "random";
  explore_ratio: number;
  temperature: number;
  updated_at: string;
  reason: string;
}

export interface NearMiss {
  experiment_id: string;
  val_bpb: number;
  delta: number;
  hypothesis: string;
  category: ExperimentCategory;
  diff_summary: string;
}

export interface ExperimentTag {
  name: string;
  branch: string;
  created_at: string;
  best_val_bpb: number;
  total_experiments: number;
  kept_experiments: number;
}

export function experimentId(): string {
  return `exp-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

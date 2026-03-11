export const CONFIG = {
  WS_URL: process.env.III_WS_URL ?? "ws://localhost:49134",
  REST_PORT: parseInt(process.env.III_REST_PORT ?? "3111", 10),
  WORKER_NAME: "n-autoresearch-orchestrator",
  VERSION: "0.1.0",

  TRAIN_SCRIPT: process.env.TRAIN_SCRIPT ?? "train.py",
  PREPARE_SCRIPT: process.env.PREPARE_SCRIPT ?? "prepare.py",
  REPO_DIR: process.env.REPO_DIR ?? process.cwd(),
  TIME_BUDGET: parseInt(process.env.TIME_BUDGET ?? "300", 10),
  KILL_TIMEOUT: parseInt(process.env.KILL_TIMEOUT ?? "600", 10),

  MAX_CONSECUTIVE_CRASHES: parseInt(process.env.MAX_CONSECUTIVE_CRASHES ?? "3", 10),
  NEAR_MISS_THRESHOLD: parseFloat(process.env.NEAR_MISS_THRESHOLD ?? "0.002"),

  REPORT_INTERVAL_CRON: process.env.REPORT_CRON ?? "0 */2 * * *",
  ANALYSIS_INTERVAL_CRON: process.env.ANALYSIS_CRON ?? "*/30 * * * *",
} as const;

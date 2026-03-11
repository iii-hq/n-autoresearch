import { StateKV } from "../state/kv.js";
import { SCOPES, type GpuWorker } from "../state/schema.js";

type SDK = {
  registerFunction(
    opts: { id: string; description: string },
    handler: (input: any) => Promise<any>
  ): void;
};

export function registerPoolFunctions(sdk: SDK, kv: StateKV) {
  sdk.registerFunction(
    {
      id: "pool::register_gpu",
      description:
        "Register a GPU worker in the pool. Called by gpu workers on startup.",
    },
    async (input: {
      gpu_id: string;
      gpu_name: string;
      gpu_index: number;
      vram_mb: number;
    }) => {
      const worker: GpuWorker = {
        id: input.gpu_id,
        name: `gpu-${input.gpu_index}`,
        gpu_index: input.gpu_index,
        gpu_name: input.gpu_name,
        vram_mb: input.vram_mb,
        status: "idle",
        current_experiment_id: null,
        registered_at: new Date().toISOString(),
        last_heartbeat: new Date().toISOString(),
      };
      await kv.set(SCOPES.GPU_POOL, input.gpu_id, worker);
      return { registered: true, worker };
    }
  );

  sdk.registerFunction(
    {
      id: "pool::heartbeat",
      description: "GPU worker heartbeat. Updates last_heartbeat timestamp.",
    },
    async (input: { gpu_id: string }) => {
      const worker = await kv.get<GpuWorker>(SCOPES.GPU_POOL, input.gpu_id);
      if (!worker) return { error: "GPU worker not found" };
      worker.last_heartbeat = new Date().toISOString();
      await kv.set(SCOPES.GPU_POOL, input.gpu_id, worker);
      return { ok: true };
    }
  );

  sdk.registerFunction(
    {
      id: "pool::list",
      description: "List all GPU workers with their current status.",
    },
    async () => {
      const workers = await kv.list<GpuWorker>(SCOPES.GPU_POOL);
      const now = Date.now();
      const STALE_MS = 60_000;
      for (const w of workers) {
        const elapsed = now - new Date(w.last_heartbeat).getTime();
        if (elapsed > STALE_MS && w.status !== "offline") {
          w.status = "offline";
          await kv.set(SCOPES.GPU_POOL, w.id, w);
        }
      }
      return {
        workers,
        total: workers.length,
        idle: workers.filter((w) => w.status === "idle").length,
        training: workers.filter((w) => w.status === "training").length,
        offline: workers.filter((w) => w.status === "offline").length,
      };
    }
  );

  sdk.registerFunction(
    {
      id: "pool::acquire",
      description:
        "Acquire an idle GPU for an experiment. Returns gpu_id or null if none available.",
    },
    async (input: { experiment_id: string }) => {
      const workers = await kv.list<GpuWorker>(SCOPES.GPU_POOL);
      const idle = workers.find((w) => w.status === "idle");
      if (!idle) {
        return { acquired: false, gpu_id: null, reason: "no idle GPUs" };
      }
      idle.status = "training";
      idle.current_experiment_id = input.experiment_id;
      await kv.set(SCOPES.GPU_POOL, idle.id, idle);
      return { acquired: true, gpu_id: idle.id, gpu_index: idle.gpu_index };
    }
  );

  sdk.registerFunction(
    {
      id: "pool::release",
      description: "Release a GPU back to idle after experiment finishes.",
    },
    async (input: { gpu_id: string }) => {
      const worker = await kv.get<GpuWorker>(SCOPES.GPU_POOL, input.gpu_id);
      if (!worker) return { error: "GPU worker not found" };
      worker.status = "idle";
      worker.current_experiment_id = null;
      await kv.set(SCOPES.GPU_POOL, input.gpu_id, worker);
      return { released: true };
    }
  );

  sdk.registerFunction(
    {
      id: "pool::deregister",
      description: "Remove a GPU worker from the pool.",
    },
    async (input: { gpu_id: string }) => {
      await kv.delete(SCOPES.GPU_POOL, input.gpu_id);
      return { deregistered: true };
    }
  );
}

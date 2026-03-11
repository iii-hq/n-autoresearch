import { init } from "iii-sdk";
import { CONFIG } from "./config.js";
import { StateKV } from "./state/kv.js";
import { registerExperimentFunctions } from "./functions/experiment.js";
import { registerSearchFunctions } from "./functions/search.js";
import { registerPoolFunctions } from "./functions/pool.js";
import { registerReportFunctions } from "./functions/report.js";
import { registerApiTriggers } from "./triggers/api.js";
import { registerEventTriggers } from "./triggers/events.js";
import { registerCronTriggers } from "./triggers/cron.js";

const sdk = init(CONFIG.WS_URL, {
  workerName: CONFIG.WORKER_NAME,
  otel: {
    enabled: true,
    serviceName: "n-autoresearch",
    serviceVersion: CONFIG.VERSION,
    metricsEnabled: true,
  },
});

const kv = new StateKV(sdk);

registerExperimentFunctions(sdk, kv);
registerSearchFunctions(sdk, kv);
registerPoolFunctions(sdk, kv);
registerReportFunctions(sdk, kv);

registerApiTriggers(sdk);
registerEventTriggers(sdk);
registerCronTriggers(sdk);

const shutdown = async () => {
  console.log("Shutting down orchestrator...");
  await sdk.shutdown();
  process.exit(0);
};

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

console.log(`n-autoresearch orchestrator v${CONFIG.VERSION}`);
console.log(`Connected to iii-engine at ${CONFIG.WS_URL}`);
console.log(`REST API at http://localhost:${CONFIG.REST_PORT}`);
console.log("Functions: 21 | Triggers: 23 | Ready.");

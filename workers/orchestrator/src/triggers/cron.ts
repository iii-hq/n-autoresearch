import { CONFIG } from "../config.js";

type SDK = {
  registerTrigger(config: {
    trigger_type: string;
    function_id: string;
    config: Record<string, unknown>;
  }): void;
};

export function registerCronTriggers(sdk: SDK) {
  sdk.registerTrigger({
    trigger_type: "cron",
    function_id: "pool::list",
    config: { expression: "*/30 * * * * *" },
  });
}

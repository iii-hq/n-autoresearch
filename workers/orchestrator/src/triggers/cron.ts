import { CONFIG } from "../config.js";

type SDK = {
  registerTrigger(config: {
    trigger_type: string;
    function_path: string;
    config: Record<string, unknown>;
  }): void;
};

export function registerCronTriggers(sdk: SDK) {
  sdk.registerTrigger({
    trigger_type: "cron",
    function_path: "pool::list",
    config: { expression: "*/30 * * * * *" },
  });
}

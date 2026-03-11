type SDK = {
  registerTrigger(config: {
    trigger_type: string;
    function_id: string;
    config: Record<string, unknown>;
  }): void;
};

export function registerEventTriggers(sdk: SDK) {
  sdk.registerTrigger({
    trigger_type: "queue",
    function_id: "search::adapt",
    config: { topic: "experiment.completed" },
  });

  sdk.registerTrigger({
    trigger_type: "queue",
    function_id: "search::adapt",
    config: { topic: "experiment.crashed" },
  });
}

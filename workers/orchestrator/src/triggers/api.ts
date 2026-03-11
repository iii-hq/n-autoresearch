type SDK = {
  registerTrigger(config: {
    trigger_type: string;
    function_id: string;
    config: Record<string, unknown>;
  }): void;
};

export function registerApiTriggers(sdk: SDK) {
  const httpTriggers: Array<{
    path: string;
    method: string;
    fn: string;
  }> = [
    { path: "/api/experiment/setup", method: "POST", fn: "experiment::setup" },
    { path: "/api/experiment/register", method: "POST", fn: "experiment::register" },
    { path: "/api/experiment/complete", method: "POST", fn: "experiment::complete" },
    { path: "/api/experiment/crash", method: "POST", fn: "experiment::crash" },
    { path: "/api/experiment/history", method: "POST", fn: "experiment::history" },
    { path: "/api/experiment/best", method: "POST", fn: "experiment::best" },
    { path: "/api/experiment/near-misses", method: "POST", fn: "experiment::near_misses" },

    { path: "/api/search/strategy", method: "POST", fn: "search::strategy" },
    { path: "/api/search/set-strategy", method: "POST", fn: "search::set_strategy" },
    { path: "/api/search/adapt", method: "POST", fn: "search::adapt" },
    { path: "/api/search/suggest", method: "POST", fn: "search::suggest_direction" },

    { path: "/api/pool/register", method: "POST", fn: "pool::register_gpu" },
    { path: "/api/pool/heartbeat", method: "POST", fn: "pool::heartbeat" },
    { path: "/api/pool/list", method: "GET", fn: "pool::list" },
    { path: "/api/pool/acquire", method: "POST", fn: "pool::acquire" },
    { path: "/api/pool/release", method: "POST", fn: "pool::release" },

    { path: "/api/report/summary", method: "POST", fn: "report::summary" },
    { path: "/api/report/tsv", method: "POST", fn: "report::tsv" },
    { path: "/api/report/diff", method: "POST", fn: "report::diff" },
    { path: "/api/report/tags", method: "GET", fn: "report::tags" },
  ];

  for (const t of httpTriggers) {
    sdk.registerTrigger({
      trigger_type: "http",
      function_id: t.fn,
      config: { api_path: t.path, http_method: t.method },
    });
  }
}

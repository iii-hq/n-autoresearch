pub struct GpuConfig {
    pub ws_url: String,
    pub gpu_index: u32,
    pub repo_dir: String,
    pub time_budget: u64,
    pub kill_timeout: u64,
}

impl GpuConfig {
    pub fn from_env() -> Self {
        Self {
            ws_url: std::env::var("III_WS_URL")
                .unwrap_or_else(|_| "ws://localhost:49134".to_string()),
            gpu_index: std::env::var("GPU_INDEX")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            repo_dir: std::env::var("REPO_DIR")
                .unwrap_or_else(|_| ".".to_string()),
            time_budget: std::env::var("TIME_BUDGET")
                .unwrap_or_else(|_| "300".to_string())
                .parse()
                .unwrap_or(300),
            kill_timeout: std::env::var("KILL_TIMEOUT")
                .unwrap_or_else(|_| "600".to_string())
                .parse()
                .unwrap_or(600),
        }
    }
}

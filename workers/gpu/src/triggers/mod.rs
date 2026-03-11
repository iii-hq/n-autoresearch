use iii_sdk::III;
use serde_json::json;
use std::sync::Arc;

pub fn register_all(iii: Arc<III>, gpu_id: &str) {
    iii.register_trigger(
        "http",
        "gpu::train",
        json!({
            "api_path": format!("/api/gpu/{}/train", gpu_id),
            "http_method": "POST",
        }),
    );

    iii.register_trigger(
        "http",
        "gpu::health",
        json!({
            "api_path": format!("/api/gpu/{}/health", gpu_id),
            "http_method": "GET",
        }),
    );

    iii.register_trigger(
        "cron",
        "gpu::health",
        json!({ "expression": "*/30 * * * * *" }),
    );
}

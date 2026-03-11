use iii_sdk::{III, WorkerMetadata};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::process::Command;
use tokio::signal;
use uuid::Uuid;

mod config;
mod functions;
mod state;
mod triggers;

use config::GpuConfig;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("n_autoresearch_gpu=info".parse().unwrap()),
        )
        .init();

    let config = GpuConfig::from_env();
    tracing::info!(
        gpu_index = config.gpu_index,
        repo_dir = %config.repo_dir,
        "Starting n-autoresearch GPU worker"
    );

    let iii = Arc::new(III::with_metadata(
        &config.ws_url,
        WorkerMetadata {
            name: format!("gpu-worker-{}", config.gpu_index),
            ..Default::default()
        },
    ));
    iii.connect().await.expect("Failed to connect to iii-engine");

    let gpu_id = format!("gpu-{}", config.gpu_index);

    functions::register_all(iii.clone(), &config, &gpu_id);
    triggers::register_all(iii.clone(), &gpu_id);

    let gpu_info = detect_gpu(&config).await;
    let reg_result = iii
        .trigger(
            "pool::register_gpu",
            json!({
                "gpu_id": gpu_id,
                "gpu_name": gpu_info.name,
                "gpu_index": config.gpu_index,
                "vram_mb": gpu_info.vram_mb,
            }),
        )
        .await;
    tracing::info!(?reg_result, "Registered with GPU pool");

    tracing::info!(gpu_id = %gpu_id, "GPU worker ready. Waiting for experiments...");

    signal::ctrl_c().await.expect("Failed to listen for ctrl-c");
    tracing::info!("Shutting down GPU worker...");

    let _ = iii
        .trigger("pool::deregister", json!({ "gpu_id": gpu_id }))
        .await;
    iii.shutdown_async().await;
}

struct GpuInfo {
    name: String,
    vram_mb: u64,
}

async fn detect_gpu(config: &GpuConfig) -> GpuInfo {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
            &format!("--id={}", config.gpu_index),
        ])
        .output()
        .await;

    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = stdout.trim().split(", ").collect();
            if parts.len() == 2 {
                return GpuInfo {
                    name: parts[0].to_string(),
                    vram_mb: parts[1].parse().unwrap_or(0),
                };
            }
        }
        _ => {}
    }

    GpuInfo {
        name: "Unknown GPU".to_string(),
        vram_mb: 0,
    }
}

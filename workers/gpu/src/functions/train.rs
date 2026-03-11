use iii_sdk::III;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

use crate::config::GpuConfig;

pub fn register(iii: Arc<III>, config: &GpuConfig, gpu_id: &str) {
    let repo_dir = config.repo_dir.clone();
    let kill_timeout = config.kill_timeout;
    let gpu_index = config.gpu_index;
    let gpu_id_owned = gpu_id.to_string();

    iii.register_function_with_description(
        "gpu::train",
        "Execute a training run on this GPU. Runs train.py with fixed time budget, captures metrics.",
        move |input: Value| {
            let repo_dir = repo_dir.clone();
            let gpu_id = gpu_id_owned.clone();

            async move {
                let experiment_id = input
                    .get("experiment_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                tracing::info!(
                    experiment_id = experiment_id,
                    gpu = gpu_index,
                    "Starting training run"
                );

                let log_path = format!("{}/run-{}.log", repo_dir, experiment_id);

                let result = timeout(
                    Duration::from_secs(kill_timeout),
                    run_training(&repo_dir, gpu_index, &log_path),
                )
                .await;

                match result {
                    Ok(Ok(metrics)) => {
                        tracing::info!(
                            experiment_id = experiment_id,
                            val_bpb = metrics.val_bpb,
                            "Training complete"
                        );
                        Ok(json!({
                            "status": "success",
                            "experiment_id": experiment_id,
                            "gpu_id": gpu_id,
                            "val_bpb": metrics.val_bpb,
                            "peak_vram_mb": metrics.peak_vram_mb,
                            "training_seconds": metrics.training_seconds,
                            "total_tokens_m": metrics.total_tokens_m,
                            "mfu_percent": metrics.mfu_percent,
                            "num_steps": metrics.num_steps,
                            "num_params_m": metrics.num_params_m,
                            "depth": metrics.depth,
                            "log_path": log_path,
                        }))
                    }
                    Ok(Err(e)) => {
                        tracing::error!(
                            experiment_id = experiment_id,
                            error = %e,
                            "Training crashed"
                        );
                        Ok(json!({
                            "status": "crash",
                            "experiment_id": experiment_id,
                            "gpu_id": gpu_id,
                            "error": e,
                            "log_path": log_path,
                        }))
                    }
                    Err(_) => {
                        tracing::error!(
                            experiment_id = experiment_id,
                            timeout_s = kill_timeout,
                            "Training timed out"
                        );
                        Ok(json!({
                            "status": "timeout",
                            "experiment_id": experiment_id,
                            "gpu_id": gpu_id,
                            "error": format!("Exceeded kill timeout of {}s", kill_timeout),
                            "log_path": log_path,
                        }))
                    }
                }
            }
        },
    );

    let iii_health = iii.clone();
    let gpu_id_health = gpu_id.to_string();
    let gpu_idx = config.gpu_index;
    iii_health.register_function_with_description(
        "gpu::health",
        "Check GPU health: temperature, memory usage, utilization.",
        move |_input: Value| {
            let gpu_id = gpu_id_health.clone();
            async move {
                let output = Command::new("nvidia-smi")
                    .args([
                        "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                        &format!("--id={}", gpu_idx),
                    ])
                    .output()
                    .await;

                match output {
                    Ok(out) if out.status.success() => {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        let parts: Vec<&str> = stdout.trim().split(", ").collect();
                        if parts.len() == 4 {
                            Ok(json!({
                                "gpu_id": gpu_id,
                                "temperature_c": parts[0].parse::<u32>().unwrap_or(0),
                                "memory_used_mb": parts[1].parse::<u64>().unwrap_or(0),
                                "memory_total_mb": parts[2].parse::<u64>().unwrap_or(0),
                                "utilization_percent": parts[3].parse::<u32>().unwrap_or(0),
                            }))
                        } else {
                            Ok(json!({ "gpu_id": gpu_id, "error": "failed to parse nvidia-smi" }))
                        }
                    }
                    _ => Ok(json!({ "gpu_id": gpu_id, "error": "nvidia-smi not available" })),
                }
            }
        },
    );
}

struct TrainingMetrics {
    val_bpb: f64,
    peak_vram_mb: f64,
    training_seconds: f64,
    total_tokens_m: f64,
    mfu_percent: f64,
    num_steps: u64,
    num_params_m: f64,
    depth: u32,
}

async fn run_training(repo_dir: &str, gpu_index: u32, log_path: &str) -> Result<TrainingMetrics, String> {
    let output = Command::new("uv")
        .args(["run", "train.py"])
        .current_dir(repo_dir)
        .env("CUDA_VISIBLE_DEVICES", gpu_index.to_string())
        .output()
        .await
        .map_err(|e| format!("Failed to spawn: {}", e))?;

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    tokio::fs::write(log_path, &combined)
        .await
        .map_err(|e| format!("Failed to write log: {}", e))?;

    if !output.status.success() {
        let tail: String = combined.lines().rev().take(50).collect::<Vec<_>>().join("\n");
        return Err(format!("Train exited with {}: {}", output.status, tail));
    }

    parse_metrics(&combined)
}

fn parse_metrics(output: &str) -> Result<TrainingMetrics, String> {
    let get = |key: &str| -> Result<f64, String> {
        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with(&format!("{}:", key)) {
                let val_str = trimmed
                    .strip_prefix(&format!("{}:", key))
                    .unwrap()
                    .trim();
                return val_str
                    .parse::<f64>()
                    .map_err(|e| format!("Failed to parse {}: {}", key, e));
            }
        }
        Err(format!("Key '{}' not found in output", key))
    };

    Ok(TrainingMetrics {
        val_bpb: get("val_bpb")?,
        peak_vram_mb: get("peak_vram_mb").unwrap_or(0.0),
        training_seconds: get("training_seconds").unwrap_or(0.0),
        total_tokens_m: get("total_tokens_M").unwrap_or(0.0),
        mfu_percent: get("mfu_percent").unwrap_or(0.0),
        num_steps: get("num_steps").unwrap_or(0.0) as u64,
        num_params_m: get("num_params_M").unwrap_or(0.0),
        depth: get("depth").unwrap_or(0.0) as u32,
    })
}

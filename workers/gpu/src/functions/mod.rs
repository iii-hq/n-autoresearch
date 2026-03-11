mod train;

use iii_sdk::III;
use std::sync::Arc;

use crate::config::GpuConfig;

pub fn register_all(iii: Arc<III>, config: &GpuConfig, gpu_id: &str) {
    train::register(iii, config, gpu_id);
}

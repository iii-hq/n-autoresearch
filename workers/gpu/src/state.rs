use iii_sdk::{III, IIIError};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::json;
use std::sync::Arc;

pub struct StateKV {
    iii: Arc<III>,
}

impl StateKV {
    pub fn new(iii: Arc<III>) -> Self {
        Self { iii }
    }

    pub async fn get<T: DeserializeOwned>(&self, scope: &str, key: &str) -> Option<T> {
        let result = self
            .iii
            .trigger("state::get", json!({ "scope": scope, "key": key }))
            .await
            .ok()?;
        let value = result.get("value")?;
        serde_json::from_value(value.clone()).ok()
    }

    pub async fn set<T: Serialize>(
        &self,
        scope: &str,
        key: &str,
        data: &T,
    ) -> Result<(), IIIError> {
        self.iii
            .trigger(
                "state::set",
                json!({ "scope": scope, "key": key, "value": data }),
            )
            .await?;
        Ok(())
    }
}

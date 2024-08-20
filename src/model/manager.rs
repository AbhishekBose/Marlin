use crate::model::traits::{Backend};
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Error;
use tokio::sync::RwLock;
use crate::model::backends::onnx::bert_onnx::BertONNX;

use super::traits::model_trait::ModelTrait;

pub struct ModelManager {
    models: RwLock<HashMap<u32, Arc<RwLock<dyn ModelTrait + Send + Sync>>>>,
    next_id: RwLock<u32>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self { 
            models: RwLock::new(HashMap::new()),
            next_id: RwLock::new(1),
        }
    }

    pub async fn load_model(&self, model_path: String, model_type: Backend) -> Result<u32, Error> {
        let model: Arc<RwLock<dyn ModelTrait + Send + Sync>> = match model_type {
            Backend::ONNX => Arc::new(RwLock::new(BertONNX::new())),
            _ => return Err(Error::msg("Not implemented")),
        };

        {
            let mut model_guard = model.write().await;
            model_guard.load_model(&model_path).await.map_err(|e| Error::msg(e.to_string()))?;
        }

        let mut next_id = self.next_id.write().await;
        let model_id = *next_id;
        *next_id += 1;

        let mut models = self.models.write().await;
        models.insert(model_id, model);

        Ok(model_id)
    }

    pub async fn predict(&self, model_id: u32, input: &str) -> Result<String, Error> {
        let models = self.models.read().await;
        match models.get(&model_id) {
            Some(model) => {
                let model_guard = model.read().await; // Lock the RwLock for reading
                model_guard.predict(input).await.map_err(|e| Error::msg(e.to_string()))
            }
            None => Err(Error::msg("Model not found")),
        }
    }
}
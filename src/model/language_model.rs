use anyhow::{Error as E, Result};
use candle_core::Device;
use candle_core::Tensor;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE, HiddenAct};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct LanguageModel {
    pub model: BertModel,
    pub tokenizer: Tokenizer,
    pub device: Device
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

fn get_mask(size: usize, device: &Device) -> Tensor {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device).unwrap()
}
impl LanguageModel {
    pub fn build_model_and_tokenizer() -> Result<LanguageModel> {
       let device = device(true)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let default_model_id = None;
        let revision = None;
        let (model_id, revision) = match (default_model_id, revision) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };
        let use_pth = true;
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // tokenizer = Tokenizer::from(tokenizer
        //     .with_padding(None)
        //     .with_truncation(None)
        //     .map_err(E::msg)?);

        let vb = if use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let approximate_gelu = false;
        if approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModel::load(vb, &config)?;

        // Ok((model, tokenizer))
        Ok(LanguageModel{
            model,
            tokenizer,
            device
        })
    }

    pub fn encode(&self, prompt: &str) -> Result<(Tensor,Tensor)> {

        let tokens = self.tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..],&self.device )?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let embeddings =self.model.forward(&token_ids, &token_type_ids)?;
        Ok((embeddings,token_ids))
    }
}
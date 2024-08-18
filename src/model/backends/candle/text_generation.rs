use anyhow::{Error as E, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Tensor;
use candle_core::{DType, Device, Error};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use candle_transformers::models::quantized_mistral::{
    Config, Model as QuantizedMistral, VarBuilder as QuantizedMistralVarBuilder,
};
use hf_hub::api::sync::ApiRepo;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::ModelWeights;
use serde::de::Deserializer;
use serde::Deserialize;
use std::collections::HashSet;
use tokenizers::Tokenizer;
use candle_core::quantized::{ggml_file,gguf_file};

pub struct GenerationModel {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
}

impl GenerationModel {
    pub fn new() -> Result<GenerationModel> {
        //let base_repo_id = ("TheBloke/CodeLlama-7B-GGUF", "codellama-7b.Q4_0.gguf");
        let base_repo_id = (
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "mistral-7b-instruct-v0.2.Q4_K_S.gguf",
        );
        //let base_repo_id = ("MaziyarPanahi/gemma-2b-it-GGUF", "gemma-2b-it.Q4_K_M.gguf");
        //let tokenizer_repo = "hf-internal-testing/llama-tokenizer";
        let tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2";
        //let tokenizer_repo = "google/gemma-2b-it";

        let repo = Repo::new(base_repo_id.0.to_string(), RepoType::Model);
        let api = Api::new()?;
        let api = api.repo(repo);
        let tokenizer_filename = get_tokenizer(&api);
        let model_path = api.get(base_repo_id.1).expect("API error");

        //let device = Device::Cpu;
        let device = Device::Cpu;
        // let mut tokenizer = Tokenizer::from_bytes(&tokenizer).map_anyhow_err()?;
        let mut tokenizer = tokenizer_filename.map_err(E::msg)?;
        let mut file = std::fs::File::open(&model_path)?;
        let model_content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(model_content, &mut file, &device)?;

        Ok(GenerationModel {
            model,
            tokenizer,
            device,
        })
    }
}

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

fn deserialize_weight_map<'de, D>(deserializer: D) -> Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}

/// Loads the safetensors files for a model from the hub based on a json index file.
// pub fn hub_load_safetensors(
//     repo: &hf_hub::api::sync::ApiRepo,
//     json_file: &str,
// ) -> Result<Vec<std::path::PathBuf>> {
//     let json_file = repo.get(json_file).map_err(Error::wrap)?;
//     let json_file = std::fs::File::open(json_file)?;
//     let json: serde_json::Value =
//         serde_json::from_reader(&json_file).map_err(Error::wrap)?;
//     let weight_map = match json.get("weight_map") {
//         None => candle_core::bail!("no weight map in {json_file:?}").into(),
//         Some(serde_json::Value::Object(map)) => map,
//         Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map").into(),
//     };
//     let mut safetensors_files: HashSet<String> = std::collections::HashSet::new();
//     for value in weight_map.values() {
//         if let Some(file) = value.as_str() {
//             safetensors_files.insert(file.to_string());
//         }
//     }
//     let safetensors_files = safetensors_files
//         .iter()
//         .map(|v| repo.get(v).map_err(Error::wrap))
//         .collect()?;
//     Ok(safetensors_files)
// }

fn get_tokenizer(repo: &ApiRepo) -> Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;
    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

pub struct TokenGenerator {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
    device: Device,
}

impl TokenGenerator {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
            device: Device::Cpu,
        }
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, anyhow::Error> {
        let decoded_str = self.tokenizer.decode(tokens, true).map_err(anyhow::Error::msg)?;
        Ok(decoded_str)
    }

    pub fn get_tokens(&self, prompt: String) -> Vec<u32> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();
        return tokens;
    }
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        // if there's nothing there, return an empty string
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            // otherwise, use the previous decode method to decode tokens to a String
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };

        // add this token to the current list of tokens already processed
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }

    // get access to inner tokenizer as a reference
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    // uses get_vocab() to get a hashmap of tokens to indexes
    // then grabs the index value with .get()
    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }
}

// enum ModelType {
//     Mistral(quantized_mistral),
// }

pub struct TextGeneration {
    model: GenerationModel,
    device: Device,
    tokenizer: TokenGenerator,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    pub fn new(
        model: GenerationModel,
        device: Device,
        _temp: Option<f64>,
        _top_p: Option<f64>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let seed: u64 = 299792458;
        let repeat_penalty: f32 = 1.1;
        let repeat_last_n: usize = 64;
        let logits_processor = LogitsProcessor::new(seed, Some(0.0), None);
        let tokenizer = TokenGenerator::new(model.tokenizer.clone());
        return Self {
            model,
            device,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        };
    }

    pub fn run(&mut self, prompt:&str) -> String {
        self.tokenizer.clear();
        let mut tokens: Vec<u32> = self.tokenizer.get_tokens(prompt.to_string());
        println!("Tokens are {:?}", tokens);

        let sample_len: usize = 1000;
        let seed: u64 = 299792458;
        let temperature: Option<f64> = Some(0.8);
        let top_p: Option<f64> = None;
        

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => panic!("cannot find the </s> token"),
        };

        let mut string = String::new();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.model.forward(&input,start_pos+index).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                .unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                println!("Found a token!");
                string.push_str(&t);
            }
        }
        string
    }
}

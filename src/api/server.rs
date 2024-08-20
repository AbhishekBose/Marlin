use std::net::TcpListener;
use std::sync::Arc;
use std::time::Instant;

use actix_web::dev::Server;
use actix_web::{web, App, HttpResponse, HttpServer};
use anyhow::Result;
use candle_core::Device;
use candle_transformers::generation;
use once_cell::sync::OnceCell;
use tokio::sync::Mutex;

use crate::model::backends::candle::language_model::LanguageModel;
use crate::model::backends::candle::text_generation::{GenerationModel, TextGeneration};

async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

pub struct AppState {
    predictor: OnceCell<Arc<Mutex<LanguageModel>>>,
}

pub struct QuantizedModelAppState {
    predictor: OnceCell<Arc<Mutex<TextGeneration>>>,
}

#[derive(serde::Deserialize)]
struct QueryPayload {
    query: String,
}

async fn create_embeddings(
    payload: web::Json<QueryPayload>,
    data: web::Data<AppState>,
) -> HttpResponse {
    let payload = &payload.query;
    let predictor = data.predictor.get().unwrap().lock().await;
    let start = Instant::now();
    let (embeddings, token_ids) = match predictor.encode(payload) {
        Ok((embeddings, token_ids)) => (embeddings, token_ids),
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {:?}", e)),
    };
    let duration = start.elapsed();
    println!("Time taken for encoding: {:?}", duration);
    // println!("Embeddings are --> {}",embeddings);
    HttpResponse::Ok().finish()
}

async fn generate_response(
    payload: web::Json<QueryPayload>,
    data: web::Data<QuantizedModelAppState>,
) -> HttpResponse {
    let payload = &payload.query;
    let mut predictor = data.predictor.get().unwrap().lock().await;
    let start = Instant::now();
    let response = predictor.run(&payload);
    println!("Response is : {}",response);
    let duration = start.elapsed();
    println!("Time taken for generation: {:?}", duration);
    // println!("Embeddings are --> {}",embeddings);
    HttpResponse::Ok().finish()
}



pub fn run_embedding_server(listener: TcpListener) -> Result<Server, std::io::Error> {
    let language_model = match LanguageModel::build_model_and_tokenizer() {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to initialize model and tokenizer: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ));
        }
    };
    let app_state = web::Data::new(AppState {
        predictor: OnceCell::new(),
    });
    let _ = app_state
        .predictor
        .set(Arc::new(Mutex::new(language_model)))
        .is_ok();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone()) // Pass the application state to the app
            .route("/health_check", web::get().to(health_check))
            .route("/embeddings", web::post().to(create_embeddings))
    })
    .listen(listener)?
    .run();

    Ok(server)
}

pub fn chat_quantized(listener: TcpListener) -> Result<Server, std::io::Error> {
    let generation_model = match GenerationModel::new() {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Failed to initialize model and tokenizer: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ));
        }
    };

    let text_generator = TextGeneration::new(generation_model, Device::Cpu,Some(0.0),None,Some(1.1),Some(1),None);
    let app_state = web::Data::new(QuantizedModelAppState {
        predictor: OnceCell::new(),
    });
    let _ = app_state
        .predictor
        .set(Arc::new(Mutex::new(text_generator)))
        .is_ok();

        let server = HttpServer::new(move || {
            App::new()
                .app_data(app_state.clone()) // Pass the application state to the app
                .route("/health_check", web::get().to(health_check))
                .route("/chat_quantized", web::post().to(generate_response))
        })
        .listen(listener)?
        .run();
    
        Ok(server)

}

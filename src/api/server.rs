use std::net::TcpListener;
use std::sync::Arc;
use std::time::Instant;

use actix_web::{App, HttpResponse, HttpServer, web};
use actix_web::dev::Server;
use anyhow::Result;
use once_cell::sync::OnceCell;
use tokio::sync::Mutex;

use crate::model::language_model::LanguageModel;

async fn health_check() -> HttpResponse {
    HttpResponse::Ok().finish()
}

pub struct AppState {
    predictor: OnceCell<Arc<Mutex<LanguageModel>>>
}

#[derive(serde::Deserialize)]
struct QueryPayload {
    query: String,
}

async fn create_embeddings(payload: web::Json<QueryPayload>,
                           data: web::Data<AppState>,) -> HttpResponse {
    let payload = &payload.query;
    let predictor = data.predictor.get().unwrap().lock().await;
    let start = Instant::now();
    let (embeddings,token_ids) = match predictor.encode(payload) {
        Ok((embeddings,token_ids)) => (embeddings,token_ids),
        Err(e) => return HttpResponse::InternalServerError().body(format!("Error: {:?}", e)),
    };
    let duration = start.elapsed();
    println!("Time taken for encoding: {:?}", duration);
    // println!("Embeddings are --> {}",embeddings);
    HttpResponse::Ok().finish()
}

pub fn run(listener: TcpListener) -> Result<Server, std::io::Error> {
    let language_model = match LanguageModel::build_model_and_tokenizer() {
            Ok(model) => model,
            Err(e) => {
                eprintln!("Failed to initialize model and tokenizer: {:?}", e);
                return Err(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()));
            }
        };
    let app_state = web::Data::new(AppState {
        predictor: OnceCell::new()
    });
    let _ = app_state.predictor.set(Arc::new(Mutex::new(language_model))).is_ok();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())  // Pass the application state to the app
            .route("/health_check", web::get().to(health_check))
            .route("/embeddings", web::post().to(create_embeddings))
    })
        .listen(listener)?
        .run();

    Ok(server)
}
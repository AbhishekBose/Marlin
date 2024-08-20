mod model;

use actix_web::{web, HttpResponse, Responder, HttpServer, App};
use serde::Deserialize;
use std::sync::Arc;
use env_logger::Env;
use tokio::sync::RwLock;

use crate::model::manager::ModelManager;
use crate::model::traits::Backend;

// Define the data structure for loading a model
#[derive(Deserialize)]
struct LoadModelRequest {
    model_path: String,
    model_type: String,
}

// Define the data structure for predictions
#[derive(Deserialize)]
struct PredictRequest {
    input_text: String,
}

async fn load_model(
    model_manager: web::Data<Arc<ModelManager>>,
    req: web::Json<LoadModelRequest>,
) -> impl Responder {
    let model_type = match req.model_type.as_str() {
        "onnx" => Backend::ONNX,
        _ => return HttpResponse::BadRequest().body("Invalid model type"),
    };

    match model_manager.load_model(req.model_path.clone(), model_type).await {
        Ok(model_id) => HttpResponse::Ok().json(model_id),
        Err(err) => HttpResponse::InternalServerError().body("Couldn't load model"),
    }
}

async fn predict(
    model_manager: web::Data<Arc<ModelManager>>,
    path: web::Path<u32>,
    req: web::Json<PredictRequest>,
) -> impl Responder {
    let model_id = path.into_inner();
    let input_text = &req.input_text;

    match model_manager.predict(model_id, input_text).await {
        Ok(prediction) => HttpResponse::Ok().json(prediction),
        Err(err) => HttpResponse::InternalServerError().body(err.to_string()),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    let model_manager = Arc::new(ModelManager::new());

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(model_manager.clone()))
            .route("/load_model", web::post().to(load_model))
            .route("/predict/{id}", web::post().to(predict))
    })
        .bind("127.0.0.1:8080")?
        .run()
        .await
}

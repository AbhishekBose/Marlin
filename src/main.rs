#[cfg(feature = "accelerate")]
extern crate accelerate_src;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::net::TcpListener;

use anyhow::Result;
use candle_core::Tensor;
use clap::Parser;
use env_logger::Env;

use api::server::run;

mod model;
mod api;

#[tokio::main]
async fn main() -> std::result::Result<(), std::io::Error> {
    // Bubble up the io::Error if we failed to bind the address
    // Otherwise call .await on our Server
    env_logger::init_from_env(Env::default().default_filter_or("info"));
    let listener = TcpListener::bind("127.0.0.1:8000")
        .expect("Failed to bind random port");
    run(listener)?.await
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

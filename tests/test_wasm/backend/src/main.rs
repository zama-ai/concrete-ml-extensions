use axum::{
    extract::{DefaultBodyLimit, Json},
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    routing::{get_service, post},
    Router,
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;
use std::{net::SocketAddr, path::PathBuf};
use tfhe::{safe_serialization::safe_deserialize, set_server_key, FheUint64, ServerKey};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, services::ServeDir};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

const DESER_LIMIT: u64 = 4_000_000_000;
const MAX_REQUEST_SIZE: usize = 4_000_000_000; // 4GB – according to wasm

#[derive(Deserialize)]
struct AddReq {
    ct1_b64: String,
    ct2_b64: String,
    sk_b64:  String,
}

#[tokio::main]
async fn main() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();

    let static_dir = PathBuf::from("../frontend");

    // Build the router
    let app = Router::new()
        .route(
            "/api/add",
            post(add)
                // lift the default 2 MiB extractor cap **only** here
                .layer(DefaultBodyLimit::max(MAX_REQUEST_SIZE)),
        )
        .nest_service("/", get_service(ServeDir::new(static_dir)))
        .layer(CorsLayer::very_permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], 8000));
    println!("➡ Open http://{addr}");

    axum::serve(TcpListener::bind(addr).await.unwrap(), app.into_make_service())
        .await
        .unwrap();
}

async fn add(Json(req): Json<AddReq>) -> impl IntoResponse {
    info!("Received add request");
    info!("Request sizes - ct1: {} bytes, ct2: {} bytes, sk: {} bytes", 
        req.ct1_b64.len(), 
        req.ct2_b64.len(), 
        req.sk_b64.len()
    );

    match process(req) {
        Ok(bytes) => {
            info!("Successfully processed request, returning result of size: {} bytes", bytes.len());
            let mut headers = HeaderMap::new();
            headers.insert(axum::http::header::CONTENT_TYPE, "application/octet-stream".parse().unwrap());
            (headers, bytes).into_response()
        }
        Err(msg) => {
            info!("Error processing request: {}", msg);
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": msg })),
            )
                .into_response()
        }
    }
}

fn process(r: AddReq) -> Result<Vec<u8>, &'static str> {
    info!("Processing request with server key");
    let sk_bytes = STANDARD.decode(r.sk_b64).map_err(|_| "bad sk_b64")?;
    info!("Server key decoded, size: {} bytes", sk_bytes.len());
    
    let sk: ServerKey = safe_deserialize(&mut std::io::Cursor::new(&sk_bytes), DESER_LIMIT)
        .map_err(|_| "sk deserialize")?;
    set_server_key(sk);
    info!("Server key set successfully");

    let c1 = STANDARD.decode(r.ct1_b64).map_err(|_| "bad ct1_b64")?;
    let c2 = STANDARD.decode(r.ct2_b64).map_err(|_| "bad ct2_b64")?;
    info!("Ciphertexts decoded - ct1: {} bytes, ct2: {} bytes", c1.len(), c2.len());

    let a1: Vec<FheUint64> = bincode::deserialize(&c1).map_err(|_| "ct1 deserialize")?;
    let a2: Vec<FheUint64> = bincode::deserialize(&c2).map_err(|_| "ct2 deserialize")?;
    info!("Ciphertexts deserialized - a1: {} elements, a2: {} elements", a1.len(), a2.len());

    if a1.len() != a2.len() {
        return Err("length mismatch");
    }

    let sum: Vec<FheUint64> = a1.iter().zip(a2).map(|(x, y)| x + y).collect();
    let result = bincode::serialize(&sum).map_err(|_| "serialize")?;
    info!("Operation completed, result size: {} bytes", result.len());
    Ok(result)
} 
use rouille::{Response, router};
use serde::{Deserialize, Serialize};
use tfhe::{FheUint64, set_server_key, ServerKey};
use tfhe::safe_serialization::safe_deserialize;
use base64::Engine;
use std::io::Cursor;
use std::io::Read;

#[derive(Serialize, Deserialize)]
struct FheAddRequest {
    ct1_b64: String,
    ct2_b64: String,
    sk_b64: String,
}

const DESERIALIZE_LIMIT: u64 = 1_000_000_000;

fn main() {
    let port = 8001;
    println!("Rust FHE Server listening on http://0.0.0.0:{}...", port);

    rouille::start_server(format!("0.0.0.0:{}", port), move |request| {
        if request.method() == "OPTIONS" {
            return Response::empty_204()
                .with_additional_header("Access-Control-Allow-Origin", "*")
                .with_additional_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                .with_additional_header("Access-Control-Allow-Headers", "Content-Type");
        }
        
        router!(request,
            (POST) (/fhe_add_rust_server) => {
                println!("Received FHE addition request");
                
                let mut body_string = String::new();
                let mut request_body_stream = match request.data() {
                    Some(stream) => stream,
                    None => {
                        eprintln!("No request body found");
                        return Response::json(&serde_json::json!({"error": "No request body found"}))
                            .with_status_code(400)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                if let Err(e) = request_body_stream.read_to_string(&mut body_string) {
                    eprintln!("Failed to read request body: {:?}", e);
                    return Response::json(&serde_json::json!({"error": "Failed to read request body"}))
                        .with_status_code(400)
                        .with_additional_header("Access-Control-Allow-Origin", "*");
                }

                println!("Attempting to parse JSON body: >>>{}<<<", body_string);

                let input: FheAddRequest = match serde_json::from_str(&body_string) {
                    Ok(req) => req,
                    Err(e) => {
                        eprintln!("Failed to parse JSON request. Body was: >>>{}<<<. Error: {:?}", body_string, e);
                        return Response::json(&serde_json::json!({"error": "Failed to parse JSON request"}))
                            .with_status_code(400)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                println!("Decoding base64 server key...");
                let sk_bytes = match base64::engine::general_purpose::STANDARD.decode(&input.sk_b64) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to decode sk_b64: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to decode sk_b64"}))
                            .with_status_code(400)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                println!("Deserializing server key using safe_deserialize...");
                let server_key: ServerKey = 
                    match safe_deserialize(&mut Cursor::new(&sk_bytes), DESERIALIZE_LIMIT) {
                    Ok(key) => key,
                    Err(e) => {
                        eprintln!("Failed to safe_deserialize server key: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to safe_deserialize server key"}))
                            .with_status_code(500)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                set_server_key(server_key);

                println!("Decoding base64 ciphertexts...");
                let ct1_bytes = match base64::engine::general_purpose::STANDARD.decode(&input.ct1_b64) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to decode ct1_b64: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to decode ct1_b64"}))
                            .with_status_code(400)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };
                let ct2_bytes = match base64::engine::general_purpose::STANDARD.decode(&input.ct2_b64) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to decode ct2_b64: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to decode ct2_b64"}))
                            .with_status_code(400)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                println!("Deserializing FHE arrays using bincode...");
                let fhe_array1: Vec<FheUint64> = match bincode::deserialize(&ct1_bytes) {
                    Ok(arr) => arr,
                    Err(e) => {
                        eprintln!("Failed to bincode::deserialize ct1: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to bincode::deserialize ct1"}))
                            .with_status_code(500)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };
                let fhe_array2: Vec<FheUint64> = match bincode::deserialize(&ct2_bytes) {
                    Ok(arr) => arr,
                    Err(e) => {
                        eprintln!("Failed to bincode::deserialize ct2: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to bincode::deserialize ct2"}))
                            .with_status_code(500)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };

                if fhe_array1.len() != fhe_array2.len() {
                    eprintln!("Array length mismatch: {} != {}", fhe_array1.len(), fhe_array2.len());
                    return Response::json(&serde_json::json!({"error": "Ciphertext arrays must have the same length"}))
                        .with_status_code(400)
                        .with_additional_header("Access-Control-Allow-Origin", "*");
                }

                println!("Performing FHE addition on arrays of length {}...", fhe_array1.len());
                let mut result_fhe: Vec<FheUint64> = Vec::with_capacity(fhe_array1.len());
                for (i, (fhe1, fhe2)) in fhe_array1.iter().zip(fhe_array2.iter()).enumerate() {
                    match std::panic::catch_unwind(|| fhe1 + fhe2) {
                        Ok(sum) => result_fhe.push(sum),
                        Err(e) => {
                            eprintln!("FHE addition failed at index {}: {:?}", i, e);
                            return Response::json(&serde_json::json!({"error": "FHE addition operation failed"}))
                                .with_status_code(500)
                                .with_additional_header("Access-Control-Allow-Origin", "*");
                        }
                    }
                }

                println!("Serializing result...");
                let result_bytes = match bincode::serialize(&result_fhe) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("Failed to serialize result: {:?}", e);
                        return Response::json(&serde_json::json!({"error": "Failed to serialize result"}))
                            .with_status_code(500)
                            .with_additional_header("Access-Control-Allow-Origin", "*");
                    }
                };
                
                println!("Successfully completed FHE addition");
                Response::from_data("application/octet-stream", result_bytes)
                    .with_additional_header("Access-Control-Allow-Origin", "*")
            },
            _ => {
                println!("Received invalid request: {} {}", request.method(), request.url());
                Response::empty_404()
                    .with_additional_header("Access-Control-Allow-Origin", "*")
            }
        )
    })
} 
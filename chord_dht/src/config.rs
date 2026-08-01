pub struct Config {
    pub http_bind: String,
    pub grpc_bind: String,
    pub self_uri: String,
    pub join_uri: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            http_bind: std::env::var("HTTP_BIND").unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            grpc_bind: std::env::var("GRPC_BIND").unwrap_or_else(|_| "0.0.0.0:50051".to_string()),
            self_uri: std::env::var("SELF_URI")
                .unwrap_or_else(|_| "http://127.0.0.1:50051".to_string()),
            join_uri: std::env::var("JOIN_URI").ok(),
        }
    }
}

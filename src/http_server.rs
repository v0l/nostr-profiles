use crate::db::Database;
use crate::job_queue::JobQueue;
use axum::{
    extract::{Request, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Json, Response},
    routing::get,
    Router,
};
use base64::engine::Engine as _;
use hyper_util::rt::TokioIo;
use nostr_relay_builder::LocalRelay;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};
use std::sync::Arc;

/// Shared application state: the database, the search relay, and the job queue.
#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub relay: LocalRelay,
    pub job_queue: Arc<JobQueue>,
}

#[derive(Serialize)]
struct ProfileResponse {
    pubkey: String,
    name: Option<String>,
    about: Option<String>,
    picture: Option<String>,
    nip05: Option<String>,
    event_count: usize,
    is_classified: bool,
    metadata_json: Option<String>,
    classification: Option<ClassificationResponse>,
}

#[derive(Serialize)]
struct ClassificationResponse {
    scores: std::collections::HashMap<String, f64>,
    bio: String,
    confidence: f64,
    analyzed_at: Option<String>,
}

#[derive(Serialize)]
struct StatsResponse {
    total_profiles: i64,
    classified_profiles: i64,
    queue_size: usize,
    labels: LabelStats,
}

#[derive(Serialize)]
struct LabelStats {
    total_unique_labels: i64,
    label_counts: Vec<LabelCount>,
}

#[derive(Serialize)]
struct LabelCount {
    label: String,
    count: i64,
}

#[derive(Serialize)]
pub struct RecentClassification {
    pub pubkey: String,
    pub name: Option<String>,
    pub picture: Option<String>,
    pub scores: std::collections::HashMap<String, f64>,
    pub bio: String,
    pub confidence: f64,
    pub analyzed_at: Option<String>,
    pub metadata_json: Option<String>,
}

#[derive(Deserialize)]
struct RecentQuery {
    limit: Option<i64>,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: Option<i64>,
}

pub async fn serve(db: Arc<Database>, relay: LocalRelay, job_queue: Arc<JobQueue>, port: u16) {
    let state = AppState { db, relay, job_queue };

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/api/profile/{pubkey}", get(get_profile))
        .route("/api/recent", get(get_recent))
        .route("/api/search", get(search))
        .route("/api/stats", get(get_stats))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    println!("Dashboard running at http://{}", addr);
    println!("Nostr search relay at ws://{}/", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

/// Root handler: serves the dashboard HTML for normal requests,
/// upgrades to WebSocket for the nostr search relay.
async fn root_handler(State(state): State<AppState>, req: Request) -> Response {
    // Check if this is a WebSocket upgrade request
    let is_ws_upgrade = req
        .headers()
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false);

    if is_ws_upgrade {
        ws_upgrade(state, req).await
    } else {
        serve_dashboard().await.into_response()
    }
}

async fn serve_dashboard() -> Html<&'static str> {
    Html(include_str!("../dashboard.html"))
}

async fn get_profile(
    State(state): State<AppState>,
    axum::extract::Path(pubkey): axum::extract::Path<String>,
) -> impl IntoResponse {
    let db = &state.db;
    let pubkey = pubkey.trim();

    if pubkey.is_empty() {
        return (StatusCode::BAD_REQUEST, "pubkey is required").into_response();
    }

    // Get profile details
    let profile = match db.get_profile_by_pubkey(pubkey).await {
        Ok(Some(p)) => p,
        Ok(None) => return (StatusCode::NOT_FOUND, "Profile not found").into_response(),
        Err(_) => return (StatusCode::NOT_FOUND, "Profile not found").into_response(),
    };

    // Get event count dynamically
    let event_count = db.get_profile_event_count(pubkey).await.unwrap_or(0);

    let mut response = ProfileResponse {
        pubkey: profile.pubkey,
        name: profile.name,
        about: profile.about,
        picture: profile.picture,
        nip05: profile.nip05,
        event_count,
        is_classified: profile.is_classified,
        metadata_json: profile.metadata_json,
        classification: None,
    };

    // Get classification if exists
    if profile.is_classified {
        if let Ok(classification) = db.get_classification(&pubkey).await {
            let analyzed_at = sqlx::query_scalar::<_, Option<chrono::DateTime<chrono::Utc>>>(
                r#"SELECT analyzed_at FROM classifications WHERE pubkey = ?"#,
            )
            .bind(&pubkey)
            .fetch_one(&db.pool)
            .await
            .ok()
            .flatten()
            .map(|t| t.to_rfc3339());

            response.classification = Some(ClassificationResponse {
                scores: classification.scores,
                bio: classification.bio,
                confidence: classification.confidence,
                analyzed_at,
            });
        }
    }

    Json(response).into_response()
}

async fn get_recent(
    State(state): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<RecentQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(20).min(100);
    match state.db.get_recent_classifications(limit).await {
        Ok(results) => Json(results).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
    }
}

async fn search(
    State(state): State<AppState>,
    axum::extract::Query(query): axum::extract::Query<SearchQuery>,
) -> impl IntoResponse {
    let q = query.q.trim();
    if q.is_empty() {
        return (StatusCode::BAD_REQUEST, "q is required").into_response();
    }
    let limit = query.limit.unwrap_or(20).min(100);
    match state.db.search_classifications(q, limit).await {
        Ok(results) => Json(results).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
    }
}

async fn get_stats(State(state): State<AppState>) -> impl IntoResponse {
    let (total_profiles, classified_profiles, total_unique_labels, label_counts) =
        state.db.get_stats().await.unwrap_or((0, 0, 0, Vec::new()));
    let queue_size = state.job_queue.queue_len().await;

    let stats = StatsResponse {
        total_profiles,
        classified_profiles,
        queue_size,
        labels: LabelStats {
            total_unique_labels,
            label_counts: label_counts
                .into_iter()
                .map(|(label, count)| LabelCount { label, count })
                .collect(),
        },
    };

    Json(stats).into_response()
}

/// WebSocket upgrade handler for the nostr search relay.
///
/// This manually constructs the HTTP 101 Switching Protocols response and passes
/// the raw upgraded stream to `LocalRelay::take_connection()`, which handles the
/// nostr relay protocol (REQ, EVENT, CLOSE messages).
async fn ws_upgrade(state: AppState, req: Request) -> Response {
    let ws_key = req
        .headers()
        .get(header::SEC_WEBSOCKET_KEY)
        .cloned();

    if ws_key.is_none() {
        return (StatusCode::BAD_REQUEST, "Missing Sec-WebSocket-Key").into_response();
    }

    // Derive the Sec-WebSocket-Accept value per RFC 6455
    let accept_key = derive_ws_accept_key(ws_key.unwrap().as_bytes());

    // Extract the hyper OnUpgrade future before consuming the request
    let on_upgrade = hyper::upgrade::on(req);

    let relay = state.relay.clone();

    // Spawn a task to handle the upgraded connection
    tokio::spawn(async move {
        match on_upgrade.await {
            Ok(upgraded) => {
                let stream = TokioIo::new(upgraded);
                // Use a placeholder address since we don't have the real one
                let addr = std::net::SocketAddr::new(
                    std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED),
                    0,
                );
                if let Err(e) = relay.take_connection(stream, addr).await {
                    tracing::error!("Relay connection error: {}", e);
                }
            }
            Err(e) => {
                tracing::error!("WebSocket upgrade failed: {}", e);
            }
        }
    });

    // Return 101 Switching Protocols response
    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(header::CONNECTION, "upgrade")
        .header(header::UPGRADE, "websocket")
        .header(header::SEC_WEBSOCKET_ACCEPT, accept_key)
        .body(axum::body::Body::empty())
        .unwrap()
}

/// Derive the Sec-WebSocket-Accept value from the client's Sec-WebSocket-Key.
/// Per RFC 6455 section 4.2.2, this is SHA-1(key + GUID) base64-encoded.
fn derive_ws_accept_key(key: &[u8]) -> String {
    const WS_GUID: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut sha1 = Sha1::new();
    sha1.update(key);
    sha1.update(WS_GUID);
    base64::engine::general_purpose::STANDARD.encode(sha1.finalize())
}

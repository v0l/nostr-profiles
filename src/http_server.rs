use crate::db::Database;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, Json, IntoResponse},
    routing::get,
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize)]
struct ProfileResponse {
    pubkey: String,
    name: Option<String>,
    about: Option<String>,
    picture: Option<String>,
    nip05: Option<String>,
    event_count: usize,
    is_classified: bool,
    classification: Option<ClassificationResponse>,
}

#[derive(Serialize)]
struct ClassificationResponse {
    labels: Vec<String>,
    bio: String,
    confidence: f64,
}

#[derive(Serialize)]
pub struct RecentClassification {
    pub pubkey: String,
    pub name: Option<String>,
    pub picture: Option<String>,
    pub labels: Vec<String>,
    pub bio: String,
    pub confidence: f64,
    pub analyzed_at: Option<String>,
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

pub async fn serve(db: Arc<Database>, port: u16) {
    let app = Router::new()
        .route("/", get(serve_dashboard))
        .route("/api/profile/{pubkey}", get(get_profile))
        .route("/api/recent", get(get_recent))
        .route("/api/search", get(search))
        .with_state(db);

    let addr = format!("0.0.0.0:{}", port);
    println!("Dashboard running at http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn serve_dashboard() -> Html<&'static str> {
    Html(include_str!("../dashboard.html"))
}

async fn get_profile(
    State(db): State<Arc<Database>>,
    axum::extract::Path(pubkey): axum::extract::Path<String>,
) -> impl IntoResponse {
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
        classification: None,
    };

    // Get classification if exists
    if profile.is_classified {
        if let Ok(classification) = db.get_classification(&pubkey).await {
            response.classification = Some(ClassificationResponse {
                labels: classification.labels,
                bio: classification.bio,
                confidence: classification.confidence,
            });
        }
    }

    Json(response).into_response()
}

async fn get_recent(
    State(db): State<Arc<Database>>,
    axum::extract::Query(query): axum::extract::Query<RecentQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(20).min(100);
    match db.get_recent_classifications(limit).await {
        Ok(results) => Json(results).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
    }
}

async fn search(
    State(db): State<Arc<Database>>,
    axum::extract::Query(query): axum::extract::Query<SearchQuery>,
) -> impl IntoResponse {
    let q = query.q.trim();
    if q.is_empty() {
        return (StatusCode::BAD_REQUEST, "q is required").into_response();
    }
    let limit = query.limit.unwrap_or(20).min(100);
    match db.search_classifications(q, limit).await {
        Ok(results) => Json(results).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response(),
    }
}

# ── Dashboard build ────────────────────────────────────────────────────────────
FROM oven/bun:1 AS dashboard-build
WORKDIR /dashboard
COPY dashboard/package.json dashboard/bun.lock ./
RUN bun install --frozen-lockfile
COPY dashboard/ ./
RUN bun run build

# ── Rust dependency cache ─────────────────────────────────────────────────────
FROM voidic/rust-ffmpeg AS rust-deps
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && \
    rm -f target/release/nostr-classify \
          target/release/deps/nostr-classify-* \
          target/release/deps/libnostr_classify-*

# ── Rust application build ────────────────────────────────────────────────────
FROM rust-deps AS rust-build
COPY src ./src
COPY migrations ./migrations
RUN touch src/main.rs
RUN cargo build --release && \
    mkdir -p /app/bin && \
    cp target/release/nostr-classify /app/bin/nostr-classify

# ── Runtime image ─────────────────────────────────────────────────────────────
FROM debian:trixie-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libx264-164 \
        libx265-215 \
        libvpx9 \
        libopus0 \
        libwebp7 \
        libwebpmux3 \
        libdav1d7 \
        va-driver-all \
        libva-drm2 \
        libva-x11-2 \
        libva-wayland2 \
        libva-glx2 && \
    if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        apt-get install -y --no-install-recommends libvpl2; \
    fi && \
    rm -rf /var/lib/apt/lists/*

COPY --from=rust-build /app/bin       ./bin
COPY --from=rust-build /app/src/ffmpeg/lib/ /lib
COPY --from=dashboard-build /dashboard/dist ./dashboard/dist

ENV RUST_BACKTRACE=1

EXPOSE 3000
CMD ["./bin/nostr-classify"]

FROM rust:trixie AS builder
WORKDIR /usr/src/app
COPY Cargo.toml Cargo.lock dashboard.html ./
COPY src/ src/
RUN cargo build --release

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/src/app/target/release/nostr-classify /usr/local/bin/nostr-classify
WORKDIR /app
EXPOSE 3000
CMD ["nostr-classify"]

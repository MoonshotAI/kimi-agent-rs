# Kimi Agent (Rust)

Kimi Agent is the Rust implementation of the Kimi Code CLI agent kernel. It runs as a
wire-only server (no Shell/Print/ACP UI) and is intended to be used by external Wire
clients or as a sidecar for the Python CLI.

## Build

```sh
cargo build -p kimi-agent
```

## Run

```sh
./target/debug/kimi-agent
# or
cargo run -p kimi-agent
```

## Relationship to `kimi-cli`

This repository is decoupled from the Python `kimi-cli` release cadence. For parity
tracking, the `kimi-cli` repo is included as a submodule at `kimi-cli/` and should be
updated whenever Rust changes need to sync with Python behavior or data formats.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

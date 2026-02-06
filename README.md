# Kimi Agent (Rust)

Rust implementation of [Kimi Code CLI](https://github.com/MoonshotAI/kimi-cli). Wire-only JSON-RPC agent server over stdio.

## Build & Run

```sh
cargo build -p kimi-agent
./target/debug/kimi-agent
```

## Test

```sh
cargo test                  # all
cargo test -p kimi-agent    # agent
cargo test -p kosong        # LLM abstraction
cargo test -p kaos          # OS abstraction
```

## Workspace

| Crate | Purpose |
|-------|---------|
| `kimi-agent` | Main binary — wire server, tools, agent loop, MCP |
| `kosong` | LLM abstraction — messages, tool schemas, providers |
| `kaos` | OS abstraction — LocalKaos, path semantics |

## Relationship to Python

This repo is a Rust rewrite of the Python `kimi-cli` runtime. The two must stay compatible on wire protocol, message formats, `~/.kimi` data layout, tool schemas, and all other externally observable behavior. Python is the source of truth.

The Python repo is pinned as a git submodule at `kimi-cli/`:

```sh
git submodule update --init
```

Version numbers must always match `kimi-cli` exactly.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

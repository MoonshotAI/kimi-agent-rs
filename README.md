# Kimi Agent (Rust)

Kimi Agent is the Rust implementation of the
[Kimi Code CLI](https://github.com/MoonshotAI/kimi-cli) agent runtime.
It runs as a **wire-only** JSON-RPC server over stdio — no Shell, Print, or ACP
UI is included. External wire clients (or the Python CLI acting as a frontend)
connect to `kimi-agent` to drive the agent loop.

## Workspace layout

| Crate | Purpose |
|-------|---------|
| **`kimi-agent`** | Main binary — wire server, tools, agent loop, MCP client |
| **`kosong`** | LLM abstraction layer (messages, tool schemas, providers) |
| **`kaos`** | OS abstraction layer (LocalKaos, path semantics) |

## Build

```sh
cargo build -p kimi-agent
```

## Run

Running the binary without a subcommand starts the JSON-RPC wire server on
stdio:

```sh
./target/debug/kimi-agent        # or: cargo run -p kimi-agent
```

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `info` | Show version and protocol information (`--json` for machine-readable output) |
| `mcp`  | Manage MCP server configurations (`add`, `remove`, `list`, `auth`, `reset-auth`, `test`) |

## Test

```sh
cargo test                  # entire workspace
cargo test -p kimi-agent    # agent crate only
cargo test -p kosong        # LLM abstraction
cargo test -p kaos          # OS abstraction
```

### End-to-end tests against the Python suite

The `kimi-cli` Python test suite can be pointed at the Rust binary:

```sh
cargo build -p kimi-agent
cd kimi-cli
KIMI_E2E_WIRE_CMD=../target/debug/kimi-agent uv run pytest tests_e2e
```

## Lint

```sh
cargo fmt
cargo clippy --workspace --all-targets
```

## Relationship to `kimi-cli` (Python)

This repository is the Rust rewrite of the Python `kimi-cli` runtime. The two
implementations must stay compatible on the wire protocol, message envelopes,
data formats under `~/.kimi`, tool schemas, and all other externally observable
behavior. When in doubt, the Python code and `docs/zh/` in `kimi-cli` are the
source of truth.

### Submodule

The Python [`kimi-cli`](https://github.com/MoonshotAI/kimi-cli) repository is
pinned as a **git submodule** at `kimi-cli/`. Updating the submodule records
which Python commit the Rust implementation currently tracks. After cloning,
run:

```sh
git submodule update --init
```

### Versioning

The Rust workspace version and the Python package version **must use the same
`MAJOR.MINOR` numbers**. The patch component is always `0` (e.g. `1.8.0`).
When a minor bump happens in Python, the Rust version is bumped to match —
though Rust may lag behind in feature completeness, the version number stays
aligned so that both sides can be identified at a glance.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

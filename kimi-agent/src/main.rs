#[tokio::main]
async fn main() {
    if let Err(err) = kimi_agent::cli::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

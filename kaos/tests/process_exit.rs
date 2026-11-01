#![cfg(unix)]

use std::time::Duration;

use kaos::{Kaos, LocalKaos};
use tokio::time::timeout;

#[tokio::test]
async fn wait_preserves_signal_returncodes() {
    for (signal, expected) in [("TERM", -15), ("KILL", -9)] {
        let mut process = LocalKaos::new()
            .exec(&["/bin/sh".into(), "-c".into(), format!("kill -{signal} $$")])
            .await
            .expect("spawn shell");
        let code = timeout(Duration::from_secs(5), process.wait())
            .await
            .expect("shell should terminate")
            .expect("wait for shell");
        assert_eq!(code, expected);
        assert_eq!(process.returncode(), Some(expected));
    }
}

#[tokio::test]
async fn polling_preserves_signal_returncodes() {
    for (signal, expected) in [("TERM", -15), ("KILL", -9)] {
        let mut process = LocalKaos::new()
            .exec(&["/bin/sh".into(), "-c".into(), format!("kill -{signal} $$")])
            .await
            .expect("spawn shell");
        let code = timeout(Duration::from_secs(5), async {
            loop {
                if let Some(code) = process.returncode() {
                    break code;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("shell should terminate");
        assert_eq!(code, expected);
        assert_eq!(process.wait().await.expect("wait after polling"), expected);
    }
}

#[tokio::test]
async fn normal_exit_codes_are_unchanged() {
    for expected in [0, 7] {
        let mut process = LocalKaos::new()
            .exec(&["/bin/sh".into(), "-c".into(), format!("exit {expected}")])
            .await
            .expect("spawn shell");
        assert_eq!(process.wait().await.expect("wait for shell"), expected);
        assert_eq!(process.returncode(), Some(expected));
    }
}

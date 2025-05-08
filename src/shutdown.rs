use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct Shutdown {
    token: CancellationToken,
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            token: CancellationToken::new(),
        }
    }
    pub fn subscribe(&self) -> CancellationToken {
        self.token.clone()
    }
    pub fn shutdown(&self) {
        self.token.cancel();
    }
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

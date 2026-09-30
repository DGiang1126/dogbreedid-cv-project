"""Early stopping on validation Macro-F1 (higher is better)."""


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.wait = 0

    def step(self, score: float) -> tuple[bool, bool]:
        """Return (improved, should_stop)."""
        if score > self.best + self.min_delta:
            self.best = score
            self.wait = 0
            return True, False
        self.wait += 1
        return False, self.wait >= self.patience

"""
PyTorch MLP regressor for NBA point prediction.
Wraps a PyTorch model in a sklearn-compatible API (fit/predict)
so it plugs into the existing ensemble pipeline without changes.

Architecture: Input → 128(BN,ReLU,Drop) → 64(BN,ReLU,Drop) → 32(ReLU) → 1
"""

import numpy as np

from config.logging_config import setup_logging

logger = setup_logging("neural_net")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. TorchMLPRegressor will be unavailable.")


class _NBANet(nn.Module if _TORCH_AVAILABLE else object):
    """
    Feed-forward network for point prediction.
    Input → 128(BN,ReLU,Drop0.2) → 64(BN,ReLU,Drop0.2) → 32(ReLU) → 1
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TorchMLPRegressor:
    """
    Sklearn-compatible wrapper around _NBANet.
    Supports fit(X, y) and predict(X) with numpy arrays.
    Serializable via joblib (state_dict is stored as plain dict).
    """

    def __init__(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: str = "cpu",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchMLPRegressor. Run: pip install torch")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = torch.device(device)
        self._model: "_NBANet | None" = None
        self._state_dict: dict | None = None  # for joblib serialization
        self._n_features: int = 0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._impute_values: np.ndarray | None = None

    # ── internal helpers ─────────────────────────────────────────────────────

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / (self._std + 1e-8)

    def _to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        return torch.tensor(X, dtype=torch.float32, device=self.device)

    def _build_model(self, n_features: int) -> "_NBANet":
        model = _NBANet(n_features).to(self.device)
        if self._state_dict is not None:
            model.load_state_dict(self._state_dict)
        return model

    # ── sklearn API ──────────────────────────────────────────────────────────

    def fit(self, X, y):
        """Train on numpy arrays X (n_samples, n_features), y (n_samples,)."""
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # Replace NaN/inf with column medians (mirrors sklearn SimpleImputer).
        # A column that is entirely NaN has no median, so fall back to 0.0 and
        # keep the result finite.
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0, posinf=0.0, neginf=0.0)
        self._impute_values = col_medians
        nan_mask = ~np.isfinite(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        # Store normalization stats
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._n_features = X.shape[1]

        X_norm = self._normalize(X)

        dataset = TensorDataset(self._to_tensor(X_norm), self._to_tensor(y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = _NBANet(self._n_features).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0
        # Seed with the initial weights: if epoch 1 never improves on inf
        # (which happens whenever the loss is NaN), the restore below used to
        # raise UnboundLocalError instead of failing cleanly.
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for Xb, yb in loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(Xb)

            avg_loss = epoch_loss / len(dataset)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug(f"Early stop at epoch {epoch+1}, best loss={best_loss:.4f}")
                    break

        # Restore best weights
        model.load_state_dict(best_state)
        self._model = model
        self._state_dict = {k: v.cpu() for k, v in best_state.items()}
        logger.info(f"TorchMLP trained: {self._n_features} features, best_loss={best_loss:.4f}")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict point totals for numpy array X."""
        if self._model is None:
            # Try to restore from saved state_dict
            if self._state_dict is not None and self._n_features > 0:
                self._model = self._build_model(self._n_features)
            else:
                raise RuntimeError("TorchMLPRegressor has not been fitted yet.")

        X = np.array(X, dtype=np.float32)
        # Impute with the values learned during fit, not zeros — imputing zeros
        # here while fit() used medians made every missing feature land far
        # from where the model was trained to expect it.
        col_medians = getattr(self, "_impute_values", None)
        if col_medians is None or len(col_medians) != X.shape[1]:
            col_medians = np.zeros(X.shape[1])
        nan_mask = ~np.isfinite(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        X_norm = self._normalize(X)

        self._model.eval()
        with torch.no_grad():
            tensor = self._to_tensor(X_norm)
            out = self._model(tensor).cpu().numpy()

        return out.astype(np.float32)

    # ── joblib serialization support ──────────────────────────────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        # Store state_dict as plain dict (already CPU tensors)
        if self._model is not None:
            state["_state_dict"] = {k: v.cpu() for k, v in self._model.state_dict().items()}
        state["_model"] = None  # Don't pickle the live module
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Model will be rebuilt lazily on first predict() call

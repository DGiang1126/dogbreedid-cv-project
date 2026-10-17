"""Scalar Temperature Scaling for DogBreedID E6.

The classifier is not trained here. Only one scalar temperature is optimized
against NLL on the approved Calibration split.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_logits_labels(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {tuple(logits.shape)}")
    if logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits must contain at least one sample and two classes")
    if not torch.is_floating_point(logits):
        logits = logits.float()
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits contain NaN or Inf")

    if labels.ndim != 1:
        raise ValueError(f"labels must have shape [N], got {tuple(labels.shape)}")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(
            f"logits/labels length mismatch: {logits.shape[0]} vs {labels.shape[0]}"
        )
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    if torch.is_floating_point(labels):
        if not torch.isfinite(labels).all().item():
            raise ValueError("labels contain NaN or Inf")
        if not torch.equal(labels, labels.round()):
            raise ValueError("labels must contain integer class IDs")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if (labels < 0).any().item() or (labels >= logits.shape[1]).any().item():
        raise ValueError("labels contain class IDs outside [0, C-1]")
    return logits, labels


class TemperatureScaler(nn.Module):
    """One positive scalar temperature represented by an unconstrained log T."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(float(initial_temperature)) or initial_temperature <= 0:
            raise ValueError("initial_temperature must be finite and > 0")
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(initial_temperature)), dtype=torch.float64)
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise ValueError("logits contain NaN or Inf")
        t = self.temperature.to(device=logits.device, dtype=logits.dtype)
        if not torch.isfinite(t).item() or t.item() <= 0:
            raise RuntimeError("learned temperature is not finite and positive")
        return logits / t

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        learning_rate: float = 0.1,
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """Fit log-temperature by minimizing cross-entropy/NLL with LBFGS.

        The best parameter encountered (including the initial value) is kept,
        so the returned NLL cannot be worse merely because the last LBFGS step
        was not the best one. If there is no improvement, the returned value
        remains the initial temperature and the caller should report that fact.
        """
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and > 0")
        if int(max_iterations) < 1:
            raise ValueError("max_iterations must be >= 1")

        logits, labels = _validate_logits_labels(logits, labels)
        # Float64 is inexpensive for [300, 30] logits and helps stabilize the
        # scalar optimization without touching model parameters.
        fit_logits = logits.detach().to(dtype=torch.float64)
        fit_labels = labels.detach()
        self.log_temperature.data = self.log_temperature.data.to(
            device=fit_logits.device, dtype=torch.float64
        )

        with torch.no_grad():
            initial_nll = float(F.cross_entropy(fit_logits, fit_labels).item())
        best_loss = initial_nll
        best_log_t = self.log_temperature.detach().clone()
        closure_evaluations = 0

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=float(learning_rate),
            max_iter=int(max_iterations),
        )

        def closure() -> torch.Tensor:
            nonlocal best_loss, best_log_t, closure_evaluations
            optimizer.zero_grad(set_to_none=True)
            scaled_logits = fit_logits / self.log_temperature.exp()
            loss = F.cross_entropy(scaled_logits, fit_labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("LBFGS produced a non-finite NLL")
            loss.backward()
            closure_evaluations += 1
            value = float(loss.detach().item())
            if value < best_loss:
                best_loss = value
                best_log_t = self.log_temperature.detach().clone()
            return loss

        optimizer.step(closure)
        with torch.no_grad():
            self.log_temperature.copy_(best_log_t)
            final_temperature = float(self.temperature.item())
            final_nll = float(
                F.cross_entropy(fit_logits / self.temperature, fit_labels).item()
            )

        if not math.isfinite(final_temperature) or final_temperature <= 0:
            raise RuntimeError("Temperature optimization returned invalid T")
        if not math.isfinite(final_nll):
            raise RuntimeError("Temperature optimization returned non-finite NLL")

        return {
            "temperature": final_temperature,
            "nll_before": initial_nll,
            "nll_after": final_nll,
            "nll_improved": final_nll < initial_nll - 1e-12,
            "optimizer": "LBFGS",
            "learning_rate": float(learning_rate),
            "max_iterations": int(max_iterations),
            "closure_evaluations": int(closure_evaluations),
            "parameterization": "log_temperature",
            "objective": "nll",
        }


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    initial_temperature: float = 1.0,
    learning_rate: float = 0.1,
    max_iterations: int = 100,
) -> dict[str, Any]:
    """Convenience wrapper returning fit diagnostics and the learned T."""
    scaler = TemperatureScaler(initial_temperature=initial_temperature)
    return scaler.fit(
        logits,
        labels,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
    )


def save_temperature(
    path: str | Path, temperature: float, *, metadata: Mapping[str, Any] | None = None
) -> Path:
    """Persist a positive finite temperature in a JSON artifact.

    Metadata fields are copied first; the validated ``temperature`` field is
    then written by this function so it cannot be accidentally overridden.
    """
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("temperature must be finite and > 0")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata or {})
    payload["temperature"] = value
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return output


def load_temperature(path: str | Path) -> float:
    """Load and validate the scalar Temperature from an E6 JSON artifact."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Temperature artifact not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "temperature" not in payload:
        raise ValueError("Temperature JSON must be an object containing 'temperature'")
    try:
        value = float(payload["temperature"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Temperature JSON field must be numeric") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Temperature JSON field must be finite and > 0")
    return value

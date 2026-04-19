"""Typed experiment configs (Pydantic): validate, coerce, and fill defaults."""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)

LabelMode = Literal[
    "cross_sectional_return",
    "raw_return",
    "equal_weighted_return",
    "sector_neutral_return",
]
RankingLoss = Literal["mse", "listnet", "approx_ndcg"]
StoreBackend = Literal["csv", "parquet"]
DataSource = Literal["rest", "mcp"]


class SingleSymbolExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    experiment_mode: Literal["single_symbol"] | None = None
    symbol: str = Field(min_length=1)
    timeframes: list[str] = Field(min_length=1)
    prediction_timeframe: str = "daily"
    lookbacks: dict[str, int] = Field(default_factory=dict)
    lookback: int = Field(default=32, ge=1)
    max_seq_len: int = Field(default=256, ge=1)

    train_bars: int = Field(ge=1)
    val_bars: int = Field(ge=1)
    test_bars: int = Field(ge=1)
    step_bars: int = Field(ge=1)

    d_model: int = Field(default=64, ge=1)
    nhead: int = Field(default=4, ge=1)
    num_layers: int = Field(default=2, ge=1)
    dim_feedforward: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.0005, gt=0.0)
    loss_alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    device: str = "auto"
    seed: int = 42
    default_threshold: float = 0.5
    artifacts_dir: str = "artifacts"
    cache_dir: str = "data"
    synthetic_n_daily: int = Field(default=1200, ge=1)

    use_adjusted_daily: bool = True
    use_adjusted_weekly: bool = True
    use_adjusted_monthly: bool = True
    intraday_month: str | None = None
    intraday_extended_hours: bool = False
    intraday_outputsize: str = "full"
    daily_outputsize: str = "full"

    early_stopping_patience: int = Field(default=0, ge=0)
    lr_reduce_on_plateau: bool = False
    lr_scheduler_patience: int = Field(default=3, ge=1)
    lr_scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    lr_scheduler_min_lr: float = Field(default=1e-7, ge=0.0)

    inference_batch_size: int = Field(default=256, ge=1)

    @model_validator(mode="before")
    @classmethod
    def warn_unknown_keys_single(cls, data: Any) -> Any:
        """Log unknown YAML keys so typos are visible while keeping ``extra=ignore``."""
        if not isinstance(data, dict):
            return data
        known = set(cls.model_fields.keys())
        for k in data:
            if k not in known:
                logger.warning("Unknown config key %r (typo?)", k)
        return data

    @field_validator("symbol")
    @classmethod
    def strip_symbol(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("symbol must be non-empty")
        return s

    @field_validator("timeframes", mode="before")
    @classmethod
    def coerce_timeframes(cls, v: Any) -> list[str]:
        if not v:
            raise ValueError("timeframes must be non-empty")
        return [str(x) for x in v]

    @field_validator("nhead")
    @classmethod
    def nhead_divides_d_model(cls, v: int, info: ValidationInfo) -> int:
        dm = info.data.get("d_model")
        if dm is not None and int(dm) % int(v) != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v


class UniverseExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    experiment_mode: Literal["universe"] | None = None
    symbols: list[str] = Field(min_length=1)
    target_symbol: str | None = None
    timeframe: str = "daily"
    lookback: int = Field(ge=2)
    min_coverage_symbols: int | None = None
    label_mode: LabelMode = "cross_sectional_return"
    sector_map_path: str | None = None

    store: StoreBackend | None = None
    loss: RankingLoss = "mse"
    portfolio_sim: dict[str, Any] | None = None
    features: list[str] | None = None
    data_source: DataSource = "rest"

    train_bars: int = Field(ge=1)
    val_bars: int = Field(ge=1)
    test_bars: int = Field(ge=1)
    step_bars: int = Field(ge=1)

    d_model: int = Field(default=64, ge=1)
    nhead: int = Field(default=4, ge=1)
    num_temporal_layers: int = Field(default=2, ge=1)
    num_cross_layers: int = Field(default=1, ge=1)
    dim_feedforward: int = Field(default=128, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    epochs: int = Field(default=8, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=0.0005, gt=0.0)

    device: str = "auto"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    cache_dir: str = "data"
    synthetic_n_bars: int = Field(default=600, ge=1)
    save_models: bool = False

    use_adjusted_daily: bool = True
    use_adjusted_weekly: bool = True
    use_adjusted_monthly: bool = True
    intraday_month: str | None = None
    intraday_extended_hours: bool = False
    intraday_outputsize: str = "full"
    daily_outputsize: str = "full"

    early_stopping_patience: int = Field(default=0, ge=0)
    lr_reduce_on_plateau: bool = False
    lr_scheduler_patience: int = Field(default=3, ge=1)
    lr_scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    lr_scheduler_min_lr: float = Field(default=1e-7, ge=0.0)

    inference_batch_size: int = Field(default=128, ge=1)

    @model_validator(mode="before")
    @classmethod
    def warn_unknown_keys_universe(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        known = set(cls.model_fields.keys())
        for k in data:
            if k not in known:
                logger.warning("Unknown config key %r (typo?)", k)
        return data

    @field_validator("symbols", mode="before")
    @classmethod
    def upper_symbols(cls, v: Any) -> list[str]:
        if not v:
            raise ValueError("symbols must be non-empty")
        return [str(s).upper() for s in v]

    @field_validator("nhead")
    @classmethod
    def nhead_divides_d_model_uni(cls, v: int, info: ValidationInfo) -> int:
        dm = info.data.get("d_model")
        if dm is not None and int(dm) % int(v) != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v

    @model_validator(mode="after")
    def coverage_and_target(self) -> UniverseExperimentConfig:
        n = len(self.symbols)
        m = self.min_coverage_symbols
        if m is None:
            m = max(2, n - 1) if n else 2
        if m < 1:
            raise ValueError("min_coverage_symbols must be >= 1")
        if m > n:
            raise ValueError("min_coverage_symbols cannot exceed len(symbols)")
        self.min_coverage_symbols = m

        if self.target_symbol is None:
            self.target_symbol = self.symbols[0]
        else:
            self.target_symbol = str(self.target_symbol).upper()
        return self


def coerce_single_symbol_config(raw: dict[str, Any]) -> dict[str, Any]:
    return SingleSymbolExperimentConfig.model_validate(raw).model_dump(mode="json")


def coerce_universe_config(raw: dict[str, Any]) -> dict[str, Any]:
    return UniverseExperimentConfig.model_validate(raw).model_dump(mode="json")


def coerce_experiment_config(raw: dict[str, Any]) -> dict[str, Any]:
    mode = str(raw.get("experiment_mode", "single_symbol")).lower()
    if mode == "universe":
        return coerce_universe_config(raw)
    return coerce_single_symbol_config(raw)

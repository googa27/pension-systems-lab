"""Validated configuration objects."""

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Numerical and statistical settings for the demographic model."""

    model_config = ConfigDict(frozen=True)

    maximum_age: float = Field(default=105.0, gt=85.0)
    age_nodes: int = Field(default=211, ge=51)
    time_step_years: float = Field(default=1.0 / 12.0, gt=0.0, le=1.0)
    artificial_diffusion: float = Field(default=0.015, ge=0.0)
    fertility_spline_df: int = Field(default=8, ge=5)
    mortality_spline_df: int = Field(default=10, ge=5)
    mortality_monotone_from: float = Field(default=40.0, ge=0.0)
    spline_penalty: float = Field(default=20.0, gt=0.0)
    dynamic_penalty: float = Field(default=30.0, gt=0.0)
    sex_ratio_male: float = Field(default=0.5053, gt=0.45, lt=0.55)

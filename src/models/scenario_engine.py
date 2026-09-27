"""Assumption-based pricing scenario engine.

This module deliberately does not estimate causal price elasticity from the
Airbnb calendar data. Instead, it evaluates candidate prices under explicit,
user-visible price-sensitivity assumptions.

The demand proxy is the model's calendar-unavailability probability. The
"revenue proxy" is therefore a decision-support score:

    revenue_proxy = price * demand_proxy

It is not a claim of causal or realized revenue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.config import load_config

SENSITIVITY_PRESETS: dict[str, float] = {
    "low": 0.60,
    "moderate": 1.00,
    "high": 1.50,
}


@dataclass(frozen=True)
class ScenarioResult:
    """Result for one explicit price-sensitivity assumption."""

    scenario_name: str
    sensitivity: float
    reference_price: float
    base_demand_proxy: float
    recommended_price: float
    demand_proxy_at_recommended: float
    revenue_proxy_at_reference: float
    revenue_proxy_at_recommended: float
    revenue_proxy_change_pct: float
    price_bounds: tuple[float, float]


def _validate_inputs(
    base_demand_proxy: float,
    reference_price: float,
    sensitivity: float,
) -> None:
    if not 0.0 <= base_demand_proxy <= 1.0:
        raise ValueError("base_demand_proxy must be between 0 and 1")
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    if sensitivity < 0:
        raise ValueError("sensitivity must be non-negative")


def scenario_demand_proxy(
    price: float,
    base_demand_proxy: float,
    reference_price: float,
    sensitivity: float,
) -> float:
    """Return demand proxy under an explicit sensitivity assumption.

    The scenario response is exponential around the reference price:

        D(P) = D0 * exp[-s * (P / P0 - 1)]

    where ``s`` is an assumed sensitivity parameter, not an estimated causal
    elasticity. At the reference price, D(P0) = D0. Higher assumed
    sensitivity makes the demand proxy fall faster as price rises.

    The result is clipped to [0, 1] because the underlying demand proxy is a
    probability-like model output.
    """
    _validate_inputs(base_demand_proxy, reference_price, sensitivity)

    if price <= 0:
        return 0.0

    adjusted = base_demand_proxy * np.exp(
        -sensitivity * (price / reference_price - 1.0)
    )
    return float(np.clip(adjusted, 0.0, 1.0))


def revenue_proxy(
    price: float,
    base_demand_proxy: float,
    reference_price: float,
    sensitivity: float,
) -> float:
    """Return price multiplied by the scenario demand proxy."""
    if price <= 0:
        return 0.0

    demand = scenario_demand_proxy(
        price=price,
        base_demand_proxy=base_demand_proxy,
        reference_price=reference_price,
        sensitivity=sensitivity,
    )
    return float(price * demand)


def compute_scenario_curve(
    base_demand_proxy: float,
    reference_price: float,
    sensitivity: float,
    n_points: int = 101,
    floor_price: float | None = None,
    ceiling_price: float | None = None,
    config: dict | None = None,
) -> dict[str, list[float]]:
    """Compute candidate prices and their assumption-based proxy scores."""
    _validate_inputs(base_demand_proxy, reference_price, sensitivity)

    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    if config is None:
        config = load_config()

    opt_cfg = config["optimization"]

    if floor_price is None:
        floor_price = (
            reference_price * opt_cfg["price_floor_multiplier"]
        )
    if ceiling_price is None:
        ceiling_price = (
            reference_price * opt_cfg["price_ceiling_multiplier"]
        )

    floor_price = max(float(floor_price), 1.0)
    ceiling_price = float(ceiling_price)

    if ceiling_price <= floor_price:
        raise ValueError("ceiling_price must be greater than floor_price")

    prices = np.linspace(
        floor_price,
        ceiling_price,
        n_points,
    )

    demand_proxies = [
        scenario_demand_proxy(
            price=float(price),
            base_demand_proxy=base_demand_proxy,
            reference_price=reference_price,
            sensitivity=sensitivity,
        )
        for price in prices
    ]

    revenue_proxies = [
        float(price * demand)
        for price, demand in zip(
            prices,
            demand_proxies,
            strict=True,
        )
    ]

    return {
        "prices": prices.astype(float).tolist(),
        "demand_proxies": demand_proxies,
        "revenue_proxies": revenue_proxies,
    }


def optimize_price_scenario(
    base_demand_proxy: float,
    reference_price: float,
    sensitivity: float,
    scenario_name: str = "custom",
    floor_price: float | None = None,
    ceiling_price: float | None = None,
    n_points: int = 401,
    config: dict | None = None,
) -> ScenarioResult:
    """Choose the best candidate price under one stated assumption.

    This is deterministic grid-based scenario analysis. It does not claim
    that changing price will causally change demand by the returned amount.
    """
    curve = compute_scenario_curve(
        base_demand_proxy=base_demand_proxy,
        reference_price=reference_price,
        sensitivity=sensitivity,
        n_points=n_points,
        floor_price=floor_price,
        ceiling_price=ceiling_price,
        config=config,
    )

    revenues = np.asarray(
        curve["revenue_proxies"],
        dtype=float,
    )
    best_idx = int(np.argmax(revenues))

    recommended_price = float(
        curve["prices"][best_idx]
    )
    demand_at_recommended = float(
        curve["demand_proxies"][best_idx]
    )
    revenue_at_recommended = float(
        curve["revenue_proxies"][best_idx]
    )

    reference_revenue = revenue_proxy(
        price=reference_price,
        base_demand_proxy=base_demand_proxy,
        reference_price=reference_price,
        sensitivity=sensitivity,
    )

    change_pct = (
        (
            revenue_at_recommended
            - reference_revenue
        )
        / reference_revenue
        * 100.0
        if reference_revenue > 0
        else 0.0
    )

    return ScenarioResult(
        scenario_name=scenario_name,
        sensitivity=float(sensitivity),
        reference_price=round(
            float(reference_price),
            2,
        ),
        base_demand_proxy=round(
            float(base_demand_proxy),
            6,
        ),
        recommended_price=round(
            recommended_price,
            2,
        ),
        demand_proxy_at_recommended=round(
            demand_at_recommended,
            6,
        ),
        revenue_proxy_at_reference=round(
            reference_revenue,
            4,
        ),
        revenue_proxy_at_recommended=round(
            revenue_at_recommended,
            4,
        ),
        revenue_proxy_change_pct=round(
            change_pct,
            4,
        ),
        price_bounds=(
            round(float(curve["prices"][0]), 2),
            round(float(curve["prices"][-1]), 2),
        ),
    )


def evaluate_sensitivity_presets(
    base_demand_proxy: float,
    reference_price: float,
    floor_price: float | None = None,
    ceiling_price: float | None = None,
    config: dict | None = None,
) -> dict[str, ScenarioResult]:
    """Evaluate the low/moderate/high sensitivity presets."""
    return {
        name: optimize_price_scenario(
            base_demand_proxy=base_demand_proxy,
            reference_price=reference_price,
            sensitivity=sensitivity,
            scenario_name=name,
            floor_price=floor_price,
            ceiling_price=ceiling_price,
            config=config,
        )
        for name, sensitivity in SENSITIVITY_PRESETS.items()
    }

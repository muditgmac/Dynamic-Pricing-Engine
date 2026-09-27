"""Unit tests for the assumption-based pricing scenario engine."""

import pytest

from src.models.scenario_engine import (
    SENSITIVITY_PRESETS,
    ScenarioResult,
    compute_scenario_curve,
    evaluate_sensitivity_presets,
    optimize_price_scenario,
    revenue_proxy,
    scenario_demand_proxy,
)


class TestScenarioDemandProxy:
    def test_reference_price_preserves_base_proxy(self):
        result = scenario_demand_proxy(
            price=150,
            base_demand_proxy=0.65,
            reference_price=150,
            sensitivity=1.0,
        )
        assert result == pytest.approx(0.65)

    def test_higher_price_reduces_proxy_when_sensitive(self):
        base = scenario_demand_proxy(
            150,
            0.65,
            150,
            1.0,
        )
        higher = scenario_demand_proxy(
            180,
            0.65,
            150,
            1.0,
        )
        assert higher < base

    def test_zero_sensitivity_keeps_proxy_constant(self):
        result = scenario_demand_proxy(
            220,
            0.65,
            150,
            0.0,
        )
        assert result == pytest.approx(0.65)

    def test_proxy_is_bounded(self):
        result = scenario_demand_proxy(
            20,
            0.95,
            150,
            3.0,
        )
        assert 0.0 <= result <= 1.0

    @pytest.mark.parametrize(
        ("base_proxy", "reference_price", "sensitivity"),
        [
            (-0.1, 150, 1.0),
            (1.1, 150, 1.0),
            (0.6, 0, 1.0),
            (0.6, 150, -0.1),
        ],
    )
    def test_invalid_inputs_raise(
        self,
        base_proxy,
        reference_price,
        sensitivity,
    ):
        with pytest.raises(ValueError):
            scenario_demand_proxy(
                150,
                base_proxy,
                reference_price,
                sensitivity,
            )


class TestRevenueProxy:
    def test_equals_price_times_demand_proxy(self):
        price = 165
        demand = scenario_demand_proxy(
            price,
            0.65,
            150,
            1.2,
        )
        value = revenue_proxy(
            price,
            0.65,
            150,
            1.2,
        )
        assert value == pytest.approx(
            price * demand
        )


class TestOptimizePriceScenario:
    def test_returns_scenario_result(self):
        result = optimize_price_scenario(
            0.65,
            150,
            1.0,
        )
        assert isinstance(
            result,
            ScenarioResult,
        )

    def test_recommendation_within_bounds(self):
        result = optimize_price_scenario(
            0.65,
            150,
            1.0,
            floor_price=100,
            ceiling_price=200,
        )
        assert (
            100
            <= result.recommended_price
            <= 200
        )

    def test_recovers_known_exponential_optimum(self):
        # For D = D0 * exp[-s(P/P0 - 1)],
        # unconstrained revenue is maximized at P = P0 / s.
        result = optimize_price_scenario(
            0.50,
            150,
            1.25,
            floor_price=75,
            ceiling_price=225,
            n_points=3001,
        )
        assert result.recommended_price == pytest.approx(
            120.0,
            abs=0.15,
        )

    def test_low_sensitivity_pushes_price_up(self):
        result = optimize_price_scenario(
            0.50,
            150,
            SENSITIVITY_PRESETS["low"],
        )
        assert result.recommended_price >= 150

    def test_moderate_sensitivity_stays_near_reference(self):
        result = optimize_price_scenario(
            0.50,
            150,
            SENSITIVITY_PRESETS["moderate"],
            n_points=3201,
        )
        assert result.recommended_price == pytest.approx(
            150,
            abs=0.2,
        )

    def test_high_sensitivity_pushes_price_down(self):
        result = optimize_price_scenario(
            0.50,
            150,
            SENSITIVITY_PRESETS["high"],
        )
        assert result.recommended_price <= 150

    def test_zero_base_proxy_has_zero_value(self):
        result = optimize_price_scenario(
            0.0,
            150,
            1.0,
        )
        assert (
            result.revenue_proxy_at_reference
            == 0.0
        )
        assert (
            result.revenue_proxy_at_recommended
            == 0.0
        )


class TestScenarioCurve:
    def test_returns_expected_keys(self):
        curve = compute_scenario_curve(
            0.65,
            150,
            1.0,
        )
        assert set(curve) == {
            "prices",
            "demand_proxies",
            "revenue_proxies",
        }

    def test_correct_length(self):
        curve = compute_scenario_curve(
            0.65,
            150,
            1.0,
            n_points=50,
        )
        assert len(curve["prices"]) == 50
        assert len(curve["demand_proxies"]) == 50
        assert len(curve["revenue_proxies"]) == 50

    def test_prices_are_sorted(self):
        curve = compute_scenario_curve(
            0.65,
            150,
            1.0,
        )
        assert curve["prices"] == sorted(
            curve["prices"]
        )


class TestPresetEvaluation:
    def test_all_presets_returned(self):
        results = evaluate_sensitivity_presets(
            0.65,
            150,
        )
        assert set(results) == {
            "low",
            "moderate",
            "high",
        }

    def test_results_preserve_scenario_names(self):
        results = evaluate_sensitivity_presets(
            0.65,
            150,
        )
        for name, result in results.items():
            assert result.scenario_name == name

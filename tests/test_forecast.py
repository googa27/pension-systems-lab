import numpy as np

from chile_demographic_pde.diagnostics.forecast import rolling_origin_backtest


def test_rolling_origin_backtest_uses_only_past_data() -> None:
    y = np.array([100, 102, 105, 104, 108, 110, 111, 115, 114], dtype=float)
    backtest = rolling_origin_backtest(y, minimum_training=5, penalty=10.0)
    assert len(backtest) == 4
    assert list(backtest.columns) == ["origin", "actual", "forecast", "error"]
    assert backtest["origin"].iloc[0] == 5

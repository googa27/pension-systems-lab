import numpy as np

from chile_demographic_pde.data.observation import AgeBin
from chile_demographic_pde.rates.poisson_spline import BinnedPoissonSplineRate


def test_poisson_spline_produces_positive_finite_rates() -> None:
    bins = [AgeBin(float(a), float(a + 5), f"{a}-{a + 4}") for a in range(15, 50, 5)]
    exposure = np.array([600_000, 625_000, 690_000, 770_000, 715_000, 655_000, 602_000])
    counts = np.array([420, 1922, 3224, 3614, 2625, 659, 36])
    model = BinnedPoissonSplineRate(df=7, penalty=10.0).fit(bins, counts, exposure)
    ages = np.linspace(15.0, 50.0, 100)
    rates = model.rate(ages)
    assert np.all(np.isfinite(rates))
    assert np.all(rates > 0.0)
    fitted = model.expected_bin_counts(bins, exposure)
    assert fitted.sum() > 0.95 * counts.sum()
    assert fitted.sum() < 1.05 * counts.sum()


def test_monotone_constraint_prevents_old_age_hazard_decline() -> None:
    bins = [AgeBin(float(a), float(a + 10), f"{a}-{a + 9}") for a in range(0, 100, 10)]
    exposure = np.array(
        [1_000_000, 900_000, 800_000, 700_000, 600_000, 500_000, 400_000, 300_000, 200_000, 100_000]
    )
    counts = np.array([100, 80, 100, 160, 300, 650, 1_300, 2_600, 4_000, 3_500])
    fitted = BinnedPoissonSplineRate(df=9, penalty=5.0, monotone_from=40.0).fit(
        bins, counts, exposure
    )
    ages = np.linspace(40.0, 100.0, 121)
    assert np.min(np.diff(np.log(fitted.rate(ages)))) > -1e-7


def test_poisson_spline_resolves_an_open_tail_only_at_an_explicit_endpoint() -> None:
    bins = [AgeBin(0.0, 5.0, "0-4"), AgeBin(5.0, None, "5+")]
    exposure = np.array([1_000.0, 500.0])
    counts = np.array([10.0, 20.0])

    fitted = BinnedPoissonSplineRate(df=5, penalty=1.0).fit(
        bins,
        counts,
        exposure,
        open_bin_end=10.0,
    )

    assert fitted.upper == 10.0
    expected = fitted.expected_bin_counts(
        bins,
        exposure,
        open_bin_end=10.0,
    )
    assert expected.shape == counts.shape
    assert np.all(np.isfinite(expected))

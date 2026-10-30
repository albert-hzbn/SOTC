import numpy as np
import pytest

from sotc import design_theory as th

SIG2, W2, C, D, S2 = 1.0, 4.0, 0.6, 0.2, 0.5


def test_inverse_moment_factors():
    assert th.inverse_moment_factor(10, "canonical") == pytest.approx(10 / 8)
    assert th.inverse_moment_factor(16, "paired") == pytest.approx(8 / 6)
    assert th.inverse_moment_factor(16, "fixed") == 1.0
    assert th.inverse_moment_factor(4, "paired") == float("inf")


@pytest.mark.parametrize("design", ["canonical", "paired", "fixed"])
@pytest.mark.parametrize("k_calls", [8, 16])
def test_bias_and_variance_match_monte_carlo(design, k_calls):
    rng = np.random.default_rng(1)
    x = th.simulate(k_calls, design, SIG2, W2, C, D, S2, 200000, rng)
    var = (th.noise_variance(k_calls, design, SIG2, S2)
           + th.cubic_variance(k_calls, design, SIG2, C)
           + th.quartic_variance(k_calls, design, SIG2, D))
    bias = th.quartic_bias(k_calls, design, SIG2, D)
    assert x.var() == pytest.approx(var, rel=0.03)
    assert x.mean() - (W2 + 3 * D * SIG2) == pytest.approx(bias, abs=0.006)


def test_fixed_amplitudes_remove_cubic_and_amplitude_noise():
    assert th.cubic_variance(8, "fixed", SIG2, C) == 0.0
    assert th.quartic_variance(8, "fixed", SIG2, D) == 0.0
    assert th.cubic_variance(8, "paired", SIG2, C) == 0.0
    assert th.cubic_variance(8, "canonical", SIG2, C) > 0.0


def test_crossover_exists_and_grows_with_noise():
    k_low = th.crossover_calls(SIG2, C, D, S2)
    k_high = th.crossover_calls(SIG2, C, D, 4 * S2)
    assert 6 <= k_low < k_high
    assert th.mse(k_low - 2, "fixed", SIG2, C, D, S2) < th.mse(k_low - 2, "paired", SIG2, C, D, S2)


def test_inverse_pairs_repeat_quadratic_features():
    rng = np.random.default_rng(2)
    half = rng.normal(size=(3, 5))
    assert th.pair_feature_rank(np.concatenate([half, -half])) == 3
    assert th.pair_feature_rank(rng.normal(size=(6, 5))) == 6

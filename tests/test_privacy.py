"""Tests for the differential-privacy step.

`apply_differential_privacy` is plain pandas/numpy, so the part of this project
that a reviewer can actually check without the dataset is checkable here.
"""
import numpy as np
import pandas as pd
import pytest

from src.D_privacy import apply_differential_privacy


def frame(rows=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Flow Duration": rng.integers(1, 10_000, rows).astype(float),
            "Total Fwd Packets": rng.integers(1, 200, rows).astype(float),
            "Label": ["BENIGN"] * rows,
        }
    )


def test_only_numeric_columns_come_back():
    out = apply_differential_privacy(frame())
    assert list(out.columns) == ["Flow Duration", "Total Fwd Packets"]


def test_the_shape_survives_the_round_trip():
    df = frame(30)
    assert apply_differential_privacy(df, epsilon=1.0).shape == (30, 2)


def test_noise_actually_changes_the_values():
    df = frame()
    out = apply_differential_privacy(df, epsilon=1.0)
    assert not np.allclose(out.to_numpy(), df[out.columns].to_numpy())


def test_a_larger_epsilon_means_less_noise():
    df = frame(200)
    tight = apply_differential_privacy(df, epsilon=50.0) - df[["Flow Duration", "Total Fwd Packets"]]
    loose = apply_differential_privacy(df, epsilon=0.5) - df[["Flow Duration", "Total Fwd Packets"]]
    assert tight.abs().to_numpy().std() < loose.abs().to_numpy().std()


def test_gaussian_needs_a_delta_and_also_perturbs():
    df = frame(20)
    out = apply_differential_privacy(df, epsilon=1.0, method="gaussian", delta=1e-5)
    assert out.shape == (20, 2)
    assert not np.allclose(out.to_numpy(), df[out.columns].to_numpy())


def test_an_unknown_mechanism_is_rejected():
    with pytest.raises(ValueError):
        apply_differential_privacy(frame(), method="exponential")

import math
from importlib.metadata import PackageNotFoundError, version

import numpy
import pandas
import pytest

import averagers


def test_version_is_exposed():
    assert averagers.__version__ == "0.1.0"
    try:
        installed_version = version("averagers")
    except PackageNotFoundError:
        installed_version = averagers.__version__
    assert installed_version == averagers.__version__


def test_dh2006_accepts_day_fractions_and_legacy_hours():
    base = pandas.DataFrame(
        {
            "Min": [10.0],
            "Max": [20.0],
            "Min_next": [12.0],
            "Sunset_nondimensional": [0.75],
        }
    )
    params = {"CD": 0.5, "CN": 0.25}

    from_fraction = averagers.get_average_temperature(base, params, method="DH2006")
    from_hours = averagers.get_average_temperature(
        base.assign(Sunset_nondimensional=18.0),
        params,
        method="DH2006",
    )

    assert numpy.allclose(from_fraction, [14.75])
    assert numpy.allclose(from_hours, from_fraction)


def test_simple_average_temperature():
    df = pandas.DataFrame({"Min": [10.0, 12.0], "Max": [20.0, 24.0]})

    simple_average = averagers.get_simple_average_temperature(df)

    assert numpy.allclose(simple_average, [15.0, 18.0])


def test_plot_temperature_estimates_writes_file(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    df = pandas.DataFrame(
        {
            "Date": pandas.date_range("2020-06-01", periods=2),
            "Min": [10.0, 12.0],
            "Max": [20.0, 24.0],
            "Ave_sim": [14.5, 18.5],
        }
    )
    output = tmp_path / "plot.png"

    fig, ax = averagers.plot_temperature_estimates(
        df,
        output=output,
        minmax_linewidth=0.8,
        simple_average_linewidth=0.9,
        estimated_linewidth=1.1,
    )

    labels = [line.get_label() for line in ax.get_lines()]
    assert "Estimated mean" in labels
    assert "Simple average of min/max" in labels
    assert ax.get_lines()[-1].get_linewidth() == 1.1
    assert len(ax.collections) == 0
    assert output.exists()
    assert output.stat().st_size > 0
    matplotlib.pyplot.close(fig)


def test_estimation_error_metrics_and_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    df = pandas.DataFrame(
        {
            "Ave": [10.0, 12.0, 14.0],
            "Ave_simple": [11.0, 11.5, 13.0],
            "Ave_est": [10.5, 12.0, 13.5],
        }
    )

    metrics = averagers.get_estimation_error_metrics(
        df,
        estimate_columns=["Ave_simple", "Ave_est"],
        labels={"Ave_simple": "Simple mean", "Ave_est": "Estimated mean"},
    )

    assert list(metrics["estimate"]) == ["Simple mean", "Estimated mean"]
    assert numpy.isclose(metrics.loc[metrics["column"] == "Ave_est", "RMSE"].iloc[0], (0.5**2 + 0 + 0.5**2) ** 0.5 / 3**0.5)

    output = tmp_path / "comparison.png"
    fig, axes, plotted_metrics = averagers.plot_estimation_error_comparison(
        df,
        estimate_columns=["Ave_simple", "Ave_est"],
        labels={"Ave_simple": "Simple mean", "Ave_est": "Estimated mean"},
        output=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert plotted_metrics.shape[0] == 2
    assert len(axes) == 2
    matplotlib.pyplot.close(fig)


def test_kf_uses_c3_for_post_sunset_temperature():
    df = pandas.DataFrame(
        {
            "Min": [10.0],
            "Max": [22.0],
            "Max_prev": [18.0],
            "Min_next": [12.0],
            "Sunrise_nondimensional": [0.25],
            "Sunset_nondimensional": [0.75],
        }
    )

    ave = averagers.get_average_temperature(
        df,
        params={"C1": 0.0, "C2": 1.0, "C3": 0.5},
        method="KF",
    )

    assert numpy.allclose(ave, [17.75])


def test_validation_errors_are_explicit():
    df = pandas.DataFrame(
        {
            "Min": [10.0],
            "Max": [20.0],
            "Min_next": [12.0],
            "Ave": [15.0],
            "Sunset_nondimensional": [0.75],
        }
    )

    with pytest.raises(ValueError, match="method"):
        averagers.get_average_temperature(df, {"CD": 0.5, "CN": 0.5}, method="bad")

    with pytest.raises(ValueError, match="max_step"):
        averagers.get_params(df, method="DH2006", max_step=0)

    with pytest.raises(ValueError, match="at least one row"):
        averagers.get_params(df.iloc[0:0], method="DH2006")


def test_get_temp_dif_matches_dh2006_average_minus_observed():
    assert math.isclose(
        averagers.get_temp_dif(
            CD=0.5,
            CN=0.25,
            min0=10.0,
            max0=20.0,
            ave0=14.0,
            min1=12.0,
            sunset_nondimensional=18.0,
        ),
        0.75,
    )


def test_get_month_params_wraps_windows_across_year_boundary():
    df = pandas.DataFrame(
        {
            "Month": [12, 1, 2],
            "Min": [10.0, 11.0, 12.0],
            "Max": [20.0, 21.0, 22.0],
            "Min_next": [11.0, 12.0, 13.0],
            "Ave": [15.0, 16.0, 17.0],
            "Sunset_nondimensional": [0.75, 0.75, 0.75],
        }
    )

    params = averagers.get_month_params(
        df,
        param_min=0,
        param_max=1,
        max_step=1,
        method="DH2006",
        window_size=1,
    )

    assert set(params) == {"1", "2", "12"}


def test_get_photoperiod_returns_local_day_fractions():
    photoperiod = averagers.get_photoperiod(
        start_date="2020-06-01",
        end_date="2020-06-01",
        lat=35.681,
        lon=139.767,
        timezone=9,
    )
    row = photoperiod.iloc[0]

    assert 0 <= row["Sunrise_nondimensional"] < row["Sunset_nondimensional"] <= 1
    assert 13 < row["Daytime"] < 15


def test_get_photoperiod_handles_polar_day():
    photoperiod = averagers.get_photoperiod(
        start_date="2020-06-01",
        end_date="2020-06-01",
        lat=89,
        lon=0,
        timezone=0,
    )
    row = photoperiod.iloc[0]

    assert row["Daytime"] == 24.0
    assert pandas.isna(row["Sunrise"])
    assert pandas.isna(row["Sunset"])

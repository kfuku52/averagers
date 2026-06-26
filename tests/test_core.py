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
    assert [tick.get_text() for tick in axes[1].get_xticklabels()] == ["RMSE"]
    matplotlib.pyplot.close(fig)


def test_estimation_error_plot_can_hide_scatter(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    df = pandas.DataFrame(
        {
            "Ave": [10.0, 12.0, 14.0],
            "Ave_simple": [11.0, 11.5, 13.0],
            "Ave_est": [10.5, 12.0, 13.5],
        }
    )
    output = tmp_path / "metrics_only.png"

    fig, axes, metrics = averagers.plot_estimation_error_comparison(
        df,
        estimate_columns=["Ave_simple", "Ave_est"],
        output=output,
        show_scatter=False,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(axes) == 1
    assert axes[0].get_title() == "Error metrics"
    assert [tick.get_text() for tick in axes[0].get_xticklabels()] == ["RMSE"]
    assert metrics.shape[0] == 2
    matplotlib.pyplot.close(fig)


def test_plot_estimation_metric_by_setting_groups_methods(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    metrics = pandas.DataFrame(
        {
            "setting": ["simple mean", "yearly", "yearly", "ws0", "ws0"],
            "method": ["Simple mean", "DH2006", "Diurnal3", "DH2006", "Diurnal3"],
            "RMSE": [0.53, 0.5, 0.48, 0.46, 0.47],
        }
    )
    output = tmp_path / "setting_plot.png"

    fig, ax = averagers.plot_estimation_metric_by_setting(
        metrics,
        setting_order=["simple mean", "yearly", "ws0"],
        method_order=["Simple mean", "DH2006", "Diurnal3"],
        output=output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "simple mean",
        "yearly",
        "ws0",
    ]
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Simple mean",
        "DH2006",
        "Diurnal3",
    ]
    matplotlib.pyplot.close(fig)


def test_diurnal3_uses_c3_for_post_sunset_temperature():
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
        method="Diurnal3",
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


def test_least_squares_recovers_dh2006_params():
    df = pandas.DataFrame(
        {
            "Min": [9.0, 11.0, 8.0, 13.0],
            "Max": [19.0, 24.0, 22.0, 27.0],
            "Min_next": [10.0, 9.0, 12.0, 14.0],
            "Sunset_nondimensional": [0.68, 0.72, 0.65, 0.7],
        }
    )
    expected = {"CD": 0.55, "CN": 0.22}
    df["Ave"] = averagers.get_average_temperature(df, expected, method="DH2006")

    params = averagers.get_params(
        df,
        method="DH2006",
        param_min=0,
        param_max=1,
        optimizer="least_squares",
    )

    assert numpy.isclose(params["CD"], expected["CD"])
    assert numpy.isclose(params["CN"], expected["CN"])
    assert numpy.isclose(params["variance"], 0)


def test_least_squares_recovers_diurnal3_params():
    df = pandas.DataFrame(
        {
            "Min": [9.0, 11.0, 8.0, 13.0],
            "Max": [19.0, 24.0, 22.0, 27.0],
            "Max_prev": [18.0, 19.0, 24.0, 22.0],
            "Min_next": [10.0, 9.0, 12.0, 14.0],
            "Sunrise_nondimensional": [0.25, 0.23, 0.28, 0.24],
            "Sunset_nondimensional": [0.68, 0.72, 0.65, 0.7],
        }
    )
    expected = {"C1": 0.2, "C2": 0.6, "C3": 0.15}
    df["Ave"] = averagers.get_average_temperature(df, expected, method="Diurnal3")

    params = averagers.get_params(
        df,
        method="Diurnal3",
        param_min=0,
        param_max=1,
        optimizer="least_squares",
    )

    assert numpy.isclose(params["C1"], expected["C1"])
    assert numpy.isclose(params["C2"], expected["C2"])
    assert numpy.isclose(params["C3"], expected["C3"])
    assert numpy.isclose(params["variance"], 0)


def test_linear_temporal_params_and_cross_validation():
    rows = []
    for year in [2020, 2021, 2022]:
        for month in range(1, 13):
            sunrise = 0.2 + (month % 4) * 0.015
            sunset = 0.62 + (month % 5) * 0.025
            min_value = 4.0 + month * 0.7 + (year - 2020) * 0.1
            max_value = min_value + 8.0 + (month % 3)
            rows.append(
                {
                    "Date": pandas.Timestamp(year=year, month=month, day=15),
                    "Year": year,
                    "Month": month,
                    "Min": min_value,
                    "Max": max_value,
                    "Min_next": min_value + 0.3,
                    "Max_prev": max_value - 0.4,
                    "Sunrise_nondimensional": sunrise,
                    "Sunset_nondimensional": sunset,
                    "Daytime": (sunset - sunrise) * 24,
                }
            )
    df = pandas.DataFrame(rows)
    expected = {
        "feature_set": "temporal",
        "coefficients": {
            "Intercept": 1.0,
            "Min": 0.25,
            "Max": 0.45,
            "Min_next": 0.15,
            "Max_prev": 0.12,
            "Sunrise": 0.3,
            "Sunset": -0.2,
        },
    }
    df["Ave"] = averagers.get_linear_average_temperature(df, expected)

    params = averagers.get_linear_params(df, feature_set="temporal")
    predicted = averagers.get_linear_average_temperature(df, params)

    assert numpy.isclose(params["variance"], 0)
    assert numpy.allclose(predicted, df["Ave"])

    monthly_params = averagers.get_month_linear_params(
        df,
        feature_set="temporal",
        window_size=1,
        ridge=0.1,
    )
    monthly_predicted = averagers.get_month_linear_average_temperature(
        df.copy(),
        monthly_params,
    )
    assert monthly_predicted.notna().all()

    monthly_linear_specs = [
        {
            "name": f"Monthly linear temporal ws{window_size}",
            "column": f"Ave_est_monthly_linear_ws{window_size}",
            "kind": "monthly_linear",
            "method": "Monthly linear",
            "setting": f"ws{window_size}",
            "feature_set": "temporal",
            "window_size": window_size,
            "ridge": 0.1,
        }
        for window_size in range(4)
    ]
    auto_candidates = [
        {
            "name": "DH2006 monthly ws0",
            "column": "_auto_dh2006_monthly_ws0",
            "kind": "monthly",
            "method": "DH2006",
            "setting": "ws0",
            "window_size": 0,
            "optimizer": "least_squares",
        },
        {
            "name": "Diurnal3 monthly ws0",
            "column": "_auto_diurnal3_monthly_ws0",
            "kind": "monthly",
            "method": "Diurnal3",
            "setting": "ws0",
            "window_size": 0,
            "optimizer": "least_squares",
        },
        {
            "name": "Monthly linear temporal ws0",
            "column": "_auto_monthly_linear_ws0",
            "kind": "monthly_linear",
            "method": "Monthly linear",
            "setting": "ws0",
            "feature_set": "temporal",
            "window_size": 0,
            "ridge": 0.1,
        },
        {
            "name": "Linear harmonic",
            "column": "_auto_linear_harmonic",
            "kind": "linear",
            "method": "Harmonic",
            "setting": "harmonic",
            "feature_set": "harmonic",
        },
    ]
    default_candidates = averagers.get_default_auto_candidates(
        windows=[0],
        smoothed_windows=[1],
    )
    assert any(candidate["method"] == "Diurnal3" for candidate in default_candidates)

    predictions, metrics = averagers.cross_validate_estimates(
        df,
        specs=[
            *monthly_linear_specs,
            {
                "name": "Linear harmonic",
                "column": "Ave_est_harmonic",
                "kind": "linear",
                "method": "Harmonic",
                "setting": "harmonic",
                "feature_set": "harmonic",
            },
            {
                "name": "Auto best",
                "column": "Ave_est_auto",
                "kind": "auto",
                "method": "Auto",
                "setting": "auto",
                "candidates": auto_candidates,
                "selection_scope": "global",
            },
        ],
    )

    expected_columns = {
        *(f"Ave_est_monthly_linear_ws{window_size}" for window_size in range(4)),
        "Ave_est_harmonic",
        "Ave_est_auto",
    }
    assert expected_columns.issubset(predictions.columns)
    assert list(metrics["feature_set"].head(4)) == ["temporal"] * 4
    assert metrics.loc[metrics["estimate"] == "Auto best", "setting"].item() == "auto"
    assert "all" in predictions.attrs["selected_candidates"]["Ave_est_auto"]


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


def test_smooth_month_params_wraps_across_year_boundary():
    params = {
        "12": {"CD": 0.2, "CN": 0.4},
        "1": {"CD": 0.5, "CN": 0.7},
        "2": {"CD": 0.8, "CN": 1.0},
    }

    smoothed = averagers.smooth_month_params(params, smooth_window=1, method="DH2006")

    assert numpy.isclose(smoothed["1"]["CD"], 0.5)
    assert numpy.isclose(smoothed["1"]["CN"], 0.7)


def test_cross_validate_estimates_and_select_month_window():
    rows = []
    for year in [2020, 2021, 2022]:
        for month in range(1, 13):
            rows.append(
                {
                    "Year": year,
                    "Month": month,
                    "Min": 5.0 + month / 2,
                    "Max": 15.0 + month / 2 + (year - 2020) * 0.2,
                    "Min_next": 5.5 + month / 2,
                    "Sunset_nondimensional": 0.6 + (month % 3) * 0.03,
                }
            )
    df = pandas.DataFrame(rows)
    df["Ave"] = averagers.get_average_temperature(
        df,
        {"CD": 0.5, "CN": 0.25},
        method="DH2006",
    )

    predictions, metrics = averagers.cross_validate_estimates(
        df,
        specs=[
            {"name": "Simple mean", "column": "Ave_simple", "kind": "simple"},
            {
                "name": "DH2006 monthly ws1",
                "column": "Ave_est_monthly_ws1",
                "kind": "monthly",
                "method": "DH2006",
                "window_size": 1,
                "optimizer": "least_squares",
            },
            {
                "name": "DH2006 smoothed ws1",
                "column": "Ave_est_seasonal",
                "kind": "smoothed",
                "method": "DH2006",
                "window_size": 1,
                "smooth_window": 1,
                "optimizer": "least_squares",
            },
        ],
    )

    assert {"Ave_simple", "Ave_est_monthly_ws1", "Ave_est_seasonal"}.issubset(
        predictions.columns
    )
    assert list(metrics["estimate"]) == [
        "Simple mean",
        "DH2006 monthly ws1",
        "DH2006 smoothed ws1",
    ]

    selection = averagers.select_month_window(
        df,
        windows=[0, 1],
        method="DH2006",
        optimizer="least_squares",
    )

    assert selection["best_window"] in {0, 1}
    assert selection["metrics"].shape[0] == 2

    auto_predictions, auto_metrics = averagers.cross_validate_estimates(
        df,
        specs=[
            {
                "name": "DH2006 auto",
                "column": "Ave_est_monthly_auto",
                "kind": "auto",
                "method": "DH2006",
                "setting": "auto",
                "windows": [0, 1],
                "smoothed_windows": [1],
                "optimizer": "least_squares",
            }
        ],
    )

    assert "Ave_est_monthly_auto" in auto_predictions.columns
    assert auto_predictions["Ave_est_monthly_auto"].notna().any()
    assert auto_metrics.loc[0, "estimate"] == "DH2006 auto"
    assert auto_metrics.loc[0, "setting"] == "auto"
    assert set(auto_predictions.attrs["selected_windows"]["Ave_est_monthly_auto"]) == {
        2020,
        2021,
        2022,
    }
    assert set(auto_predictions.attrs["selected_settings"]["Ave_est_monthly_auto"]) == {
        2020,
        2021,
        2022,
    }


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

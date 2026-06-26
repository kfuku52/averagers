from pathlib import Path
from time import perf_counter

import pandas as pd

import averagers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "window_size_comparison.png"
WINDOW_SIZES = [0, 1, 2, 3]
SMOOTHED_WINDOW_SIZES = [1, 2, 3]
METHODS = ["DH2006", "Diurnal3"]
PLOT_METHODS = ["Simple mean", *METHODS, "Monthly linear", "Harmonic", "Auto"]
LINEAR_RIDGE = 0.25


def build_base_data():
    start_date = "2020-01-01"
    end_date = "2022-12-31"
    lat = 35.681
    lon = 139.767
    timezone = 9

    weather = averagers.fetch_power_daily_temperature(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        add_max_prev=True,
    )
    weather["Date"] = pd.to_datetime(weather["Date"])
    weather["Year"] = weather["Date"].dt.year

    photoperiod = averagers.get_photoperiod(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )
    weather = weather.join(
        photoperiod[["Sunrise_nondimensional", "Sunset_nondimensional", "Daytime"]]
    )
    return weather


def auto_candidates():
    return averagers.get_default_auto_candidates(
        windows=WINDOW_SIZES,
        smoothed_windows=SMOOTHED_WINDOW_SIZES,
        methods=METHODS,
        linear_ridge=LINEAR_RIDGE,
    )


def comparison_specs():
    specs = [
        {
            "name": "Simple mean",
            "column": "Ave_simple",
            "kind": "simple",
            "method": "Simple mean",
            "setting": "simple mean",
        },
    ]
    specs.extend(
        {
            "name": f"{method} yearly",
            "column": f"Ave_est_{method.lower()}_yearly",
            "kind": "yearly",
            "method": method,
            "setting": "yearly",
            "optimizer": "least_squares",
        }
        for method in METHODS
    )
    for window_size in WINDOW_SIZES:
        specs.extend(
            {
                "name": f"{method} monthly ws{window_size}",
                "column": f"Ave_est_{method.lower()}_monthly_ws{window_size}",
                "kind": "monthly",
                "method": method,
                "setting": f"ws{window_size}",
                "window_size": window_size,
                "optimizer": "least_squares",
            }
            for method in METHODS
        )
    for window_size in SMOOTHED_WINDOW_SIZES:
        specs.extend(
            {
                "name": f"{method} smoothed ws{window_size}",
                "column": f"Ave_est_{method.lower()}_smoothed_ws{window_size}",
                "kind": "smoothed",
                "method": method,
                "setting": f"smoothed ws{window_size}",
                "window_size": window_size,
                "smooth_window": window_size,
                "optimizer": "least_squares",
            }
            for method in METHODS
        )
    specs.extend(
        {
            "name": f"Monthly linear temporal ws{window_size}",
            "column": f"Ave_est_monthly_linear_temporal_ws{window_size}",
            "kind": "monthly_linear",
            "method": "Monthly linear",
            "setting": f"ws{window_size}",
            "feature_set": "temporal",
            "window_size": window_size,
            "ridge": LINEAR_RIDGE,
        }
        for window_size in WINDOW_SIZES
    )
    specs.extend(
        [
            {
                "name": "Linear harmonic",
                "column": "Ave_est_linear_harmonic",
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
                "candidates": auto_candidates(),
                "selection_scope": "global",
            },
        ]
    )
    return specs


def add_cross_validated_estimates(weather):
    started = perf_counter()
    weather, metrics = averagers.cross_validate_estimates(
        weather,
        specs=comparison_specs(),
    )
    selections = {
        method: averagers.select_month_window(
            weather,
            windows=WINDOW_SIZES,
            method=method,
            optimizer="least_squares",
        )
        for method in METHODS
    }
    weather.attrs["fit_seconds"] = perf_counter() - started
    weather.attrs["metrics"] = metrics
    weather.attrs["best_windows"] = {
        method: selection["best_window"] for method, selection in selections.items()
    }
    return weather


def main():
    weather = add_cross_validated_estimates(build_base_data())
    metrics = weather.attrs["metrics"]
    plot_metrics = metrics.dropna(subset=["method", "setting"])

    _fig, _ax = averagers.plot_estimation_metric_by_setting(
        plot_metrics,
        setting_order=[
            "simple mean",
            "yearly",
            *[f"ws{window_size}" for window_size in WINDOW_SIZES],
            *[f"smoothed ws{window_size}" for window_size in SMOOTHED_WINDOW_SIZES],
            "harmonic",
            "auto",
        ],
        method_order=PLOT_METHODS,
        output=OUTPUT,
        title="Parameter-setting RMSE, Tokyo 2020-2022",
        xtick_rotation=25,
    )
    print(metrics[["estimate", "n", "RMSE"]].round(3).to_string(index=False))
    for method, best_window in weather.attrs["best_windows"].items():
        print(f"Best {method} monthly window by leave-one-year-out RMSE: {best_window}")
    selected_candidates = weather.attrs.get("selected_candidates", {}).get("Ave_est_auto", {})
    if selected_candidates:
        print(f"Auto-selected candidate: {selected_candidates.get('all')}")
    print(f"Fitted leave-one-year-out parameter options in {weather.attrs['fit_seconds']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

from pathlib import Path
from time import perf_counter

import pandas as pd

import averagers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "window_size_comparison.png"
WINDOW_SIZES = [0, 1, 2, 3]


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
        photoperiod[["Sunrise_nondimensional", "Sunset_nondimensional"]]
    )
    return weather


def comparison_specs():
    specs = [
        {"name": "Simple mean", "column": "Ave_simple", "kind": "simple"},
        {
            "name": "DH2006 yearly",
            "column": "Ave_est_dh2006_yearly",
            "kind": "yearly",
            "method": "DH2006",
            "optimizer": "least_squares",
        },
    ]
    specs.extend(
        {
            "name": f"DH2006 monthly ws{window_size}",
            "column": f"Ave_est_dh2006_monthly_ws{window_size}",
            "kind": "monthly",
            "method": "DH2006",
            "window_size": window_size,
            "optimizer": "least_squares",
        }
        for window_size in WINDOW_SIZES
    )
    specs.extend(
        [
            {
                "name": "DH2006 seasonal smooth",
                "column": "Ave_est_dh2006_seasonal",
                "kind": "cyclic",
                "method": "DH2006",
                "window_size": 1,
                "smooth_window": 1,
                "optimizer": "least_squares",
            },
            {
                "name": "KF yearly",
                "column": "Ave_est_kf_yearly",
                "kind": "yearly",
                "method": "KF",
                "optimizer": "least_squares",
            },
            {
                "name": "KF monthly ws1",
                "column": "Ave_est_kf_monthly_ws1",
                "kind": "monthly",
                "method": "KF",
                "window_size": 1,
                "optimizer": "least_squares",
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
    selection = averagers.select_month_window(
        weather,
        windows=WINDOW_SIZES,
        method="DH2006",
        optimizer="least_squares",
    )
    weather.attrs["fit_seconds"] = perf_counter() - started
    weather.attrs["metrics"] = metrics
    weather.attrs["best_window"] = selection["best_window"]
    return weather


def main():
    weather = add_cross_validated_estimates(build_base_data())
    specs = comparison_specs()
    estimate_columns = [spec["column"] for spec in specs]
    labels = {spec["column"]: spec["name"] for spec in specs}

    _fig, _axes, metrics = averagers.plot_estimation_error_comparison(
        weather,
        estimate_columns=estimate_columns,
        labels=labels,
        output=OUTPUT,
        title="Parameter and method error, Tokyo 2020-2022",
        scatter_alpha=0.12,
        scatter_size=7,
        legend_outside=True,
    )
    print(metrics[["estimate", "n", "RMSE", "Min error", "Max error"]].round(3).to_string(index=False))
    print(
        "Best DH2006 monthly window by leave-one-year-out RMSE: "
        f"{weather.attrs['best_window']}"
    )
    print(f"Fitted leave-one-year-out parameter options in {weather.attrs['fit_seconds']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

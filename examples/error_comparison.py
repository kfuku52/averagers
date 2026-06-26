from pathlib import Path
from time import perf_counter

import pandas as pd

import averagers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "error_comparison.png"
WINDOW_SIZES = [0, 1, 2, 3]
SMOOTHED_WINDOW_SIZES = [1, 2, 3]
METHODS = ["DH2006", "Diurnal3"]
LINEAR_RIDGE = 0.25


def auto_candidates():
    return averagers.get_default_auto_candidates(
        windows=WINDOW_SIZES,
        smoothed_windows=SMOOTHED_WINDOW_SIZES,
        methods=METHODS,
        linear_ridge=LINEAR_RIDGE,
    )


def build_comparison_data():
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
    weather["Year"] = weather["Date"].dt.year

    started = perf_counter()
    weather, metrics = averagers.cross_validate_estimates(
        weather,
        specs=[
            {"name": "Simple mean", "column": "Ave_simple", "kind": "simple"},
            {
                "name": "Auto best",
                "column": "Ave_est_auto",
                "kind": "auto",
                "method": "Auto",
                "setting": "auto",
                "candidates": auto_candidates(),
                "selection_scope": "global",
            },
        ],
    )
    weather.attrs["fit_seconds"] = perf_counter() - started
    weather.attrs["metrics"] = metrics
    return weather


def main():
    weather = build_comparison_data()
    _fig, _axes, metrics = averagers.plot_estimation_error_comparison(
        weather,
        estimate_columns=["Ave_simple", "Ave_est_auto"],
        labels={
            "Ave_simple": "Simple mean",
            "Ave_est_auto": "Auto best",
        },
        output=OUTPUT,
        title="Daily mean temperature error, Tokyo 2020-2022",
    )
    print(metrics[["estimate", "n", "RMSE"]].round(3).to_string(index=False))
    selected = weather.attrs.get("selected_candidates", {}).get("Ave_est_auto", {}).get("all")
    print(f"Auto-selected candidate: {selected}")
    print(f"Fitted leave-one-year-out options in {weather.attrs['fit_seconds']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

from pathlib import Path
from time import perf_counter

import pandas as pd

import averagers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "example_plot.png"


def build_example_data():
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
                "candidates": averagers.get_default_auto_candidates(),
                "selection_scope": "global",
            },
        ],
    )
    fit_seconds = perf_counter() - started
    weather.attrs["metrics"] = metrics
    weather.attrs["fit_seconds"] = fit_seconds
    return weather


def main():
    weather = build_example_data()

    averagers.plot_temperature_estimates(
        weather,
        estimated_column="Ave_est_auto",
        simple_average_column="Ave_simple",
        output=OUTPUT,
        title="Auto-estimated daily mean temperature, Tokyo 2020-2022",
    )
    selected = weather.attrs["selected_candidates"]["Ave_est_auto"]["all"]
    rmse = weather.attrs["metrics"].loc[
        weather.attrs["metrics"]["estimate"] == "Auto best",
        "RMSE",
    ].item()
    print(
        f"Auto-selected candidate: {selected}; "
        f"RMSE={rmse:.3f}; fit={weather.attrs['fit_seconds']:.3f} s"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

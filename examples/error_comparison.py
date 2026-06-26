from pathlib import Path
from time import perf_counter

import pandas as pd

import averagers


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "error_comparison.png"


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
    )
    weather["Date"] = pd.to_datetime(weather["Date"])

    photoperiod = averagers.get_photoperiod(
        start_date=start_date,
        end_date=end_date,
        lat=lat,
        lon=lon,
        timezone=timezone,
    )
    weather = weather.join(photoperiod[["Sunset_nondimensional"]])
    weather["Ave_simple"] = averagers.get_simple_average_temperature(weather)
    weather["Ave_est_cv"] = pd.NA
    weather["Year"] = weather["Date"].dt.year

    started = perf_counter()
    fitted_params = {}
    for year in sorted(weather["Year"].unique()):
        train = weather.loc[weather["Year"] != year, :].dropna(
            subset=["Ave", "Min", "Max", "Min_next", "Sunset_nondimensional"]
        )
        test_mask = weather["Year"] == year
        params = averagers.get_params(train, method="DH2006")
        fitted_params[int(year)] = params
        weather.loc[test_mask, "Ave_est_cv"] = averagers.get_average_temperature(
            weather.loc[test_mask, :],
            params=params,
            method="DH2006",
        )

    weather.attrs["fit_seconds"] = perf_counter() - started
    weather.attrs["fitted_params"] = fitted_params
    return weather


def main():
    weather = build_comparison_data()
    _fig, _axes, metrics = averagers.plot_estimation_error_comparison(
        weather,
        estimate_columns=["Ave_simple", "Ave_est_cv"],
        labels={
            "Ave_simple": "Simple mean",
            "Ave_est_cv": "DH2006 estimated",
        },
        output=OUTPUT,
        title="Daily mean temperature error, Tokyo 2020-2022",
    )
    print(metrics[["estimate", "n", "RMSE", "Min error", "Max error"]].round(3).to_string(index=False))
    print(f"Fitted leave-one-year-out DH2006 params in {weather.attrs['fit_seconds']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

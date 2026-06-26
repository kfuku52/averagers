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
    weather = weather.join(photoperiod[["Sunset_nondimensional"]])
    weather["Ave_simple"] = averagers.get_simple_average_temperature(weather)
    return weather


def add_cross_validated_estimates(weather):
    weather = weather.copy()
    weather["Ave_est_yearly"] = pd.NA
    for window_size in WINDOW_SIZES:
        weather[f"Ave_est_monthly_ws{window_size}"] = pd.NA

    started = perf_counter()
    for year in sorted(weather["Year"].unique()):
        train = weather.loc[weather["Year"] != year].dropna(
            subset=["Ave", "Min", "Max", "Min_next", "Sunset_nondimensional", "Month"]
        )
        test_mask = weather["Year"] == year

        yearly_params = averagers.get_params(train, method="DH2006")
        weather.loc[test_mask, "Ave_est_yearly"] = averagers.get_average_temperature(
            weather.loc[test_mask, :],
            params=yearly_params,
            method="DH2006",
        )

        for window_size in WINDOW_SIZES:
            monthly_params = averagers.get_month_params(
                train,
                method="DH2006",
                window_size=window_size,
            )
            column = f"Ave_est_monthly_ws{window_size}"
            weather.loc[test_mask, column] = averagers.get_month_average_temperature(
                weather.loc[test_mask, :].copy(),
                monthly_params,
                method="DH2006",
            )

    weather.attrs["fit_seconds"] = perf_counter() - started
    return weather


def main():
    weather = add_cross_validated_estimates(build_base_data())
    estimate_columns = ["Ave_simple", "Ave_est_yearly"] + [
        f"Ave_est_monthly_ws{window_size}" for window_size in WINDOW_SIZES
    ]
    labels = {
        "Ave_simple": "Simple mean",
        "Ave_est_yearly": "DH2006 yearly",
        **{
            f"Ave_est_monthly_ws{window_size}": f"DH2006 monthly ws{window_size}"
            for window_size in WINDOW_SIZES
        },
    }

    _fig, _axes, metrics = averagers.plot_estimation_error_comparison(
        weather,
        estimate_columns=estimate_columns,
        labels=labels,
        output=OUTPUT,
        title="DH2006 parameter-window error, Tokyo 2020-2022",
        scatter_alpha=0.12,
        scatter_size=7,
        legend_outside=True,
    )
    print(metrics[["estimate", "n", "RMSE", "Min error", "Max error"]].round(3).to_string(index=False))
    print(f"Fitted leave-one-year-out DH2006 parameter options in {weather.attrs['fit_seconds']:.3f} s")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

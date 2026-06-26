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
    started = perf_counter()
    params = averagers.get_params(weather, method="DH2006", optimizer="least_squares")
    fit_seconds = perf_counter() - started
    weather["Ave_sim"] = averagers.get_average_temperature(
        weather,
        params=params,
        method="DH2006",
    )
    weather.attrs["params"] = params
    weather.attrs["fit_seconds"] = fit_seconds
    return weather


def main():
    weather = build_example_data()

    averagers.plot_temperature_estimates(
        weather,
        output=OUTPUT,
        title="Estimated daily mean temperature, Tokyo 2020-2022",
    )
    params = weather.attrs["params"]
    print(
        "Fitted DH2006 params from NASA POWER T2M: "
        f"CD={params['CD']:.3f}, CN={params['CN']:.3f} "
        f"in {weather.attrs['fit_seconds']:.3f} s"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()

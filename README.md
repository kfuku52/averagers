# averagers

[![Python >=3.8](https://img.shields.io/badge/python-%3E=3.8-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/kfuku52/averagers.svg)](https://github.com/kfuku52/averagers/commits/master)

## Overview

**averagers** is a Python package for estimating daily mean temperature from daily minimum and maximum temperatures.

## Dependencies

* [Python](https://www.python.org/) 3.8 or later
* [NumPy](https://github.com/numpy/numpy)
* [pandas](https://github.com/pandas-dev/pandas)
* [PyEphem](https://github.com/brandon-rhodes/pyephem)
* [Matplotlib](https://matplotlib.org/) for plotting

## Installation

```bash
pip install git+https://github.com/kfuku52/averagers
```

For local development:

```bash
pip install -e ".[test]"
pytest
```

## Data Source

`fetch_power_daily_temperature` downloads real daily near-surface temperature data from the [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/). It requests `T2M_MIN`, `T2M_MAX`, and `T2M`, then returns columns named `Min`, `Max`, `Ave`, and `Min_next` for direct use with the package functions.

`Sunrise_nondimensional` and `Sunset_nondimensional` are fractions of the day between 0 and 1. Calculation functions also accept legacy 0-24 hour values for these columns.

## Estimated Mean Plot

![Estimated daily mean temperature, Tokyo 2020-2022](docs/example_plot.png)

This example downloads three years of daily NASA POWER data for Tokyo, fits DH2006 parameters against NASA POWER `T2M`, and plots the estimated daily mean against the simple min/max mean.

```python
from time import perf_counter

import pandas as pd

import averagers

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
params = averagers.get_params(weather, method="DH2006")
fit_seconds = perf_counter() - started

weather["Ave_sim"] = averagers.get_average_temperature(
    weather,
    params=params,
    method="DH2006",
)

averagers.plot_temperature_estimates(
    weather,
    output="docs/example_plot.png",
    title="Estimated daily mean temperature, Tokyo 2020-2022",
)

print(f"CD={params['CD']:.3f}, CN={params['CN']:.3f}, fit={fit_seconds:.3f} s")
```

Run the script version:

```bash
python examples/plot_example.py
```

The script downloads daily data from 2020-01-01 to 2022-12-31 and writes `docs/example_plot.png`.

## Error Comparison Plot

![Daily mean temperature error comparison, Tokyo 2020-2022](docs/error_comparison.png)

This plot is similar to the cross-validation plots in the original notebook: it compares observed daily mean temperature against a simple min/max mean and a leave-one-year-out DH2006 estimate, then summarizes RMSE and extreme errors.

```python
import pandas as pd

import averagers

weather = averagers.fetch_power_daily_temperature("2020-01-01", "2022-12-31", 35.681, 139.767)
weather["Date"] = pd.to_datetime(weather["Date"])
weather = weather.join(
    averagers.get_photoperiod("2020-01-01", "2022-12-31", 35.681, 139.767, timezone=9)[
        ["Sunset_nondimensional"]
    ]
)
weather["Ave_simple"] = averagers.get_simple_average_temperature(weather)
weather["Ave_est_cv"] = pd.NA
weather["Year"] = weather["Date"].dt.year

for year in sorted(weather["Year"].unique()):
    train = weather.loc[weather["Year"] != year].dropna(
        subset=["Ave", "Min", "Max", "Min_next", "Sunset_nondimensional"]
    )
    params = averagers.get_params(train, method="DH2006")
    test_mask = weather["Year"] == year
    weather.loc[test_mask, "Ave_est_cv"] = averagers.get_average_temperature(
        weather.loc[test_mask],
        params=params,
        method="DH2006",
    )

averagers.plot_estimation_error_comparison(
    weather,
    estimate_columns=["Ave_simple", "Ave_est_cv"],
    labels={
        "Ave_simple": "Simple mean",
        "Ave_est_cv": "DH2006 estimated",
    },
    output="docs/error_comparison.png",
)
```

Run the script version:

```bash
python examples/error_comparison.py
```

The script downloads daily data from 2020-01-01 to 2022-12-31 and writes `docs/error_comparison.png`.

## Parameter Window Comparison

![DH2006 parameter-window error comparison, Tokyo 2020-2022](docs/window_size_comparison.png)

The monthly parameter estimator can be run with different month-window sizes. This example compares a single yearly DH2006 fit with monthly DH2006 fits using `window_size=0..3`.

```python
import pandas as pd

import averagers

weather = averagers.fetch_power_daily_temperature("2020-01-01", "2022-12-31", 35.681, 139.767)
weather["Date"] = pd.to_datetime(weather["Date"])
weather["Year"] = weather["Date"].dt.year
weather = weather.join(
    averagers.get_photoperiod("2020-01-01", "2022-12-31", 35.681, 139.767, timezone=9)[
        ["Sunset_nondimensional"]
    ]
)
weather["Ave_simple"] = averagers.get_simple_average_temperature(weather)
weather["Ave_est_yearly"] = pd.NA
for window_size in [0, 1, 2, 3]:
    weather[f"Ave_est_monthly_ws{window_size}"] = pd.NA

for year in sorted(weather["Year"].unique()):
    train = weather.loc[weather["Year"] != year].dropna(
        subset=["Ave", "Min", "Max", "Min_next", "Sunset_nondimensional", "Month"]
    )
    test_mask = weather["Year"] == year

    yearly_params = averagers.get_params(train, method="DH2006")
    weather.loc[test_mask, "Ave_est_yearly"] = averagers.get_average_temperature(
        weather.loc[test_mask],
        params=yearly_params,
        method="DH2006",
    )

    for window_size in [0, 1, 2, 3]:
        monthly_params = averagers.get_month_params(
            train,
            method="DH2006",
            window_size=window_size,
        )
        weather.loc[test_mask, f"Ave_est_monthly_ws{window_size}"] = (
            averagers.get_month_average_temperature(
                weather.loc[test_mask].copy(),
                monthly_params,
                method="DH2006",
            )
        )

averagers.plot_estimation_error_comparison(
    weather,
    estimate_columns=[
        "Ave_simple",
        "Ave_est_yearly",
        "Ave_est_monthly_ws0",
        "Ave_est_monthly_ws1",
        "Ave_est_monthly_ws2",
        "Ave_est_monthly_ws3",
    ],
    labels={
        "Ave_simple": "Simple mean",
        "Ave_est_yearly": "DH2006 yearly",
        "Ave_est_monthly_ws0": "DH2006 monthly ws0",
        "Ave_est_monthly_ws1": "DH2006 monthly ws1",
        "Ave_est_monthly_ws2": "DH2006 monthly ws2",
        "Ave_est_monthly_ws3": "DH2006 monthly ws3",
    },
    output="docs/window_size_comparison.png",
    legend_outside=True,
)
```

Run the script version:

```bash
python examples/window_size_comparison.py
```

In the generated Tokyo 2020-2022 example, the lowest RMSE is from `DH2006 monthly ws1`.

## Citation

This program was reported in:

**Fukushima et al. 2021.** A discordance of seasonally covarying cues uncovers misregulated phenotypes in the heterophyllous pitcher plant *Cephalotus follicularis*. Proceedings of the Royal Society B 288(1943): 20202568. https://royalsocietypublishing.org/doi/10.1098/rspb.2020.2568

Also, this program implements the method reported in the following paper.

**Dall'Amico and Hornsteiner. 2006.** A simple method for estimating daily and monthly mean temperatures from daily minima and maxima. International Journal of Climatology 26: 1929-1936. https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.1363

## Licensing

**averagers** is MIT-licensed. See [LICENSE](LICENSE) for details.

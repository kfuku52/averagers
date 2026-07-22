# Cherry-flowering pilot analysis

This case study tests whether estimating daily mean temperature from daily
minimum and maximum temperatures changes a simple thermal-time analysis of
cherry flowering.

## Scope

- 10 JMA stations from Sapporo to Kagoshima
- 45 complete years per station (1981–2025)
- 450 JMA cherry-flowering observations
- NASA POWER `T2M` as the pilot daily-mean reference
- simple min-max mean, DH2006, and Diurnal3 reconstructions
- leave-one-year-out fitting for both temperature reconstruction and flowering
  thresholds

The flowering model accumulates growing degree-days from 1 January. Base
temperatures of 0, 5, and 10 degrees Celsius are included as sensitivity checks.
This is a diagnostic use case, not a complete biological model: chilling,
photoperiod, station history, and observation-tree changes are not modeled.
At a 10 degrees Celsius base, one station has a zero POWER event-GDD reference,
so its percentage thermal-requirement shift is undefined and excluded from the
corresponding aggregate.

## Main result

Diurnal3 reduced daily-mean temperature RMSE from 0.519 degrees Celsius for the
simple min-max mean to 0.464 degrees Celsius, a 10.6% reduction. After each
method's flowering threshold was recalibrated, flowering-date accuracy changed
little at the primary 0 degrees Celsius base: MAE values were 4.30–4.42 days.

The clearer downstream effect was on inferred thermal requirement. At the
0 degrees Celsius base, the simple min-max mean changed median event GDD by a
median of +4.76% across stations and by as much as +13.16%. Diurnal3 stayed much
closer to the POWER reference, with a maximum absolute station shift of 1.85%.

These results suggest that improved daily-mean reconstruction may matter more
for comparing fitted biological parameters than for predictions made after
local recalibration.

## Reproduce

From the repository root:

```bash
python analysis/cherry_phenology/run.py
```

The command downloads JMA source files and NASA POWER temperatures, caches them
under `.cache/cherry_phenology/`, and rewrites `summary.json` in this directory.
Use `--help` to select another cache or output directory. The full run downloads
approximately 45 years of daily data for ten locations.

## Outputs

[`summary.json`](summary.json) contains the aggregate temperature errors,
flowering prediction errors, paired bootstrap intervals, and thermal-requirement
shifts. Raw observations, API caches, reconstructed daily tables, and individual
station-year predictions are intentionally not versioned because they are
downloadable or generated.

## Data sources and caveats

- JMA cumulative phenological observations:
  <https://www.data.jma.go.jp/sakura/data/ruinenchi/004.csv>
- JMA AMeDAS station table:
  <https://www.jma.go.jp/bosai/amedas/const/amedastable.json>
- NASA POWER Daily API:
  <https://power.larc.nasa.gov/docs/services/api/temporal/daily/>

NASA POWER is a gridded product rather than an hourly observation at each JMA
station. A stronger validation would integrate hourly AMeDAS observations and
use a chilling-plus-forcing phenology model.

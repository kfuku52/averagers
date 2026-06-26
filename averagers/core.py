import datetime
import itertools
from pathlib import Path

import ephem
import numpy
import pandas


_METHOD_PARAMS = {
    "DH2006": ("CD", "CN"),
    "KF": ("C1", "C2", "C3"),
}
_SECONDS_PER_DAY = 24 * 60 * 60


def _validate_method(method):
    if method not in _METHOD_PARAMS:
        valid = ", ".join(sorted(_METHOD_PARAMS))
        raise ValueError(f"method must be one of: {valid}")
    return method


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _require_params(params, names):
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")


def _method_required_columns(method):
    method = _validate_method(method)
    required_columns = ["Min", "Max", "Min_next", "Sunset_nondimensional"]
    if method == "KF":
        required_columns.extend(["Max_prev", "Sunrise_nondimensional"])
    return required_columns


def _as_day_fraction(values, name):
    """Return day fractions, accepting legacy 0-24 hour values."""
    if isinstance(values, pandas.Series):
        out = pandas.to_numeric(values, errors="raise").astype(float)
        valid = out.dropna()
        if valid.empty:
            return out

        min_value = valid.min()
        max_value = valid.max()
        if max_value > 1:
            if min_value < 0 or max_value > 24:
                raise ValueError(f"{name} must be between 0 and 1, or 0 and 24 hours")
            if (valid <= 1).any():
                raise ValueError(f"{name} mixes day fractions and hour values")
            out = out / 24
            valid = out.dropna()

        if valid.min() < 0 or valid.max() > 1:
            raise ValueError(f"{name} must be between 0 and 1")
        return out

    if pandas.isna(values):
        return numpy.nan

    value = float(values)
    if value > 1:
        if value > 24:
            raise ValueError(f"{name} must be between 0 and 1, or 0 and 24 hours")
        value = value / 24
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def get_temp_dif(CD, CN, min0, max0, ave0, min1, sunset_nondimensional):
    """Return the DH2006 estimated-minus-observed temperature difference."""
    sunset = _as_day_fraction(sunset_nondimensional, "Sunset_nondimensional")
    daytime_temp = min0 + (CD * (max0 - min0))
    nighttime_temp = min1 + (CN * (max0 - min1))
    return (daytime_temp * sunset) + (nighttime_temp * (1 - sunset)) - ave0


def get_params_pulp(df, CDmin=0, CDmax=100, CNmin=0, CNmax=100):
    try:
        import pulp
    except ImportError as exc:
        raise ImportError("get_params_pulp requires the optional 'pulp' package.") from exc

    _require_columns(df, ["Min", "Max", "Ave", "Min_next", "Sunset_nondimensional"])
    if df.empty:
        raise ValueError("df must contain at least one row")

    CD = pulp.LpVariable("CD", CDmin, CDmax, cat="Continuous")
    CN = pulp.LpVariable("CN", CNmin, CNmax, cat="Continuous")
    abs_errors = {
        index: pulp.LpVariable(f"abs_error_{position}", lowBound=0)
        for position, index in enumerate(df.index)
    }

    model = pulp.LpProblem("averagers_params", sense=pulp.LpMinimize)
    model += pulp.lpSum(abs_errors.values())

    for index in df.index:
        dif = get_temp_dif(
            CD,
            CN,
            df.loc[index, "Min"],
            df.loc[index, "Max"],
            df.loc[index, "Ave"],
            df.loc[index, "Min_next"],
            df.loc[index, "Sunset_nondimensional"],
        )
        model += dif <= abs_errors[index]
        model += -dif <= abs_errors[index]

    status = model.solve()
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"pulp optimization failed: {pulp.LpStatus[status]}")

    return {"CD": float(CD.value()), "CN": float(CN.value())}


def get_average_temperature(df, params, method):
    method = _validate_method(method)
    _require_params(params, _METHOD_PARAMS[method])
    _require_columns(df, _method_required_columns(method))

    min1 = df.loc[:, "Min"]
    max1 = df.loc[:, "Max"]
    min2 = df.loc[:, "Min_next"]
    sunset = _as_day_fraction(df.loc[:, "Sunset_nondimensional"], "Sunset_nondimensional")

    if method == "DH2006":
        daytime_temp = min1 + (params["CD"] * (max1 - min1))
        nighttime_temp = min2 + (params["CN"] * (max1 - min2))
        return (daytime_temp * sunset) + (nighttime_temp * (1 - sunset))

    sunrise = _as_day_fraction(df.loc[:, "Sunrise_nondimensional"], "Sunrise_nondimensional")
    if ((sunset - sunrise).dropna() < 0).any():
        raise ValueError("Sunset_nondimensional must be later than Sunrise_nondimensional")

    max0 = df.loc[:, "Max_prev"]
    prop1 = sunrise
    prop2 = sunset - sunrise
    prop3 = 1 - sunset
    temp1 = min1 + (params["C1"] * (max0 - min1))
    temp2 = min1 + (params["C2"] * (max1 - min1))
    temp3 = min2 + (params["C3"] * (max1 - min2))
    return (temp1 * prop1) + (temp2 * prop2) + (temp3 * prop3)


def get_simple_average_temperature(df, min_column="Min", max_column="Max"):
    _require_columns(df, [min_column, max_column])
    return (df.loc[:, min_column] + df.loc[:, max_column]) / 2


def plot_temperature_estimates(
    df,
    estimated_column="Ave_sim",
    date_column="Date",
    min_column="Min",
    max_column="Max",
    simple_average_column=None,
    output=None,
    ax=None,
    title="Estimated daily mean temperature",
    ylabel="Temperature (deg C)",
    dpi=160,
    estimated_marker="auto",
    max_marker_points=120,
    show_range=False,
    range_alpha=0.22,
    minmax_linewidth=1.0,
    simple_average_linewidth=1.0,
    estimated_linewidth=1.3,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_temperature_estimates requires matplotlib.") from exc

    _require_columns(df, [min_column, max_column, estimated_column])
    if date_column is not None and date_column in df.columns:
        x = df.loc[:, date_column]
        xlabel = date_column
    else:
        x = df.index
        xlabel = "Index"

    if simple_average_column is None:
        simple_average = get_simple_average_temperature(
            df,
            min_column=min_column,
            max_column=max_column,
        )
    else:
        _require_columns(df, [simple_average_column])
        simple_average = df.loc[:, simple_average_column]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    if estimated_marker == "auto":
        estimated_marker = None if len(df) > max_marker_points else "o"

    if show_range:
        ax.fill_between(
            x,
            df.loc[:, min_column],
            df.loc[:, max_column],
            color="#9ecae1",
            alpha=range_alpha,
            label="Daily min-max range",
        )
    ax.plot(x, df.loc[:, min_column], color="#3182bd", linewidth=minmax_linewidth, label="Minimum")
    ax.plot(x, df.loc[:, max_column], color="#de2d26", linewidth=minmax_linewidth, label="Maximum")
    ax.plot(
        x,
        simple_average,
        color="#31a354",
        linewidth=simple_average_linewidth,
        linestyle="--",
        label="Simple average of min/max",
    )
    ax.plot(
        x,
        df.loc[:, estimated_column],
        color="#111111",
        linewidth=estimated_linewidth,
        marker=estimated_marker,
        markersize=4,
        label="Estimated mean",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d9d9d9", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)

    return fig, ax


def _estimate_label(column, labels):
    if labels is None:
        return column
    if isinstance(labels, dict):
        return labels.get(column, column)
    return column


def get_estimation_error_metrics(df, estimate_columns, observed_column="Ave", labels=None):
    _require_columns(df, [observed_column, *estimate_columns])

    rows = []
    for i, column in enumerate(estimate_columns):
        observed_estimated = df.loc[:, [observed_column, column]].dropna()
        if observed_estimated.empty:
            raise ValueError(f"No complete observations for {column}")
        error = observed_estimated.loc[:, column] - observed_estimated.loc[:, observed_column]
        rows.append(
            {
                "column": column,
                "estimate": _estimate_label(
                    column,
                    labels if not isinstance(labels, list) else dict(zip(estimate_columns, labels)),
                ),
                "n": int(error.shape[0]),
                "RMSE": float((error.pow(2).mean()) ** 0.5),
                "MAE": float(error.abs().mean()),
                "Mean error": float(error.mean()),
                "Min error": float(error.min()),
                "Max error": float(error.max()),
            }
        )
    return pandas.DataFrame(rows)


def plot_estimation_error_comparison(
    df,
    estimate_columns,
    observed_column="Ave",
    labels=None,
    output=None,
    title="Daily mean temperature estimation error",
    dpi=160,
    scatter_alpha=0.25,
    scatter_size=9,
    legend_outside=False,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_estimation_error_comparison requires matplotlib.") from exc

    label_map = None
    if isinstance(labels, list):
        label_map = dict(zip(estimate_columns, labels))
    elif isinstance(labels, dict):
        label_map = labels

    metrics = get_estimation_error_metrics(
        df,
        estimate_columns=estimate_columns,
        observed_column=observed_column,
        labels=label_map,
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.2))
    scatter_ax, metric_ax = axes
    colors = plt.get_cmap("tab10").colors

    value_columns = [observed_column, *estimate_columns]
    values = df.loc[:, value_columns].dropna()
    xymin = float(values.min().min())
    xymax = float(values.max().max())
    padding = (xymax - xymin) * 0.05 if xymax > xymin else 1
    xymin -= padding
    xymax += padding

    for i, column in enumerate(estimate_columns):
        clean = df.loc[:, [observed_column, column]].dropna()
        label = _estimate_label(column, label_map)
        scatter_ax.scatter(
            clean.loc[:, observed_column],
            clean.loc[:, column],
            s=scatter_size,
            alpha=scatter_alpha,
            color=colors[i % len(colors)],
            label=label,
            edgecolors="none",
        )
    scatter_ax.plot([xymin, xymax], [xymin, xymax], color="#111111", linestyle="--", linewidth=0.9)
    scatter_ax.set_xlim(xymin, xymax)
    scatter_ax.set_ylim(xymin, xymax)
    scatter_ax.set_xlabel("Observed mean temperature (deg C)")
    scatter_ax.set_ylabel("Estimated mean temperature (deg C)")
    scatter_ax.set_title("Observed vs estimated")
    scatter_ax.grid(True, color="#d9d9d9", linewidth=0.8)
    scatter_ax.legend(frameon=False, loc="upper left")

    metric_names = ["RMSE", "Min error", "Max error"]
    width = 0.8 / len(estimate_columns)
    x = numpy.arange(len(metric_names))
    for i, row in metrics.iterrows():
        offset = (i - (len(metrics) - 1) / 2) * width
        metric_ax.bar(
            x + offset,
            [row[name] for name in metric_names],
            width=width,
            color=colors[i % len(colors)],
            label=row["estimate"],
        )
    metric_ax.axhline(0, color="#111111", linewidth=0.8)
    metric_ax.set_xticks(x)
    metric_ax.set_xticklabels(metric_names)
    metric_ax.set_ylabel("Error (deg C)")
    metric_ax.set_title("Error metrics")
    metric_ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    if legend_outside:
        metric_ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        metric_ax.legend(frameon=False, loc="upper left")

    fig.suptitle(title)
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)

    return fig, axes, metrics


def get_month_average_temperature(df, mparams, method):
    _validate_method(method)
    _require_columns(df, ["Month"])

    result = pandas.Series(numpy.nan, index=df.index, name="Ave_simM")
    for month, params in mparams.items():
        month = int(month)
        is_month = df["Month"] == month
        if is_month.any():
            result.loc[is_month] = get_average_temperature(df.loc[is_month, :], params, method=method)

    df.loc[:, "Ave_simM"] = result
    return result


def _temperature_model_terms(df, method):
    method = _validate_method(method)
    _require_columns(df, _method_required_columns(method))

    min1 = pandas.to_numeric(df.loc[:, "Min"], errors="raise").astype(float)
    max1 = pandas.to_numeric(df.loc[:, "Max"], errors="raise").astype(float)
    min2 = pandas.to_numeric(df.loc[:, "Min_next"], errors="raise").astype(float)
    sunset = _as_day_fraction(df.loc[:, "Sunset_nondimensional"], "Sunset_nondimensional")

    if method == "DH2006":
        base = (min1 * sunset) + (min2 * (1 - sunset))
        design = pandas.DataFrame(
            {
                "CD": (max1 - min1) * sunset,
                "CN": (max1 - min2) * (1 - sunset),
            },
            index=df.index,
        )
        return base.astype(float), design.astype(float)

    max0 = pandas.to_numeric(df.loc[:, "Max_prev"], errors="raise").astype(float)
    sunrise = _as_day_fraction(df.loc[:, "Sunrise_nondimensional"], "Sunrise_nondimensional")
    if ((sunset - sunrise).dropna() < 0).any():
        raise ValueError("Sunset_nondimensional must be later than Sunrise_nondimensional")

    prop1 = sunrise
    prop2 = sunset - sunrise
    prop3 = 1 - sunset
    base = (min1 * prop1) + (min1 * prop2) + (min2 * prop3)
    design = pandas.DataFrame(
        {
            "C1": (max0 - min1) * prop1,
            "C2": (max1 - min1) * prop2,
            "C3": (max1 - min2) * prop3,
        },
        index=df.index,
    )
    return base.astype(float), design.astype(float)


def get_params_least_squares(df, param_min=0, param_max=10, method="DH2006"):
    """Fit temperature-estimation parameters with linear least squares."""
    method = _validate_method(method)
    if param_min > param_max:
        raise ValueError("param_min must be <= param_max")
    if df.empty:
        raise ValueError("df must contain at least one row")
    _require_columns(df, ["Ave"])

    base, design = _temperature_model_terms(df, method)
    training = pandas.concat(
        [
            pandas.to_numeric(df.loc[:, "Ave"], errors="raise").rename("Ave"),
            base.rename("_base"),
            design,
        ],
        axis=1,
    ).dropna()
    if training.empty:
        raise ValueError("df must contain at least one complete row")

    param_names = list(_METHOD_PARAMS[method])
    matrix = training.loc[:, param_names].to_numpy(dtype=float)
    target = (
        training.loc[:, "Ave"].to_numpy(dtype=float)
        - training.loc[:, "_base"].to_numpy(dtype=float)
    )
    params, *_ = numpy.linalg.lstsq(matrix, target, rcond=None)
    params = numpy.clip(params, param_min, param_max)

    out = dict(zip(param_names, [float(value) for value in params]))
    estimated = get_average_temperature(df.loc[training.index, :], out, method=method)
    variance = ((estimated - training.loc[:, "Ave"]) ** 2).mean()
    out["variance"] = float(variance)
    return out


def get_params(
    df,
    param_min=0,
    param_max=10,
    max_step=1000,
    small_dif=10**-6,
    method="DH2006",
    num_grid=3,
    verbose=False,
    optimizer="grid",
):
    method = _validate_method(method)
    if param_min > param_max:
        raise ValueError("param_min must be <= param_max")
    if df.empty:
        raise ValueError("df must contain at least one row")
    _require_columns(df, ["Ave"])

    optimizer = optimizer.lower().replace("-", "_")
    if optimizer in {"least_squares", "ls"}:
        out = get_params_least_squares(
            df,
            param_min=param_min,
            param_max=param_max,
            method=method,
        )
        if verbose:
            print(f"Least-squares optimization completed: params={out}")
        return out
    if optimizer != "grid":
        raise ValueError("optimizer must be 'grid' or 'least_squares'")

    if num_grid < 3:
        raise ValueError("num_grid must be >= 3")
    if max_step < 1:
        raise ValueError("max_step must be >= 1")
    if small_dif <= 0:
        raise ValueError("small_dif must be > 0")

    param_names = _METHOD_PARAMS[method]
    param_ranges = {name: [param_min, param_max] for name in param_names}
    results = None
    best_index = None

    for step in range(1, max_step + 1):
        param_units = {
            name: (param_ranges[name][1] - param_ranges[name][0]) / num_grid
            for name in param_names
        }
        param_grids = {}
        for name in param_names:
            if param_units[name] < (small_dif / num_grid):
                param_grids[name] = [(param_ranges[name][0] + param_ranges[name][1]) / 2]
            else:
                param_grids[name] = numpy.linspace(
                    param_ranges[name][0],
                    param_ranges[name][1],
                    num_grid + 1,
                )

        results = {"variance": []}
        for name in param_names:
            results[name] = []

        for grid in itertools.product(*[param_grids[name] for name in param_names]):
            current_params = dict(zip(param_names, grid))
            for name in param_names:
                results[name].append(current_params[name])
            ave = get_average_temperature(df, current_params, method)
            variance = ((ave - df.loc[:, "Ave"]) ** 2).mean()
            results["variance"].append(float(variance))

        variances = numpy.asarray(results["variance"], dtype=float)
        if not numpy.isfinite(variances).any():
            raise ValueError("No finite variance could be calculated")

        best_index = int(numpy.nanargmin(variances))
        best_params = {name: results[name][best_index] for name in param_names}

        for name in param_names:
            param_ranges[name][0] = max(best_params[name] - param_units[name], param_min)
            param_ranges[name][1] = min(best_params[name] + param_units[name], param_max)

        if verbose:
            print(f"Optimization round {step}: params={best_params}")

        if all(unit < (small_dif / num_grid) for unit in param_units.values()):
            if verbose:
                print("Optimization completed successfully.")
            break

    out = {"variance": float(results["variance"][best_index])}
    for name in param_names:
        out[name] = float(results[name][best_index])
    return out


def get_month_params(
    df,
    param_min=0,
    param_max=10,
    max_step=1000,
    small_dif=10**-6,
    method="DH2006",
    num_grid=3,
    window_size=0,
    verbose=False,
    optimizer="grid",
):
    if window_size < 0:
        raise ValueError("window_size must be >= 0")
    _require_columns(df, ["Month"])

    mparams = {}
    for month in df.Month.dropna().unique():
        month = int(month)
        month_window = numpy.arange(month - window_size, month + window_size + 1, 1)
        month_window = [((int(value) - 1) % 12) + 1 for value in month_window]
        is_month = df["Month"].isin(month_window)
        mparams[str(month)] = get_params(
            df.loc[is_month, :],
            param_min=param_min,
            param_max=param_max,
            max_step=max_step,
            small_dif=small_dif,
            method=method,
            num_grid=num_grid,
            verbose=verbose,
            optimizer=optimizer,
        )
        if verbose:
            print("Month", month_window, mparams[str(month)])
    return mparams


def smooth_month_params(mparams, smooth_window=1, method="DH2006"):
    method = _validate_method(method)
    if smooth_window < 0:
        raise ValueError("smooth_window must be >= 0")
    if not mparams:
        raise ValueError("mparams must contain at least one month")

    param_names = _METHOD_PARAMS[method]
    smoothed = {}
    month_params = {int(month): params for month, params in mparams.items()}
    for month, params in month_params.items():
        _require_params(params, param_names)
        month_window = numpy.arange(month - smooth_window, month + smooth_window + 1, 1)
        month_window = [((int(value) - 1) % 12) + 1 for value in month_window]
        neighbors = [month_params[value] for value in month_window if value in month_params]
        smoothed_params = {
            name: float(numpy.mean([neighbor[name] for neighbor in neighbors]))
            for name in param_names
        }
        variances = [
            neighbor["variance"]
            for neighbor in neighbors
            if "variance" in neighbor and pandas.notna(neighbor["variance"])
        ]
        if variances:
            smoothed_params["variance"] = float(numpy.mean(variances))
        smoothed[str(month)] = smoothed_params
    return smoothed


def get_cyclic_month_params(
    df,
    param_min=0,
    param_max=10,
    max_step=1000,
    small_dif=10**-6,
    method="DH2006",
    num_grid=3,
    window_size=1,
    smooth_window=1,
    verbose=False,
    optimizer="least_squares",
):
    monthly_params = get_month_params(
        df,
        param_min=param_min,
        param_max=param_max,
        max_step=max_step,
        small_dif=small_dif,
        method=method,
        num_grid=num_grid,
        window_size=window_size,
        verbose=verbose,
        optimizer=optimizer,
    )
    return smooth_month_params(
        monthly_params,
        smooth_window=smooth_window,
        method=method,
    )


def _spec_column(spec):
    if "column" in spec:
        return spec["column"]
    name = spec.get("name")
    if name:
        return str(name).lower().replace(" ", "_").replace("-", "_")
    kind = spec.get("kind", "yearly")
    method = spec.get("method", "DH2006")
    return f"Ave_est_{method}_{kind}"


def _spec_name(spec, column):
    return spec.get("name", column)


def _fit_kwargs_from_spec(spec):
    keys = ["param_min", "param_max", "max_step", "small_dif", "num_grid", "optimizer"]
    return {key: spec[key] for key in keys if key in spec}


def cross_validate_estimates(
    df,
    specs=None,
    observed_column="Ave",
    fold_column="Year",
):
    if specs is None:
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
    if not specs:
        raise ValueError("specs must contain at least one estimate specification")

    _require_columns(df, [observed_column, fold_column])
    out = df.copy()
    labels = {}
    estimate_columns = []

    for spec in specs:
        column = _spec_column(spec)
        labels[column] = _spec_name(spec, column)
        estimate_columns.append(column)
        if column not in out.columns:
            out[column] = numpy.nan

        kind = spec.get("kind", "yearly").lower().replace("-", "_")
        if kind in {"simple", "simple_mean", "minmax", "min_max"}:
            out.loc[:, column] = get_simple_average_temperature(out)
            continue

        method = _validate_method(spec.get("method", "DH2006"))
        required_columns = [observed_column, *_method_required_columns(method)]
        if kind in {
            "monthly",
            "monthly_smoothed",
            "smoothed_monthly",
            "seasonal",
            "cyclic",
            "cyclic_monthly",
        }:
            required_columns.append("Month")

        fit_kwargs = _fit_kwargs_from_spec(spec)
        for fold in sorted(out.loc[:, fold_column].dropna().unique()):
            train = out.loc[out.loc[:, fold_column] != fold, :].dropna(subset=required_columns)
            if train.empty:
                continue
            test_mask = out.loc[:, fold_column] == fold

            if kind in {"yearly", "global", "single"}:
                params = get_params(train, method=method, **fit_kwargs)
                out.loc[test_mask, column] = get_average_temperature(
                    out.loc[test_mask, :],
                    params=params,
                    method=method,
                )
            elif kind in {
                "monthly",
                "monthly_smoothed",
                "smoothed_monthly",
                "seasonal",
                "cyclic",
                "cyclic_monthly",
            }:
                if kind in {"monthly"}:
                    monthly_params = get_month_params(
                        train,
                        method=method,
                        window_size=spec.get("window_size", 0),
                        **fit_kwargs,
                    )
                else:
                    monthly_params = get_cyclic_month_params(
                        train,
                        method=method,
                        window_size=spec.get("window_size", 1),
                        smooth_window=spec.get("smooth_window", 1),
                        **fit_kwargs,
                    )
                out.loc[test_mask, column] = get_month_average_temperature(
                    out.loc[test_mask, :].copy(),
                    monthly_params,
                    method=method,
                )
            else:
                raise ValueError(f"Unsupported estimate kind: {spec.get('kind')}")

    metrics = get_estimation_error_metrics(
        out,
        estimate_columns=estimate_columns,
        observed_column=observed_column,
        labels=labels,
    )
    return out, metrics


def select_month_window(
    df,
    windows=(0, 1, 2, 3),
    method="DH2006",
    observed_column="Ave",
    fold_column="Year",
    optimizer="least_squares",
    metric="RMSE",
    smooth_window=None,
):
    method = _validate_method(method)
    windows = list(windows)
    if not windows:
        raise ValueError("windows must contain at least one value")

    specs = []
    column_to_window = {}
    for window_size in windows:
        if window_size < 0:
            raise ValueError("window sizes must be >= 0")
        kind = "monthly_smoothed" if smooth_window is not None else "monthly"
        suffix = (
            f"{method} monthly ws{window_size}"
            if smooth_window is None
            else f"{method} monthly ws{window_size} smooth{smooth_window}"
        )
        column = (
            f"Ave_est_{method.lower()}_monthly_ws{window_size}"
            if smooth_window is None
            else f"Ave_est_{method.lower()}_monthly_ws{window_size}_smooth{smooth_window}"
        )
        specs.append(
            {
                "name": suffix,
                "column": column,
                "kind": kind,
                "method": method,
                "window_size": window_size,
                "smooth_window": smooth_window or 0,
                "optimizer": optimizer,
            }
        )
        column_to_window[column] = int(window_size)

    predictions, metrics = cross_validate_estimates(
        df,
        specs=specs,
        observed_column=observed_column,
        fold_column=fold_column,
    )
    if metric not in metrics.columns:
        raise ValueError(f"metric must be one of: {', '.join(metrics.columns)}")
    best_index = metrics.loc[:, metric].astype(float).idxmin()
    best = metrics.loc[best_index, :].to_dict()
    return {
        "best_window": column_to_window[best["column"]],
        "best": best,
        "metric": metric,
        "metrics": metrics,
        "predictions": predictions,
    }


def _ephem_to_local_datetime(value, timezone):
    return ephem.Date(value).datetime() + datetime.timedelta(hours=timezone)


def _fraction_since_midnight(value, local_midnight):
    return (value - local_midnight).total_seconds() / _SECONDS_PER_DAY


def get_photoperiod(start_date, end_date, lat, lon, timezone=0, elevation=3):
    place = ephem.Observer()
    place.lon = str(lon)
    place.lat = str(lat)
    place.elev = elevation
    sun = ephem.Sun(place)

    rows = []
    for date in pandas.date_range(start_date, end_date, freq="D"):
        local_midnight = date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        utc_start = local_midnight - datetime.timedelta(hours=timezone)

        try:
            sunrise_utc = place.next_rising(sun, start=utc_start)
            sunset_utc = place.next_setting(sun, start=sunrise_utc)
        except ephem.AlwaysUpError:
            sunrise = pandas.NaT
            sunset = pandas.NaT
            sunrise_fraction = numpy.nan
            sunset_fraction = numpy.nan
            daytime = 24.0
        except ephem.NeverUpError:
            sunrise = pandas.NaT
            sunset = pandas.NaT
            sunrise_fraction = numpy.nan
            sunset_fraction = numpy.nan
            daytime = 0.0
        else:
            sunrise = _ephem_to_local_datetime(sunrise_utc, timezone)
            sunset = _ephem_to_local_datetime(sunset_utc, timezone)
            daytime = (sunset - sunrise).total_seconds() / 3600
            sunrise_fraction = _fraction_since_midnight(sunrise, local_midnight)
            sunset_fraction = _fraction_since_midnight(sunset, local_midnight)

        rows.append(
            {
                "Daytime": daytime,
                "Year": date.year,
                "Month": date.month,
                "Day": date.day,
                "Sunrise": sunrise,
                "Sunset": sunset,
                "Sunrise_nondimensional": sunrise_fraction,
                "Sunset_nondimensional": sunset_fraction,
            }
        )

    return pandas.DataFrame(rows)

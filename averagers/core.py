import datetime
import itertools
from pathlib import Path

import ephem
import numpy
import pandas


_METHOD_PARAMS = {
    "DH2006": ("CD", "CN"),
    "Diurnal3": ("C1", "C2", "C3"),
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
    if method == "Diurnal3":
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
    show_scatter=True,
    metric_names=("RMSE",),
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
    metric_names = list(metric_names)
    if not metric_names:
        raise ValueError("metric_names must contain at least one metric")
    missing_metrics = [name for name in metric_names if name not in metrics.columns]
    if missing_metrics:
        raise ValueError(f"Unknown metric names: {', '.join(missing_metrics)}")

    colors = plt.get_cmap("tab10").colors

    if show_scatter:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.2))
        scatter_ax, metric_ax = axes

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
    else:
        fig, metric_ax = plt.subplots(figsize=(8, 4.2))
        axes = (metric_ax,)

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
    if not show_scatter:
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


def plot_estimation_metric_by_setting(
    metrics,
    setting_column="setting",
    method_column="method",
    metric="RMSE",
    output=None,
    title="Error by parameter setting",
    ylabel=None,
    dpi=160,
    setting_order=None,
    method_order=None,
    ax=None,
    legend_outside=True,
    xtick_rotation=0,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_estimation_metric_by_setting requires matplotlib.") from exc

    _require_columns(metrics, [setting_column, method_column, metric])
    clean = metrics.loc[:, [setting_column, method_column, metric]].dropna()
    if clean.empty:
        raise ValueError("metrics must contain at least one complete row")

    if setting_order is None:
        setting_order = list(dict.fromkeys(clean.loc[:, setting_column]))
    else:
        setting_order = list(setting_order)
    if method_order is None:
        method_order = list(dict.fromkeys(clean.loc[:, method_column]))
    else:
        method_order = list(method_order)

    if ax is None:
        fig_width = max(8, (1.15 * len(setting_order)) + (1.5 if legend_outside else 0))
        fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    else:
        fig = ax.figure

    colors = plt.get_cmap("tab10").colors
    x = numpy.arange(len(setting_order))
    present_by_setting = {}
    for setting in setting_order:
        present_by_setting[setting] = [
            method
            for method in method_order
            if not clean.loc[
                (clean.loc[:, setting_column] == setting)
                & (clean.loc[:, method_column] == method),
                metric,
            ].empty
        ]
    max_present = max(len(methods) for methods in present_by_setting.values())
    width = min(0.8 / max_present, 0.35)
    labelled_methods = set()
    for setting_index, setting in enumerate(setting_order):
        present_methods = present_by_setting[setting]
        for method_index, method in enumerate(present_methods):
            current = clean.loc[
                (clean.loc[:, setting_column] == setting)
                & (clean.loc[:, method_column] == method),
                metric,
            ]
            offset = (method_index - (len(present_methods) - 1) / 2) * width
            method_color_index = method_order.index(method)
            label = method if method not in labelled_methods else None
            ax.bar(
                x[setting_index] + offset,
                float(current.iloc[0]),
                width=width,
                color=colors[method_color_index % len(colors)],
                label=label,
            )
            labelled_methods.add(method)

    ax.set_xticks(x)
    ax.set_xticklabels(setting_order)
    if xtick_rotation:
        for label in ax.get_xticklabels():
            label.set_rotation(xtick_rotation)
            label.set_horizontalalignment("right")
    ax.set_ylabel(ylabel or f"{metric} (deg C)")
    ax.set_title(title)
    ax.grid(True, axis="y", color="#d9d9d9", linewidth=0.8)
    if legend_outside:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    else:
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)

    return fig, ax


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


def _validate_linear_feature_set(feature_set):
    feature_set = str(feature_set).lower().replace("-", "_")
    if feature_set not in {"temporal", "harmonic"}:
        raise ValueError("feature_set must be 'temporal' or 'harmonic'")
    return feature_set


def _linear_required_columns(feature_set):
    feature_set = _validate_linear_feature_set(feature_set)
    columns = [
        "Min",
        "Max",
        "Min_next",
        "Max_prev",
        "Sunrise_nondimensional",
        "Sunset_nondimensional",
    ]
    if feature_set == "harmonic":
        columns.append("Date")
    return columns


def _seasonal_terms(df):
    dates = pandas.to_datetime(df.loc[:, "Date"])
    angle = 2 * numpy.pi * (dates.dt.dayofyear.astype(float) - 1) / 365.25
    return numpy.sin(angle), numpy.cos(angle)


def _linear_daytime(df):
    if "Daytime" in df.columns:
        return pandas.to_numeric(df.loc[:, "Daytime"], errors="raise").astype(float)
    sunrise = _as_day_fraction(df.loc[:, "Sunrise_nondimensional"], "Sunrise_nondimensional")
    sunset = _as_day_fraction(df.loc[:, "Sunset_nondimensional"], "Sunset_nondimensional")
    return (sunset - sunrise) * 24


def _linear_design_matrix(df, feature_set="temporal"):
    feature_set = _validate_linear_feature_set(feature_set)
    _require_columns(df, _linear_required_columns(feature_set))

    sunrise = _as_day_fraction(df.loc[:, "Sunrise_nondimensional"], "Sunrise_nondimensional")
    sunset = _as_day_fraction(df.loc[:, "Sunset_nondimensional"], "Sunset_nondimensional")
    min1 = pandas.to_numeric(df.loc[:, "Min"], errors="raise").astype(float)
    max1 = pandas.to_numeric(df.loc[:, "Max"], errors="raise").astype(float)
    min2 = pandas.to_numeric(df.loc[:, "Min_next"], errors="raise").astype(float)
    max0 = pandas.to_numeric(df.loc[:, "Max_prev"], errors="raise").astype(float)

    data = {
        "Intercept": pandas.Series(1.0, index=df.index),
        "Min": min1,
        "Max": max1,
        "Min_next": min2,
        "Max_prev": max0,
        "Sunrise": sunrise,
        "Sunset": sunset,
    }
    if feature_set == "temporal":
        return pandas.DataFrame(data, index=df.index).astype(float)

    sin_doy, cos_doy = _seasonal_terms(df)
    base_features = {
        "Min": min1,
        "Max": max1,
        "Min_next": min2,
        "Max_prev": max0,
        "Sunrise": sunrise,
        "Sunset": sunset,
        "Daytime": _linear_daytime(df),
        "Range": max1 - min1,
    }
    harmonic = {"Intercept": pandas.Series(1.0, index=df.index)}
    for name, values in base_features.items():
        harmonic[name] = values
        harmonic[f"{name}*sinDOY"] = values * sin_doy
        harmonic[f"{name}*cosDOY"] = values * cos_doy
    return pandas.DataFrame(harmonic, index=df.index).astype(float)


def _fit_linear_coefficients(design, observed, ridge=0.0):
    if ridge < 0:
        raise ValueError("ridge must be >= 0")
    matrix = design.to_numpy(dtype=float)
    target = observed.to_numpy(dtype=float)
    if ridge > 0:
        penalty = ridge * numpy.eye(matrix.shape[1])
        penalty[0, 0] = 0
        return numpy.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
    coefficients, *_ = numpy.linalg.lstsq(matrix, target, rcond=None)
    return coefficients


def get_linear_params(df, feature_set="temporal", observed_column="Ave", ridge=0.0):
    feature_set = _validate_linear_feature_set(feature_set)
    if df.empty:
        raise ValueError("df must contain at least one row")
    _require_columns(df, [observed_column])

    design = _linear_design_matrix(df, feature_set=feature_set)
    training = pandas.concat(
        [
            pandas.to_numeric(df.loc[:, observed_column], errors="raise").rename(observed_column),
            design,
        ],
        axis=1,
    ).dropna()
    if training.empty:
        raise ValueError("df must contain at least one complete row")

    feature_names = list(design.columns)
    clean_design = training.loc[:, feature_names]
    observed = training.loc[:, observed_column]
    coefficients = _fit_linear_coefficients(clean_design, observed, ridge=ridge)
    predicted = clean_design.to_numpy(dtype=float) @ coefficients
    variance = ((predicted - observed.to_numpy(dtype=float)) ** 2).mean()
    return {
        "feature_set": feature_set,
        "coefficients": dict(zip(feature_names, [float(value) for value in coefficients])),
        "ridge": float(ridge),
        "variance": float(variance),
    }


def get_linear_average_temperature(df, params):
    if "coefficients" not in params:
        raise ValueError("params must contain coefficients")
    feature_set = _validate_linear_feature_set(params.get("feature_set", "temporal"))
    coefficients = params["coefficients"]
    design = _linear_design_matrix(df, feature_set=feature_set)
    missing = [name for name in coefficients if name not in design.columns]
    if missing:
        raise ValueError(f"Unknown linear coefficients: {', '.join(missing)}")
    feature_names = list(coefficients)
    values = design.loc[:, feature_names].to_numpy(dtype=float)
    coef = numpy.asarray([coefficients[name] for name in feature_names], dtype=float)
    return pandas.Series(values @ coef, index=df.index, name="Ave_linear")


def _month_window_values(month, window_size):
    month_window = numpy.arange(month - window_size, month + window_size + 1, 1)
    return [((int(value) - 1) % 12) + 1 for value in month_window]


def get_month_linear_params(
    df,
    feature_set="temporal",
    observed_column="Ave",
    window_size=0,
    ridge=0.0,
):
    if window_size < 0:
        raise ValueError("window_size must be >= 0")
    _require_columns(df, ["Month"])

    mparams = {}
    for month in df.Month.dropna().unique():
        month = int(month)
        month_window = _month_window_values(month, window_size)
        is_month = df["Month"].isin(month_window)
        mparams[str(month)] = get_linear_params(
            df.loc[is_month, :],
            feature_set=feature_set,
            observed_column=observed_column,
            ridge=ridge,
        )
    return mparams


def get_month_linear_average_temperature(df, mparams):
    _require_columns(df, ["Month"])

    result = pandas.Series(numpy.nan, index=df.index, name="Ave_linearM")
    for month, params in mparams.items():
        month = int(month)
        is_month = df["Month"] == month
        if is_month.any():
            result.loc[is_month] = get_linear_average_temperature(df.loc[is_month, :], params)
    df.loc[:, "Ave_linearM"] = result
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


def get_smoothed_month_params(*args, **kwargs):
    return get_cyclic_month_params(*args, **kwargs)


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


_SMOOTHED_MONTHLY_KINDS = {
    "monthly_smoothed",
    "smoothed",
    "smoothed_monthly",
    "smooth",
    "seasonal",
    "cyclic",
    "cyclic_monthly",
}
_MONTHLY_LIKE_KINDS = {"monthly", *_SMOOTHED_MONTHLY_KINDS}
_AUTO_KINDS = {"auto", "auto_setting", "setting_auto", "monthly_auto"}
_LINEAR_KINDS = {"linear", "linear_yearly", "linear_global"}
_MONTHLY_LINEAR_KINDS = {"monthly_linear", "linear_monthly"}


def _is_auto_window(window_size):
    return isinstance(window_size, str) and window_size.lower().replace("-", "_") == "auto"


def _normalize_kind(kind):
    return str(kind).lower().replace("-", "_")


def _is_smoothed_kind(kind):
    return _normalize_kind(kind) in _SMOOTHED_MONTHLY_KINDS


def get_default_auto_candidates(
    windows=(0, 1, 2, 3),
    smoothed_windows=(1, 2, 3),
    methods=("DH2006", "Diurnal3"),
    linear_ridge=0.25,
):
    """Return default candidate specs for all-method auto selection."""
    windows = [int(window_size) for window_size in windows]
    smoothed_windows = [int(window_size) for window_size in smoothed_windows]
    methods = [_validate_method(method) for method in methods]

    candidates = []
    candidates.extend(
        {
            "name": f"{method} yearly",
            "column": f"_auto_{method.lower()}_yearly",
            "kind": "yearly",
            "method": method,
            "setting": "yearly",
            "optimizer": "least_squares",
        }
        for method in methods
    )
    for window_size in windows:
        candidates.extend(
            {
                "name": f"{method} monthly ws{window_size}",
                "column": f"_auto_{method.lower()}_monthly_ws{window_size}",
                "kind": "monthly",
                "method": method,
                "setting": f"ws{window_size}",
                "window_size": window_size,
                "optimizer": "least_squares",
            }
            for method in methods
        )
    for window_size in smoothed_windows:
        candidates.extend(
            {
                "name": f"{method} smoothed ws{window_size}",
                "column": f"_auto_{method.lower()}_smoothed_ws{window_size}",
                "kind": "smoothed",
                "method": method,
                "setting": f"smoothed ws{window_size}",
                "window_size": window_size,
                "smooth_window": window_size,
                "optimizer": "least_squares",
            }
            for method in methods
        )
    candidates.extend(
        {
            "name": f"Monthly linear temporal ws{window_size}",
            "column": f"_auto_monthly_linear_temporal_ws{window_size}",
            "kind": "monthly_linear",
            "method": "Monthly linear",
            "setting": f"ws{window_size}",
            "feature_set": "temporal",
            "window_size": window_size,
            "ridge": linear_ridge,
        }
        for window_size in windows
    )
    candidates.append(
        {
            "name": "Linear harmonic",
            "column": "_auto_linear_harmonic",
            "kind": "linear",
            "method": "Harmonic",
            "setting": "harmonic",
            "feature_set": "harmonic",
        }
    )
    return candidates


def _candidate_setting(candidate):
    if "setting" in candidate:
        return candidate["setting"]
    kind = _normalize_kind(candidate.get("kind", "monthly"))
    window_size = candidate.get("window_size", 0 if kind == "monthly" else 1)
    if kind == "monthly":
        return f"ws{window_size}"
    if kind in _SMOOTHED_MONTHLY_KINDS:
        return f"smoothed ws{window_size}"
    if kind in _MONTHLY_LINEAR_KINDS:
        return f"ws{window_size}"
    if kind in _LINEAR_KINDS and candidate.get("feature_set") == "harmonic":
        return "harmonic"
    return kind


def _auto_candidate_specs(spec, method, fit_kwargs):
    candidates = spec.get("candidates")
    default_method = spec.get("candidate_method")
    if default_method is None:
        default_method = method if method in _METHOD_PARAMS else "DH2006"
    if candidates is None:
        candidates = [
            {
                "kind": "monthly",
                "window_size": window_size,
                "setting": f"ws{window_size}",
            }
            for window_size in spec.get("windows", (0, 1, 2, 3))
        ]
        candidates.extend(
            {
                "kind": "smoothed",
                "window_size": window_size,
                "smooth_window": window_size,
                "setting": f"smoothed ws{window_size}",
            }
            for window_size in spec.get("smoothed_windows", (1, 2, 3))
        )

    out = []
    for i, candidate in enumerate(candidates):
        candidate = dict(candidate)
        candidate_kind = _normalize_kind(candidate.get("kind", "monthly"))
        if candidate_kind in _AUTO_KINDS:
            raise ValueError("auto candidates must not use an auto kind")
        setting = _candidate_setting(candidate)
        candidate_method = candidate.get("method", default_method)
        method_slug = str(candidate_method).lower().replace(" ", "_")
        candidate["setting"] = setting
        candidate.setdefault("method", candidate_method)
        candidate.setdefault("name", f"{candidate_method} {setting}")
        candidate.setdefault("column", f"_auto_candidate_{method_slug}_{i}")
        for key, value in fit_kwargs.items():
            candidate.setdefault(key, value)
        out.append(candidate)
    if not out:
        raise ValueError("auto candidates must contain at least one candidate")
    return out


def _auto_selection_name(spec):
    if "name" in spec:
        return spec["name"]
    method = spec.get("method")
    setting = spec.get("setting", _candidate_setting(spec))
    if method:
        return f"{method} {setting}"
    return setting


def _select_auto_candidate(
    spec,
    train,
    method,
    observed_column,
    fold_column,
    fit_kwargs,
):
    candidates = _auto_candidate_specs(spec, method, fit_kwargs)
    if train.loc[:, fold_column].dropna().nunique() < 2:
        return candidates[0]

    _predictions, metrics = cross_validate_estimates(
        train,
        specs=candidates,
        observed_column=observed_column,
        fold_column=fold_column,
    )
    metric = spec.get("selection_metric", "RMSE")
    if metric not in metrics.columns:
        raise ValueError(f"selection_metric must be one of: {', '.join(metrics.columns)}")
    best_index = metrics.loc[:, metric].astype(float).idxmin()
    selected_column = metrics.loc[best_index, "column"]
    return next(candidate for candidate in candidates if candidate["column"] == selected_column)


def _resolve_month_window(
    spec,
    train,
    method,
    kind,
    observed_column,
    fold_column,
    fit_kwargs,
):
    window_size = spec.get("window_size", 0 if kind == "monthly" else 1)
    if not isinstance(window_size, str):
        return window_size
    if not _is_auto_window(window_size):
        raise ValueError("window_size must be a non-negative integer or 'auto'")

    windows = spec.get("windows", (0, 1, 2, 3))
    if train.loc[:, fold_column].dropna().nunique() < 2:
        return list(windows)[0]

    smooth_window = None if kind == "monthly" else spec.get("smooth_window", 1)
    selection = select_month_window(
        train,
        windows=windows,
        method=method,
        observed_column=observed_column,
        fold_column=fold_column,
        optimizer=fit_kwargs.get("optimizer", "least_squares"),
        metric=spec.get("selection_metric", "RMSE"),
        smooth_window=smooth_window,
    )
    return selection["best_window"]


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
    selected_windows = dict(out.attrs.get("selected_windows", {}))
    selected_settings = dict(out.attrs.get("selected_settings", {}))
    selected_candidates = dict(out.attrs.get("selected_candidates", {}))
    selected_methods = dict(out.attrs.get("selected_methods", {}))
    metric_metadata = {}

    for spec in specs:
        column = _spec_column(spec)
        labels[column] = _spec_name(spec, column)
        estimate_columns.append(column)
        metric_metadata[column] = {
            key: spec[key]
            for key in [
                "method",
                "kind",
                "setting",
                "window_size",
                "smooth_window",
                "feature_set",
                "ridge",
            ]
            if key in spec
        }
        if column not in out.columns:
            out[column] = numpy.nan

        kind = _normalize_kind(spec.get("kind", "yearly"))
        if kind in {"simple", "simple_mean", "minmax", "min_max"}:
            out.loc[:, column] = get_simple_average_temperature(out)
            continue

        if kind in _LINEAR_KINDS or kind in _MONTHLY_LINEAR_KINDS:
            feature_set = spec.get("feature_set", "temporal")
            required_columns = [observed_column, *_linear_required_columns(feature_set)]
            if kind in _MONTHLY_LINEAR_KINDS:
                required_columns.append("Month")

            for fold in sorted(out.loc[:, fold_column].dropna().unique()):
                train = out.loc[out.loc[:, fold_column] != fold, :].dropna(
                    subset=required_columns
                )
                if train.empty:
                    continue
                test_mask = out.loc[:, fold_column] == fold
                if kind in _LINEAR_KINDS:
                    params = get_linear_params(
                        train,
                        feature_set=feature_set,
                        observed_column=observed_column,
                        ridge=spec.get("ridge", 0.0),
                    )
                    out.loc[test_mask, column] = get_linear_average_temperature(
                        out.loc[test_mask, :],
                        params,
                    )
                else:
                    params = get_month_linear_params(
                        train,
                        feature_set=feature_set,
                        observed_column=observed_column,
                        window_size=spec.get("window_size", 0),
                        ridge=spec.get("ridge", 0.0),
                    )
                    out.loc[test_mask, column] = get_month_linear_average_temperature(
                        out.loc[test_mask, :].copy(),
                        params,
            )
            continue

        if kind in _AUTO_KINDS:
            method = spec.get("method", "DH2006")
            fit_kwargs = _fit_kwargs_from_spec(spec)
            selection_scope = str(spec.get("selection_scope", "fold")).lower().replace(
                "-",
                "_",
            )
            if selection_scope in {"global", "overall", "full"}:
                candidates = _auto_candidate_specs(spec, method, fit_kwargs)
                candidate_predictions, candidate_metrics = cross_validate_estimates(
                    out,
                    specs=candidates,
                    observed_column=observed_column,
                    fold_column=fold_column,
                )
                metric = spec.get("selection_metric", "RMSE")
                if metric not in candidate_metrics.columns:
                    raise ValueError(
                        f"selection_metric must be one of: {', '.join(candidate_metrics.columns)}"
                    )
                best_index = candidate_metrics.loc[:, metric].astype(float).idxmin()
                selected_column = candidate_metrics.loc[best_index, "column"]
                selected_spec = next(
                    candidate
                    for candidate in candidates
                    if candidate["column"] == selected_column
                )
                out.loc[:, column] = candidate_predictions.loc[:, selected_column]
                selected_setting = selected_spec.get(
                    "setting",
                    _candidate_setting(selected_spec),
                )
                selected_settings.setdefault(column, {})["all"] = selected_setting
                selected_candidates.setdefault(column, {})["all"] = _auto_selection_name(
                    selected_spec
                )
                if "method" in selected_spec:
                    selected_methods.setdefault(column, {})["all"] = selected_spec["method"]
                if "window_size" in selected_spec:
                    selected_windows.setdefault(column, {})["all"] = int(
                        selected_spec["window_size"]
                    )
                continue
            if selection_scope not in {"fold", "per_fold", "nested"}:
                raise ValueError("selection_scope must be 'fold' or 'global'")
            for fold in sorted(out.loc[:, fold_column].dropna().unique()):
                train = out.loc[out.loc[:, fold_column] != fold, :].dropna(
                    subset=[observed_column]
                )
                if train.empty:
                    continue
                test_mask = out.loc[:, fold_column] == fold
                selected_spec = _select_auto_candidate(
                    spec,
                    train,
                    method,
                    observed_column,
                    fold_column,
                    fit_kwargs,
                )
                selected_kind = _normalize_kind(selected_spec.get("kind", "monthly"))
                selected_fit_kwargs = _fit_kwargs_from_spec(selected_spec)
                selected_setting = selected_spec.get(
                    "setting",
                    _candidate_setting(selected_spec),
                )
                selected_settings.setdefault(column, {})[fold] = selected_setting
                selected_candidates.setdefault(column, {})[fold] = _auto_selection_name(
                    selected_spec
                )
                if "method" in selected_spec:
                    selected_methods.setdefault(column, {})[fold] = selected_spec["method"]
                if "window_size" in selected_spec:
                    selected_windows.setdefault(column, {})[fold] = int(
                        selected_spec["window_size"]
                    )

                if selected_kind in {"simple", "simple_mean", "minmax", "min_max"}:
                    out.loc[test_mask, column] = get_simple_average_temperature(
                        out.loc[test_mask, :]
                    )
                elif selected_kind in {"yearly", "global", "single"}:
                    selected_method = _validate_method(
                        selected_spec.get(
                            "method",
                            method if method in _METHOD_PARAMS else "DH2006",
                        )
                    )
                    selected_required = [
                        observed_column,
                        *_method_required_columns(selected_method),
                    ]
                    selected_train = train.dropna(subset=selected_required)
                    if selected_train.empty:
                        continue
                    params = get_params(
                        selected_train,
                        method=selected_method,
                        **selected_fit_kwargs,
                    )
                    out.loc[test_mask, column] = get_average_temperature(
                        out.loc[test_mask, :],
                        params=params,
                        method=selected_method,
                    )
                elif selected_kind in _MONTHLY_LIKE_KINDS:
                    selected_method = _validate_method(
                        selected_spec.get(
                            "method",
                            method if method in _METHOD_PARAMS else "DH2006",
                        )
                    )
                    selected_required = [
                        observed_column,
                        *_method_required_columns(selected_method),
                        "Month",
                    ]
                    selected_train = train.dropna(subset=selected_required)
                    if selected_train.empty:
                        continue
                    window_size = selected_spec.get(
                        "window_size",
                        0 if selected_kind == "monthly" else 1,
                    )
                    if selected_kind == "monthly":
                        monthly_params = get_month_params(
                            selected_train,
                            method=selected_method,
                            window_size=window_size,
                            **selected_fit_kwargs,
                        )
                    else:
                        monthly_params = get_smoothed_month_params(
                            selected_train,
                            method=selected_method,
                            window_size=window_size,
                            smooth_window=selected_spec.get("smooth_window", window_size),
                            **selected_fit_kwargs,
                        )
                    out.loc[test_mask, column] = get_month_average_temperature(
                        out.loc[test_mask, :].copy(),
                        monthly_params,
                        method=selected_method,
                    )
                elif selected_kind in _LINEAR_KINDS or selected_kind in _MONTHLY_LINEAR_KINDS:
                    feature_set = selected_spec.get("feature_set", "temporal")
                    selected_required = [
                        observed_column,
                        *_linear_required_columns(feature_set),
                    ]
                    if selected_kind in _MONTHLY_LINEAR_KINDS:
                        selected_required.append("Month")
                    selected_train = train.dropna(subset=selected_required)
                    if selected_train.empty:
                        continue
                    if selected_kind in _LINEAR_KINDS:
                        params = get_linear_params(
                            selected_train,
                            feature_set=feature_set,
                            observed_column=observed_column,
                            ridge=selected_spec.get("ridge", 0.0),
                        )
                        out.loc[test_mask, column] = get_linear_average_temperature(
                            out.loc[test_mask, :],
                            params,
                        )
                    else:
                        params = get_month_linear_params(
                            selected_train,
                            feature_set=feature_set,
                            observed_column=observed_column,
                            window_size=selected_spec.get("window_size", 0),
                            ridge=selected_spec.get("ridge", 0.0),
                        )
                        out.loc[test_mask, column] = get_month_linear_average_temperature(
                            out.loc[test_mask, :].copy(),
                            params,
                        )
                else:
                    raise ValueError(
                        f"Unsupported auto candidate kind: {selected_spec.get('kind')}"
                    )
            continue

        method = _validate_method(spec.get("method", "DH2006"))
        required_columns = [observed_column, *_method_required_columns(method)]
        if kind in _MONTHLY_LIKE_KINDS or kind in _AUTO_KINDS:
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
            elif kind in _MONTHLY_LIKE_KINDS:
                window_size = _resolve_month_window(
                    spec,
                    train,
                    method,
                    kind,
                    observed_column,
                    fold_column,
                    fit_kwargs,
                )
                if _is_auto_window(spec.get("window_size")):
                    selected_windows.setdefault(column, {})[fold] = int(window_size)
                    selected_settings.setdefault(column, {})[fold] = f"ws{window_size}"
                if kind in {"monthly"}:
                    monthly_params = get_month_params(
                        train,
                        method=method,
                        window_size=window_size,
                        **fit_kwargs,
                    )
                else:
                    monthly_params = get_smoothed_month_params(
                        train,
                        method=method,
                        window_size=window_size,
                        smooth_window=spec.get("smooth_window", 1),
                        **fit_kwargs,
                    )
                out.loc[test_mask, column] = get_month_average_temperature(
                    out.loc[test_mask, :].copy(),
                    monthly_params,
                    method=method,
                )
            elif kind in _AUTO_KINDS:
                selected_spec = _select_auto_candidate(
                    spec,
                    train,
                    method,
                    observed_column,
                    fold_column,
                    fit_kwargs,
                )
                selected_kind = _normalize_kind(selected_spec.get("kind", "monthly"))
                selected_fit_kwargs = _fit_kwargs_from_spec(selected_spec)
                window_size = selected_spec.get(
                    "window_size",
                    0 if selected_kind == "monthly" else 1,
                )
                selected_windows.setdefault(column, {})[fold] = int(window_size)
                selected_settings.setdefault(column, {})[fold] = selected_spec["setting"]
                if selected_kind == "monthly":
                    monthly_params = get_month_params(
                        train,
                        method=method,
                        window_size=window_size,
                        **selected_fit_kwargs,
                    )
                elif selected_kind in _SMOOTHED_MONTHLY_KINDS:
                    monthly_params = get_smoothed_month_params(
                        train,
                        method=method,
                        window_size=window_size,
                        smooth_window=selected_spec.get("smooth_window", window_size),
                        **selected_fit_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported auto candidate kind: {selected_spec.get('kind')}")
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
    metadata_keys = sorted({key for values in metric_metadata.values() for key in values})
    for key in metadata_keys:
        metrics.loc[:, key] = metrics.loc[:, "column"].map(
            lambda column: metric_metadata.get(column, {}).get(key)
        )
    if selected_windows:
        out.attrs["selected_windows"] = selected_windows
    if selected_settings:
        out.attrs["selected_settings"] = selected_settings
    if selected_candidates:
        out.attrs["selected_candidates"] = selected_candidates
    if selected_methods:
        out.attrs["selected_methods"] = selected_methods
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

"""Sensitivity-analysis helpers for total-data outputs."""

from pathlib import Path
import ast

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
MASTER_DATA_DIR = OUTPUT_DIR / "master_data"
PLOT_DIR = OUTPUT_DIR / "plots" / "sensitivity_analysis"
COV_OUTPUT_FILE = MASTER_DATA_DIR / "filtered_cov_by_parameter.csv"

DRIVING_CONDITION_THRESHOLDS_KMH = {
    "urban": 60.0,
    "rural": 80.0,
    "highway": 80.0,
}

DRIVING_CONDITION_VELOCITY_CLASSES = {
    "urban": {"CADC_urban", "WLTC_urban", "FTP"},
    "rural": {"CADC_rural", "WLTC_rural", "Ford_focus"},
    "highway": {"CADC_highway", "WLTC_highway", "HWFET"},
}

POLLUTANT_COLUMNS = {
    "co2": "total_co2_g",
    "nox": "total_nox_g",
    "pm": "total_pm_g",
    "fc": "total_fc",
}

DISTANCE_SPECIFIC_POLLUTANT_COLUMNS = {
    "co2": "co2_g_per_km",
    "nox": "nox_g_per_km",
    "pm": "pm_g_per_km",
    "fc": "fc_per_km",
}

PARAMETER_SYMBOLS = {
    "idm": {
        "max_acc": "$a$",
        "des_dec": "$b$",
        "len_veh": "$l$",
        "jam_dis_0": "$s_0$",
        "jam_dis_1": "$s_1$",
        "des_vel": "$v_0$",
        "safe_head": "$T$",
        "acc_exp": "$\\delta$",
    },
    "midm": {
        "max_acc": "$a$",
        "des_dec": "$b$",
        "len_veh": "$l$",
        "jam_dis_0": "$s_0$",
        "jam_dis_1": "$s_1$",
        "des_vel": "$v_0$",
        "safe_head": "$T$",
        "acc_exp": "$\\delta$",
    },
    "fvdm": {
        "V1": "$V_1$",
        "V2": "$V_2$",
        "C1": "$C_1$",
        "C2": "$C_2$",
        "sens_vel": "$\\kappa_1$",
        "sens_relvel": "$\\kappa_2$",
        "len_veh": "$l_2$",
    },
    "mfvdm": {
        "V1": "$V_1$",
        "V2": "$V_2$",
        "C1": "$C_1$",
        "C2": "$C_2$",
        "sens_vel": "$\\kappa_1$",
        "sens_relvel": "$\\kappa_2$",
        "len_veh": "$l_2$",
    },
}

LEADER_SPEED_PROFILE_ORDER = [
    "WLTC_urban",
    "FTP",
    "CADC_urban",
    "WLTC_rural",
    "CADC_rural",
    "WLTC_highway",
    "CADC_highway",
    "HWFET",
]


def model_total_data_path(model):
    """Return the master total-data CSV path for one model.

    Parameters
    ----------
    model : str
        Car-following model name, such as ``idm`` or ``mfvdm``.

    Returns
    -------
    pathlib.Path
        Path to ``outputs/master_data/{model}_total_data.csv``.
    """
    return MASTER_DATA_DIR / f"{model}_total_data.csv"


def load_parameter_ranges(yaml_file=None):
    """Load prescribed parameter ranges from parameters.yaml.

    Parameters
    ----------
    yaml_file : str, pathlib.Path, or None
        Optional path to the parameter file. When omitted, the repository-level
        ``parameters.yaml`` file is used.

    Returns
    -------
    dict
        Nested dictionary containing the parameter ranges for each model.
    """
    if yaml_file is None:
        yaml_file = REPO_DIR / "parameters.yaml"

    try:
        import yaml
    except ModuleNotFoundError:
        return _load_parameter_ranges_without_yaml(yaml_file)

    with open(yaml_file, "r", encoding="utf-8") as file:
        model_param_dict = yaml.safe_load(file)

    return model_param_dict["Parameter_range"]


def _load_parameter_ranges_without_yaml(yaml_file):
    """Parse the simple Parameter_range section when PyYAML is unavailable.

    Parameters
    ----------
    yaml_file : str or pathlib.Path
        Path to ``parameters.yaml``.

    Returns
    -------
    dict
        Nested dictionary with the same shape as the PyYAML-based loader.
    """
    ranges = {}
    in_section = False
    current_model = None

    with open(yaml_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line_without_comment = raw_line.split("#", 1)[0].rstrip()
            if not line_without_comment.strip():
                continue

            stripped = line_without_comment.strip()
            indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))

            if stripped == "Parameter_range:":
                in_section = True
                current_model = None
                continue

            if not in_section:
                continue

            if indent == 0 and stripped.endswith(":"):
                break

            if indent == 2 and stripped.endswith(":"):
                current_model = stripped[:-1]
                ranges[current_model] = {}
                continue

            if indent == 4 and current_model and ":" in stripped:
                parameter, value = stripped.split(":", 1)
                ranges[current_model][parameter.strip()] = ast.literal_eval(
                    value.strip()
                )

    return ranges


def filter_by_driving_condition(input_file, driving_condition):
    """Filter total-data rows by speed-derived driving condition.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Total-data CSV path, usually formatted as ``{model}_total_data.csv``.
    driving_condition : {"urban", "rural", "highway"}
        Driving condition to keep. The mask is evaluated for matching leader
        speed profiles and each parameter combination after excluding the first
        vehicle.

    Returns
    -------
    pandas.DataFrame
        Rows belonging to parameter combinations that satisfy the requested
        driving condition.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Total-data input file not found: {input_path}")

    data = pd.read_csv(input_path)

    condition = driving_condition.lower()
    if condition not in DRIVING_CONDITION_THRESHOLDS_KMH:
        valid_conditions = ", ".join(DRIVING_CONDITION_THRESHOLDS_KMH)
        raise ValueError(f"driving_condition must be one of: {valid_conditions}")

    required_columns = {
        "row_type",
        "model",
        "parameter",
        "parameter_value",
        "velocity_class",
        "engine_type",
        "vehicle",
        "max_speed_mps",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    group_columns = [
        "model",
        "parameter",
        "parameter_value",
        "velocity_class",
        "engine_type",
    ]

    vehicle_numbers = pd.to_numeric(data["vehicle"], errors="coerce")
    vehicle_rows = data[
        (data["row_type"] == "vehicle")
        & vehicle_numbers.notna()
        & (vehicle_numbers != 1)
        & data["velocity_class"].isin(DRIVING_CONDITION_VELOCITY_CLASSES[condition])
    ].copy()
    vehicle_rows["max_speed_kmh"] = vehicle_rows["max_speed_mps"] * 3.6

    grouped_speeds = vehicle_rows.groupby(group_columns)["max_speed_kmh"].agg(
        highest_vehicle_speed="max",
        lowest_vehicle_speed="min",
    )

    threshold = DRIVING_CONDITION_THRESHOLDS_KMH[condition]
    if condition in {"urban", "rural"}:
        matching_groups = grouped_speeds[
            grouped_speeds["highest_vehicle_speed"] < threshold
        ]
    else:
        matching_groups = grouped_speeds[
            grouped_speeds["highest_vehicle_speed"] > threshold
        ]

    if matching_groups.empty:
        return data.iloc[0:0].copy()

    filtered = data.merge(
        matching_groups.index.to_frame(index=False),
        on=group_columns,
        how="inner",
    )
    return filtered


def calculate_vehicle_averaged_distance_specific_values(filtered_data):
    """Calculate one distance-specific observation for each parameter value.

    Parameters
    ----------
    filtered_data : pandas.DataFrame
        Total-data rows after applying the driving-condition filter.

    Returns
    -------
    pandas.DataFrame
        Distance-specific emissions and fuel consumption averaged across the
        vehicles in each parameter/model/profile/engine group.

    Notes
    -----
    For each model/parameter value/profile/engine group, each vehicle's total
    emission or fuel consumption is divided by that vehicle's total distance.
    Those per-vehicle rates are then averaged to form the sensitivity
    observation for that parameter value.
    """
    required_columns = {
        "row_type",
        "model",
        "parameter",
        "parameter_value",
        "velocity_class",
        "engine_type",
        "vehicle",
        "total_distance_m",
        "total_co2_g",
        "total_nox_g",
        "total_pm_g",
        "total_fc",
    }
    missing_columns = required_columns.difference(filtered_data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Filtered data is missing required columns: {missing}")

    vehicle_rows = filtered_data[filtered_data["row_type"] == "vehicle"].copy()
    numeric_columns = [
        "parameter_value",
        "vehicle",
        "total_distance_m",
        "total_co2_g",
        "total_nox_g",
        "total_pm_g",
        "total_fc",
    ]
    for column in numeric_columns:
        vehicle_rows[column] = pd.to_numeric(vehicle_rows[column], errors="coerce")

    vehicle_rows = vehicle_rows.dropna(subset=numeric_columns)
    vehicle_rows = vehicle_rows[vehicle_rows["total_distance_m"] > 0].copy()
    vehicle_rows["total_distance_km"] = vehicle_rows["total_distance_m"] / 1000

    for pollutant, total_column in POLLUTANT_COLUMNS.items():
        rate_column = DISTANCE_SPECIFIC_POLLUTANT_COLUMNS[pollutant]
        vehicle_rows[rate_column] = (
            vehicle_rows[total_column] / vehicle_rows["total_distance_km"]
        )

    group_columns = [
        "model",
        "parameter",
        "parameter_value",
        "velocity_class",
        "engine_type",
    ]
    parameter_rows = []
    for group_values, group in vehicle_rows.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_values))
        row["vehicles_used"] = len(group)
        row["mean_vehicle_distance_km"] = group["total_distance_km"].mean()

        for pollutant, rate_column in DISTANCE_SPECIFIC_POLLUTANT_COLUMNS.items():
            row[rate_column] = group[rate_column].mean()

        if "fc_unit" in group.columns:
            units = sorted(group["fc_unit"].dropna().astype(str).unique())
            row["fc_unit"] = ";".join(units)

        parameter_rows.append(row)

    return pd.DataFrame(parameter_rows)


def calculate_filtered_cov_by_parameter(
    models=("idm", "midm", "fvdm", "mfvdm"),
    driving_conditions=("urban", "rural", "highway"),
    output_file=COV_OUTPUT_FILE,
):
    """Calculate filtered coefficient of variation by parameter.

    Parameters
    ----------
    models : tuple[str, ...]
        Models whose master total-data files should be analysed.
    driving_conditions : tuple[str, ...]
        Driving-condition filters to apply.
    output_file : str or pathlib.Path
        CSV path for the coefficient-of-variation summary.

    Returns
    -------
    pandas.DataFrame
        Filtered COV values by model, condition, leader profile, engine, and
        parameter.

    Notes
    -----
    Each coefficient of variation is calculated from distance-specific values.
    For each surviving parameter value, vehicle total emissions or fuel
    consumption are divided by vehicle total distance, averaged over vehicles,
    and then used as the observation for the sensitivity calculation.
    """
    rows = []

    for driving_condition in driving_conditions:
        for model in models:
            filtered_data = filter_by_driving_condition(
                model_total_data_path(model),
                driving_condition,
            )
            parameter_value_rows = calculate_vehicle_averaged_distance_specific_values(
                filtered_data
            )
            if parameter_value_rows.empty:
                continue

            group_columns = ["velocity_class", "engine_type", "parameter"]
            for group_values, group in parameter_value_rows.groupby(group_columns):
                velocity_class, engine_type, parameter = group_values
                row = {
                    "model": model,
                    "driving_condition": driving_condition,
                    "leader_speed_profile": velocity_class,
                    "engine_type": engine_type,
                    "parameter": parameter,
                    "n_parameter_values": len(group),
                }

                for pollutant, column in DISTANCE_SPECIFIC_POLLUTANT_COLUMNS.items():
                    valid_group = group[group[column].notna()]
                    distance_specific_values = valid_group[column]
                    mean = distance_specific_values.mean()
                    std = distance_specific_values.std(ddof=0)

                    row[f"{pollutant}_mean_per_km"] = mean
                    row[f"{pollutant}_std_per_km"] = std
                    row[f"{pollutant}_mean_total"] = mean
                    row[f"{pollutant}_std_total"] = std
                    row[f"{pollutant}_cov"] = (
                        std / mean
                        if len(distance_specific_values) > 0
                        and pd.notna(mean)
                        and mean != 0
                        else float("nan")
                    )
                    row[f"{pollutant}_cov_percent"] = row[f"{pollutant}_cov"] * 100

                if "fc_unit" in group.columns:
                    units = sorted(group["fc_unit"].dropna().astype(str).unique())
                    row["fc_unit"] = ";".join(units)

                rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            [
                "model",
                "driving_condition",
                "leader_speed_profile",
                "engine_type",
                "parameter",
            ]
        )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def plot_filtered_cov_by_pollutant(
    engine_type="D",
    model="idm",
    pollutant_key="CO2",
    input_file=COV_OUTPUT_FILE,
    output_dir=PLOT_DIR,
):
    """Plot filtered COV by parameter for one model, engine, and pollutant.

    Parameters
    ----------
    engine_type : str
        PHEMlight engine type to plot.
    model : str
        Car-following model to plot.
    pollutant_key : {"CO2", "NOx", "PM", "FC"}
        Pollutant or fuel-consumption metric.
    input_file : str or pathlib.Path
        Filtered COV summary CSV created by
        ``calculate_filtered_cov_by_parameter``.
    output_dir : str or pathlib.Path
        Directory where the PNG file is saved.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.

    Notes
    -----
    The plot uses the grouped data created by
    ``calculate_filtered_cov_by_parameter`` and excludes the Ford Focus leader
    profile.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    pollutant = pollutant_key.lower()
    cov_column = f"{pollutant}_cov"
    if cov_column not in {
        "co2_cov",
        "nox_cov",
        "pm_cov",
        "fc_cov",
    }:
        raise ValueError("pollutant_key must be one of: CO2, NOx, PM, FC")

    data = pd.read_csv(input_file)
    if cov_column not in data.columns:
        raise ValueError(f"{input_file} is missing column: {cov_column}")

    plot_data = data[
        (data["engine_type"] == engine_type)
        & (data["model"] == model)
        & (data["leader_speed_profile"] != "Ford_focus")
    ].copy()

    if plot_data.empty:
        raise ValueError(
            "No rows found for "
            f"model={model}, engine_type={engine_type}, pollutant={pollutant_key}"
        )

    parameter_order = (
        plot_data.groupby("parameter")[cov_column]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    parameter_labels = [
        PARAMETER_SYMBOLS.get(model, {}).get(parameter, parameter)
        for parameter in parameter_order
    ]

    speed_profiles = [
        profile
        for profile in LEADER_SPEED_PROFILE_ORDER
        if profile in set(plot_data["leader_speed_profile"])
    ]

    x = np.arange(len(parameter_order)) * 0.5
    markers = ["o", "s", "D", "^", "v", ">", "<", "P"]
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)

    for profile_id, profile in enumerate(speed_profiles):
        profile_data = plot_data[plot_data["leader_speed_profile"] == profile]
        cov_values = []

        for parameter in parameter_order:
            values = profile_data.loc[
                profile_data["parameter"] == parameter,
                cov_column,
            ]
            cov_values.append(values.mean() if not values.empty else np.nan)

        ax.scatter(
            x,
            cov_values,
            marker=markers[profile_id % len(markers)],
            s=80,
            color=cmap(profile_id % 10),
            label=profile.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(parameter_labels, rotation=0, ha="right", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.set_ylabel("COV", fontsize=18)
    ax.set_xlabel("Model Parameters", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.set_ylim(-0.02, 0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        fontsize=14,
        frameon=False,
    )

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / (
        f"{model}_{engine_type}_{pollutant_key}_filtered_cov_sensitivity.png"
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()

    return output_path


def plot_filtered_emission_by_parameter(
    engine_type="D",
    model="idm",
    pollutant_key="CO2",
    output_dir=PLOT_DIR,
):
    """Plot filtered average distance-specific emission against each parameter value.

    Parameters
    ----------
    engine_type : str
        PHEMlight engine type to plot.
    model : str
        Car-following model to plot.
    pollutant_key : {"CO2", "NOx", "PM", "FC"}
        Pollutant or fuel-consumption metric.
    output_dir : str or pathlib.Path
        Directory where the PNG file is saved.

    Returns
    -------
    pathlib.Path
        Path to the saved PNG file.

    Notes
    -----
    For one model, engine type, and pollutant, this function applies the
    appropriate filter for urban, rural, and highway leader speed profiles, then
    plots all eight standard leader profiles in one figure. Ford Focus is
    excluded from the plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import string

    pollutant = pollutant_key.lower()
    if pollutant not in POLLUTANT_COLUMNS:
        raise ValueError("pollutant_key must be one of: CO2, NOx, PM, FC")

    parameter_ranges = load_parameter_ranges()

    # The sensitivity analysis uses the master files because they retain the
    # parameter and parameter-value columns needed for grouping.
    filtered_data = pd.concat(
        [
            filter_by_driving_condition(model_total_data_path(model), condition)
            for condition in DRIVING_CONDITION_THRESHOLDS_KMH
        ],
        ignore_index=True,
    )
    plot_data = calculate_vehicle_averaged_distance_specific_values(filtered_data)
    plot_data = plot_data[
        (plot_data["engine_type"] == engine_type)
        & (plot_data["velocity_class"] != "Ford_focus")
    ].copy()
    plot_data = plot_data[
        plot_data[DISTANCE_SPECIFIC_POLLUTANT_COLUMNS[pollutant]].notna()
    ]

    if plot_data.empty:
        raise ValueError(
            "No rows found for "
            f"model={model}, engine_type={engine_type}, pollutant={pollutant_key}"
        )

    raw_emission_column = DISTANCE_SPECIFIC_POLLUTANT_COLUMNS[pollutant]
    max_abs_emission = plot_data[raw_emission_column].abs().max()
    emission_exponent = (
        int(np.floor(np.log10(max_abs_emission)))
        if pd.notna(max_abs_emission) and max_abs_emission > 0
        else 0
    )
    emission_scale = 10 ** emission_exponent

    emission_column = f"{pollutant}_scaled_total"
    plot_data[emission_column] = plot_data[raw_emission_column] / emission_scale

    parameter_order = (
        plot_data.groupby("parameter")[emission_column]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    speed_profiles = [
        profile
        for profile in LEADER_SPEED_PROFILE_ORDER
        if profile in set(plot_data["velocity_class"])
    ]

    n_cols = 2
    n_rows = int(np.ceil(len(speed_profiles) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(9, 2.2 * n_rows),
        sharex=False,
        dpi=300,
    )
    axes = np.atleast_1d(axes).flatten()

    cmap = plt.get_cmap("tab10")
    colors = {
        parameter: cmap(parameter_id % 10)
        for parameter_id, parameter in enumerate(parameter_order)
    }
    markers = ["o", "s", "D", "^", "v", ">", "<", "P"]
    subplot_labels = list(string.ascii_lowercase)

    for profile_id, (ax, speed_profile) in enumerate(zip(axes, speed_profiles)):
        profile_data = plot_data[plot_data["velocity_class"] == speed_profile]

        for parameter_id, parameter in enumerate(parameter_order):
            parameter_data = profile_data[
                profile_data["parameter"] == parameter
            ].copy()
            if parameter_data.empty:
                continue

            try:
                min_value, max_value = parameter_ranges[model][parameter]
            except KeyError as exc:
                raise ValueError(
                    f"No prescribed range found in parameters.yaml for "
                    f"model={model}, parameter={parameter}"
                ) from exc

            parameter_data["parameter_value"] = pd.to_numeric(
                parameter_data["parameter_value"],
                errors="coerce",
            )
            if max_value == min_value:
                parameter_data["norm_val"] = 0.0
            else:
                parameter_data["norm_val"] = (
                    (parameter_data["parameter_value"] - min_value)
                    / (max_value - min_value)
                )

            parameter_data = parameter_data.sort_values("norm_val")
            ax.plot(
                parameter_data["norm_val"],
                parameter_data[emission_column],
                marker=markers[parameter_id % len(markers)],
                linestyle="--",
                color=colors[parameter],
                label=PARAMETER_SYMBOLS.get(model, {}).get(parameter, parameter),
            )

        ax.text(
            0.06,
            1.05,
            f"{subplot_labels[profile_id]}.",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        ax.set_title(speed_profile.replace("_", " "), fontsize=13)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    for ax in axes[len(speed_profiles):]:
        ax.axis("off")

    fc_unit = ""
    if pollutant == "fc" and "fc_unit" in plot_data.columns:
        units = sorted(plot_data["fc_unit"].dropna().astype(str).unique())
        fc_unit = ";".join(units) if units else ""

    scale_label = rf"$\times 10^{{{emission_exponent}}}$"

    y_label = (
        f"{pollutant_key} consumption ({scale_label} {fc_unit}/km)"
        if pollutant == "fc"
        else f"{pollutant_key} emissions ({scale_label} g/km)"
    )
    fig.text(0.5, 0.055, "Normalized Parameter Value", ha="center", fontsize=14)
    fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=min(6, max(1, len(parameter_order))),
        fontsize=13,
        frameon=False,
    )

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / (
        f"{model}_{engine_type}_{pollutant_key}_all_profiles"
        "_filtered_emission_parameters.png"
    )
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.95])
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()

    return output_path


if __name__ == "__main__":
    models_list = ["idm", "midm", "fvdm", "mfvdm"]
    engine_list = ["G", "D", "G_HEV", "D_HEV", "B"]
    pollutant_list = ["CO2", "NOx", "PM", "FC"]
    cov_file = COV_OUTPUT_FILE

    calculate_filtered_cov_by_parameter(
        models=models_list,
        output_file=cov_file,
    )

    for model_name in models_list:
        for engine in engine_list:
            for pollutant in pollutant_list:
                try:
                    plot_filtered_cov_by_pollutant(
                        engine_type=engine,
                        model=model_name,
                        pollutant_key=pollutant,
                        input_file=cov_file,
                        output_dir=PLOT_DIR,
                    )
                except ValueError:
                    continue

    for model_name in models_list:
        for engine in engine_list:
            for pollutant in pollutant_list:
                try:
                    plot_filtered_emission_by_parameter(
                        engine_type=engine,
                        model=model_name,
                        pollutant_key=pollutant,
                        output_dir=PLOT_DIR,
                    )
                except ValueError:
                    continue

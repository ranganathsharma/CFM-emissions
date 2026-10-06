"""
Plot fleet-average emissions after filtering valid parameter combinations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_DIR / 'outputs'
MASTER_DATA_DIR = OUTPUT_DIR / 'master_data'
PLOT_DIR = OUTPUT_DIR / 'plots/fleet_emissions'
TOTAL_EVALUATION_START_S = 800
TOTAL_EVALUATION_END_S = 2600


def get_fuel_density_g_per_l(engine_type):
    """
    Description:
        Return the fuel density used to convert fuel mass to volume.

    Inputs:
        engine_type: Powertrain or fuel type code used to select units and conversion factors.

    Outputs:
        Requested value for the supplied inputs.
    """
    if engine_type in ['G', 'G_HEV']:
        return 700
    if engine_type in ['D', 'D_HEV']:
        return 800
    return None


def speed_rule_passes(velocity_class, max_speed_kmh):
    """
    Description:
        Check whether a parameter row satisfies the speed rule for the drive cycle.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        max_speed_kmh: Maximum speed in km/h used by the validity filter.

    Outputs:
        Calculated value from the function.
    """
    if velocity_class in ['WLTC_urban', 'CADC_urban', 'FTP']:
        return max_speed_kmh < 60
    if velocity_class in ['WLTC_rural', 'CADC_rural']:
        return max_speed_kmh < 80
    if velocity_class in ['WLTC_highway', 'CADC_highway', 'HWFET']:
        return max_speed_kmh > 80
    return True


def read_parameter_data(model):
    """
    Description:
        Read and validate the master total-data file for one model.

    Inputs:
        model: Model name used to select data or label output.

    Outputs:
        Validated dataframe read from the input file.
    """
    input_file = MASTER_DATA_DIR / f'{model}_total_data.csv'
    if not input_file.exists():
        raise FileNotFoundError(f'Missing parameter total-data file for {model}: {input_file}')

    df = pd.read_csv(input_file)
    required_columns = [
        'row_type', 'model', 'velocity_class', 'engine_type', 'parameter', 'parameter_value',
        'vehicle', 'total_distance_m', 'max_speed_mps', 'total_co2_g', 'total_nox_g',
        'total_pm_g', 'total_fc',
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f'{input_file} cannot be used for parameter filtering; missing columns: {missing_columns}'
        )

    numeric_columns = [
        'parameter_value', 'vehicle', 'total_distance_m', 'max_speed_mps',
        'total_co2_g', 'total_nox_g', 'total_pm_g', 'total_fc',
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df.dropna(subset=numeric_columns)


def distance_specific_totals(group_df, engine_type):
    """
    Description:
        Calculate fleet distance-specific emissions and fuel consumption for one grouped case.

    Inputs:
        group_df: Grouped dataframe used for one aggregate calculation.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.

    Outputs:
        Calculated value from the function.
    """
    distance_km = group_df['total_distance_m'].sum() / 1000
    if distance_km <= 0:
        return None

    fc_per_km = group_df['total_fc'].sum() / distance_km
    fuel_density = get_fuel_density_g_per_l(engine_type)
    if fuel_density is not None:
        fc_per_km /= fuel_density

    return {
        'co2_g_per_km': group_df['total_co2_g'].sum() / distance_km,
        'nox_g_per_km': group_df['total_nox_g'].sum() / distance_km,
        'pm_g_per_km': group_df['total_pm_g'].sum() / distance_km,
        'fc_per_km': fc_per_km,
    }


def get_valid_parameter_rows(model_names, vehicle_count=200):
    """
    Description:
        Filter parameter rows to the combinations that satisfy the manuscript validity rules.

    Inputs:
        model_names: List of model names to include in the calculation or plot.
        vehicle_count: Maximum vehicle number to include.

    Outputs:
        Requested value for the supplied inputs.
    """
    group_columns = ['model', 'velocity_class', 'engine_type', 'parameter', 'parameter_value']
    rows = []

    for model in model_names:
        df = read_parameter_data(model)
        df = df[(df['row_type'] == 'vehicle') & (df['vehicle'].between(1, vehicle_count))]

        for group_values, group_df in df.groupby(group_columns, dropna=False, sort=False):
            group_df = group_df.sort_values('vehicle').head(vehicle_count)
            if len(group_df) != vehicle_count:
                continue

            group = dict(zip(group_columns, group_values))
            speed_mask_df = group_df[group_df['vehicle'] != 1]
            max_speed_kmh = (speed_mask_df if not speed_mask_df.empty else group_df)['max_speed_mps'].max() * 3.6
            if not speed_rule_passes(group['velocity_class'], max_speed_kmh):
                continue

            totals = distance_specific_totals(group_df, group['engine_type'])
            if totals is not None:
                rows.append({**group, 'max_speed_kmh': max_speed_kmh, **totals})

    if not rows:
        raise ValueError(f'No valid parameter combinations found for {model_names}')

    return pd.DataFrame(rows)


def summarize_valid_parameters(parameter_df):
    """
    Description:
        Summarize valid parameter rows by model, drive cycle, and engine type.

    Inputs:
        parameter_df: Input value used by this function for parameter df.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    pollutants = ['co2_g_per_km', 'nox_g_per_km', 'pm_g_per_km', 'fc_per_km']
    rows = []

    for group_values, group_df in parameter_df.groupby(['model', 'velocity_class', 'engine_type'], sort=False):
        row = dict(zip(['model', 'velocity_class', 'engine_type'], group_values))
        row['valid_parameter_combinations'] = len(group_df)
        for pollutant in pollutants:
            row[f'{pollutant}_mean'] = group_df[pollutant].mean()
            row[f'{pollutant}_std'] = group_df[pollutant].std(ddof=0)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_valid_parameter_emission_distance_bars(
        velocity_class,
        model_names=None,
        engine_order=None,
        input_dir=MASTER_DATA_DIR,
        output_dir=PLOT_DIR,
        vehicle_count=200,
        urban_conditions=None,
        rural_conditions=None,
        highway_conditions=None,
        save_csv=False):
    """
    Create a create_plots_5-style horizontal bar plot for one leader profile.

    For each model, powertrain, and parameter value, the speed-rule mask is
    applied first. The plotted quantity is then calculated as
    sum(total emission) / sum(total distance) over vehicles 1..vehicle_count
    for the selected 30-minute journey window already saved in total-data CSVs.
    Error bars show the standard deviation across valid parameter combinations.
    Vehicle 0, when available, is shown as the leader baseline marker.
    """
    if model_names is None:
        model_names = ['fvdm', 'idm', 'mfvdm', 'midm']
    if engine_order is None:
        engine_order = ['D', 'D_HEV', 'G', 'G_HEV']
    if urban_conditions is None:
        urban_conditions = ['WLTC_urban', 'CADC_urban', 'FTP']
    if rural_conditions is None:
        rural_conditions = ['WLTC_rural', 'CADC_rural']
    if highway_conditions is None:
        highway_conditions = ['WLTC_highway', 'CADC_highway', 'HWFET']

    script_dir = Path(__file__).resolve().parent
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    if not input_dir.exists() and (Path('Emissions') / input_dir).exists():
        input_dir = Path('Emissions') / input_dir
    if not output_dir.exists() and (Path('Emissions') / output_dir).parent.exists():
        output_dir = Path('Emissions') / output_dir

    model_labels = {
        'idm': 'IDM',
        'fvdm': 'FVDM',
        'midm': 'MIDM',
        'mfvdm': 'MFVDM',
    }
    model_colors = {
        'fvdm': 'cornflowerblue',
        'idm': 'crimson',
        'mfvdm': 'gold',
        'midm': 'forestgreen',
    }
    engine_labels = {
        'G': 'G',
        'D': 'D',
        'G_HEV': 'GHEV',
        'D_HEV': 'DHEV',
        'B': 'B',
    }
    fc_axis_label = 'FC (L/km)'
    if engine_order and 'B' in engine_order:
        fc_axis_label = 'FC (L/km; B: kWh/km)'

    pollutants = [
        ('co2_g_per_km', 'CO2', 'CO2 (g/km)', 'total_co2_g'),
        ('nox_g_per_km', 'NOx', 'NOx (g/km)', 'total_nox_g'),
        ('pm_g_per_km', 'PM', r'PM ($10^{-4}$ g/km)', 'total_pm_g'),
        ('fc_per_km', 'FC', r'FC ($10^{-1}$ L/km)', 'total_fc'),
    ]
    axis_config = {
        'co2_g_per_km': {
            'scale': 1,
            'xlim': (70, 280),
            'xticks': [70, 140, 210, 280],
        },
        'nox_g_per_km': {
            'scale': 1,
            'split_by_fuel': True,
            'diesel_xlim': (0.04, 0.65),
            'diesel_xticks': [0.1, 0.3, 0.5],
            'gasoline_xlim': (0.02, 0.055),
            'gasoline_xticks': [0.02, 0.04],
        },
        'pm_g_per_km': {
            'scale': 1e-4,
            'split_by_fuel': True,
            'diesel_xlim': (0, 20),
            'diesel_xticks': [0, 10, 20],
            'gasoline_xlim': (5, 35),
            'gasoline_xticks': [5, 15, 25, 35],
        },
        'fc_per_km': {
            'scale': 1e-1,
            'xlim': (0.035 / 1e-1, 0.135 / 1e-1),
            'xticks': [0.4, 0.8, 1.2],
        },
    }
    diesel_engine_types = {'D', 'D_HEV'}
    gasoline_engine_types = {'G', 'G_HEV'}

    required_columns = [
        'row_type',
        'model',
        'parameter',
        'parameter_value',
        'velocity_class',
        'engine_type',
        'vehicle',
        'total_distance_m',
        'max_speed_mps',
        'total_co2_g',
        'total_nox_g',
        'total_pm_g',
        'total_fc',
    ]
    numeric_columns = [
        'parameter_value',
        'vehicle',
        'total_distance_m',
        'max_speed_mps',
        'total_co2_g',
        'total_nox_g',
        'total_pm_g',
        'total_fc',
    ]
    group_columns = ['model', 'velocity_class', 'engine_type', 'parameter', 'parameter_value']

    def speed_rule_passes(max_speed_kmh):
        """
        Description:
            Check whether a parameter row satisfies the speed rule for the drive cycle.

        Inputs:
            max_speed_kmh: Maximum speed in km/h used by the validity filter.

        Outputs:
            Calculated value from the function.
        """
        if velocity_class in urban_conditions:
            return max_speed_kmh < 60
        if velocity_class in rural_conditions:
            return max_speed_kmh < 80
        if velocity_class in highway_conditions:
            return max_speed_kmh > 80
        return True

    def distance_specific_value(total_value, total_distance_km, metric_column, engine_type):
        """
        Description:
            Convert a total quantity into a distance-specific value.

        Inputs:
            total_value: Input value used by this function for total value.
            total_distance_km: Input value used by this function for total distance km.
            metric_column: Input value used by this function for metric column.
            engine_type: Powertrain or fuel type code used to select units and conversion factors.

        Outputs:
            Calculated value from the function.
        """
        value = total_value / total_distance_km
        if metric_column != 'fc_per_km':
            return value

        fuel_density = get_fuel_density_g_per_l(engine_type)
        if fuel_density is None:
            return value
        return value / fuel_density

    parameter_rows = []
    leader_rows = []
    for model in model_names:
        input_path = input_dir / f'{model}_total_data.csv'
        if not input_path.exists():
            print(f'Missing total-data file: {input_path}')
            continue

        df = pd.read_csv(input_path, usecols=lambda column: column in required_columns)
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f'{input_path} is missing columns: {missing_columns}')

        df = df[df['row_type'] == 'vehicle'].copy()
        df = df[df['velocity_class'] == velocity_class].copy()
        df = df[df['engine_type'].isin(engine_order)].copy()
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df.dropna(subset=numeric_columns)

        follower_df = df[
            (df['vehicle'] >= 1)
            & (df['vehicle'] <= vehicle_count)
            & (df['total_distance_m'] > 0)
        ].copy()

        for group_values, group_df in follower_df.groupby(group_columns, dropna=False, sort=False):
            group_df = group_df.sort_values('vehicle').head(vehicle_count)
            if len(group_df) != vehicle_count:
                continue

            speed_mask_df = group_df[group_df['vehicle'] != 1]
            if speed_mask_df.empty:
                speed_mask_df = group_df
            max_speed_kmh = speed_mask_df['max_speed_mps'].max() * 3.6
            if not speed_rule_passes(max_speed_kmh):
                continue

            group = dict(zip(group_columns, group_values))
            total_distance_km = group_df['total_distance_m'].sum() / 1000
            if total_distance_km <= 0:
                continue

            row = {
                **group,
                'vehicles_used': len(group_df),
                'max_speed_kmh': max_speed_kmh,
                'total_distance_km': total_distance_km,
            }
            for metric_column, _, _, total_column in pollutants:
                row[metric_column] = distance_specific_value(
                    group_df[total_column].sum(),
                    total_distance_km,
                    metric_column,
                    group['engine_type'],
                )
            parameter_rows.append(row)

            leader_df = df[
                (df['vehicle'] == 0)
                & (df['model'] == group['model'])
                & (df['engine_type'] == group['engine_type'])
                & (df['parameter'] == group['parameter'])
                & (df['parameter_value'] == group['parameter_value'])
                & (df['total_distance_m'] > 0)
            ]
            if leader_df.empty:
                continue

            leader_row = leader_df.iloc[0]
            leader_distance_km = leader_row['total_distance_m'] / 1000
            if leader_distance_km <= 0:
                continue

            baseline_row = {
                **group,
                'vehicle': 0,
                'total_distance_km': leader_distance_km,
            }
            for metric_column, _, _, total_column in pollutants:
                baseline_row[metric_column] = distance_specific_value(
                    leader_row[total_column],
                    leader_distance_km,
                    metric_column,
                    group['engine_type'],
                )
            leader_rows.append(baseline_row)

    parameter_df = pd.DataFrame(parameter_rows)
    if parameter_df.empty:
        raise ValueError(f'No valid parameter combinations found for {velocity_class}')

    summary_rows = []
    for group_values, group_df in parameter_df.groupby(['model', 'velocity_class', 'engine_type'], sort=False):
        group = dict(zip(['model', 'velocity_class', 'engine_type'], group_values))
        row = {
            **group,
            'valid_parameter_combinations': len(group_df),
        }
        for metric_column, _, _, _ in pollutants:
            row[f'{metric_column}_mean'] = group_df[metric_column].mean()
            row[f'{metric_column}_std'] = group_df[metric_column].std(ddof=0)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    leader_df = pd.DataFrame(leader_rows)
    baseline_df = pd.DataFrame()
    if not leader_df.empty:
        baseline_rows = []
        for engine_type, group_df in leader_df.groupby('engine_type', sort=False):
            row = {'engine_type': engine_type}
            for metric_column, _, _, _ in pollutants:
                row[f'{metric_column}_mean'] = group_df[metric_column].mean()
            baseline_rows.append(row)
        baseline_df = pd.DataFrame(baseline_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_csv:
        summary_df.to_csv(
            output_dir / f'{velocity_class}_valid_parameter_emission_distance_bar_summary.csv',
            index=False,
        )
        parameter_df.to_csv(
            output_dir / f'{velocity_class}_valid_parameter_emission_distance_bar_parameters.csv',
            index=False,
        )
        if not baseline_df.empty:
            baseline_df.to_csv(
                output_dir / f'{velocity_class}_valid_parameter_emission_distance_bar_leader.csv',
                index=False,
            )

    fig, axes = plt.subplots(1, len(pollutants), figsize=(10, 3), dpi=600, sharey=True)
    axes = np.asarray(axes).reshape(len(pollutants))
    bar_width = 1 / (len(model_names) + 1)

    for ax_index, (ax, (metric_column, title, xlabel, _)) in enumerate(zip(axes, pollutants)):
        config = axis_config[metric_column]
        display_scale = config['scale']

        def get_fuel_axis_group(engine_type):
            """
            Description:
                Choose the display scaling group for a fuel-consumption axis.

            Inputs:
                engine_type: Powertrain or fuel type code used to select units and conversion factors.

            Outputs:
                Requested value for the supplied inputs.
            """
            if engine_type in diesel_engine_types:
                return 'diesel'
            if engine_type in gasoline_engine_types:
                return 'gasoline'
            return 'gasoline'

        def to_display_axis(value, err_value, engine_type):
            """
            Description:
                Convert raw fuel-consumption values to the selected display scale.

            Inputs:
                value: Input value used by this function for value.
                err_value: Input value used by this function for err value.
                engine_type: Powertrain or fuel type code used to select units and conversion factors.

            Outputs:
                Calculated value from the function.
            """
            scaled_value = value / display_scale
            scaled_err = err_value / display_scale
            if not config.get('split_by_fuel', False):
                return scaled_value, scaled_err

            fuel_group = get_fuel_axis_group(engine_type)
            x_min, x_max = config[f'{fuel_group}_xlim']
            x_range = x_max - x_min
            return (scaled_value - x_min) / x_range, scaled_err / x_range

        def format_tick_label(tick):
            """
            Description:
                Format one axis tick label for display.

            Inputs:
                tick: Input value used by this function for tick.

            Outputs:
                Calculated value from the function.
            """
            return f'{tick:g}'

        def normalized_ticks(fuel_group):
            """
            Description:
                Create normalized tick positions for a plotted axis.

            Inputs:
                fuel_group: Input value used by this function for fuel group.

            Outputs:
                Calculated value from the function.
            """
            x_min, x_max = config[f'{fuel_group}_xlim']
            x_range = x_max - x_min
            ticks = config[f'{fuel_group}_xticks']
            return [(tick - x_min) / x_range for tick in ticks], ticks

        for engine_index, engine_type in enumerate(engine_order):
            for model_index, model in enumerate(model_names):
                row = summary_df[
                    (summary_df['model'] == model)
                    & (summary_df['engine_type'] == engine_type)
                ]
                if row.empty:
                    continue

                mean_val, err_val = to_display_axis(
                    row.iloc[0][f'{metric_column}_mean'],
                    row.iloc[0][f'{metric_column}_std'],
                    engine_type,
                )
                y_position = engine_index + (model_index - (len(model_names) - 1) / 2) * bar_width

                ax.barh(
                    y_position,
                    mean_val,
                    xerr=err_val,
                    ecolor='k',
                    height=bar_width,
                    error_kw={'elinewidth': 1, 'capthick': 1, 'capsize': 1.5},
                    color=model_colors.get(model, None),
                    label=model_labels.get(model, model.upper()) if (ax_index == 0 and engine_index == 0) else '',
                )

            if not baseline_df.empty:
                baseline_row = baseline_df[baseline_df['engine_type'] == engine_type]
                if not baseline_row.empty:
                    baseline_value, _ = to_display_axis(
                        baseline_row.iloc[0][f'{metric_column}_mean'],
                        0,
                        engine_type,
                    )
                    ax.scatter(
                        baseline_value,
                        engine_index,
                        color='k',
                        marker='|',
                        s=300,
                        label='Leader' if (ax_index == 0 and engine_index == 0) else '',
                    )

        ax.set_yticks(range(len(engine_order)))
        ax.set_yticklabels(
            [engine_labels.get(engine_type, engine_type) for engine_type in engine_order],
            fontsize=15,
        )
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

        if config.get('split_by_fuel', False):
            fuel_groups_in_order = [get_fuel_axis_group(engine_type) for engine_type in engine_order]
            top_fuel_group = fuel_groups_in_order[0]
            bottom_fuel_group = fuel_groups_in_order[-1]
            separator_index = next(
                (
                    index
                    for index in range(1, len(fuel_groups_in_order))
                    if fuel_groups_in_order[index] != fuel_groups_in_order[index - 1]
                ),
                len(fuel_groups_in_order),
            )
            separator_y = separator_index - 0.5

            ax.set_xlim(0, 1)
            bottom_tick_positions, bottom_ticks = normalized_ticks(bottom_fuel_group)
            ax.set_xticks(bottom_tick_positions)
            ax.set_xticklabels([format_tick_label(tick) for tick in bottom_ticks], rotation=0)
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_label_position('bottom')
            ax.tick_params(axis='x', which='major', labelbottom=True, labeltop=False)
            ax.grid(False)
            ax.axhline(separator_y, color='k', linewidth=0.9, alpha=0.85)
            for tick_position in bottom_tick_positions:
                ax.vlines(
                    tick_position,
                    separator_y,
                    len(engine_order) - 0.5,
                    colors='0.75',
                    linestyles='--',
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=0,
                )

            top_ax = ax.twiny()
            top_ax.set_xlim(0, 1)
            top_tick_positions, top_ticks = normalized_ticks(top_fuel_group)
            top_ax.set_xticks(top_tick_positions)
            top_ax.set_xticklabels([format_tick_label(tick) for tick in top_ticks], rotation=0)
            top_ax.tick_params(axis='x', which='major', labelsize=15)
            top_ax.xaxis.set_ticks_position('top')
            top_ax.xaxis.set_label_position('top')
            top_ax.tick_params(axis='x', which='major', labeltop=True, labelbottom=False)
            top_ax.minorticks_off()
            for tick_position in top_tick_positions:
                ax.vlines(
                    tick_position,
                    -0.5,
                    separator_y,
                    colors='0.75',
                    linestyles='--',
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=0,
                )
        else:
            ax.set_xlim(*config['xlim'])
            ax.set_xticks(config['xticks'])
            ax.set_xticklabels([format_tick_label(tick) for tick in config['xticks']], rotation=0)
            ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_xlabel(xlabel, fontsize=18)

    fig.text(0.995, 0.5, 'Engine Type', va='center', rotation=-90, fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 0),
            ncol=4,
            fontsize=18,
            frameon=False,
        )

    output_path = output_dir / f'{velocity_class}_valid_parameter_emission_distance_bars.png'
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(output_path, bbox_inches='tight', dpi=600)
    plt.close(fig)
    return output_path, summary_df, parameter_df


def plot_valid_parameter_emission_distance_bars_for_all_profiles(
        velocity_class_list=None,
        **kwargs):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        velocity_class_list: Drive-cycle names to process.
        **kwargs: Input value used by this function for kwargs.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    if velocity_class_list is None:
        velocity_class_list = [
            'WLTC_urban',
            'CADC_urban',
            'FTP',
            'WLTC_rural',
            'CADC_rural',
            'WLTC_highway',
            'CADC_highway',
            'HWFET',
            'Ford_focus'
        ]

    output_paths = []
    summaries = []
    parameter_data = []
    for velocity_class in velocity_class_list:
        output_path, summary_df, parameter_df = plot_valid_parameter_emission_distance_bars(
            velocity_class,
            **kwargs,
        )
        output_paths.append(output_path)
        summaries.append(summary_df)
        parameter_data.append(parameter_df)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    parameter_df = pd.concat(parameter_data, ignore_index=True) if parameter_data else pd.DataFrame()
    return output_paths, summary_df, parameter_df


if __name__ == '__main__':
    plot_valid_parameter_emission_distance_bars_for_all_profiles()

"""
This file contains the codes to generate the emission rate histogram plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent
INSTANTANEOUS_DIR = RESULTS_DIR / 'outputs/instantaneous_data'
PLOT_DIR = RESULTS_DIR / 'outputs/plots/histograms'
START_SPEED_TOL = 1e-9
TOTAL_EVALUATION_START_S = 800
TOTAL_EVALUATION_END_S = 2600


def keep_rows_after_vehicle_start(df):
    """
    Description:
        Keep data only after each vehicle has started moving.

    Inputs:
        df: Input dataframe used by the function.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    rows = []

    for _, vehicle_df in df.groupby('vehicle', sort=False):
        vehicle_df = keep_rows_after_start(vehicle_df, 'speed')
        if vehicle_df.empty:
            continue

        rows.append(vehicle_df)

    if not rows:
        return df.iloc[0:0].copy()

    return pd.concat(rows, ignore_index=True)


def keep_rows_after_start(df, speed_column):
    """
    Description:
        Keep rows from the first nonzero-speed instant onward.

    Inputs:
        df: Input dataframe used by the function.
        speed_column: Name of the speed column used to detect when movement starts.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    moving_rows = df[df[speed_column].abs() > START_SPEED_TOL]
    if moving_rows.empty:
        return df.iloc[0:0].copy()

    start_time = moving_rows['time'].iloc[0]
    return df[df['time'] >= start_time].copy()


def read_instantaneous_data(model, velocity_class, engine_type):
    """
    Description:
        Read one instantaneous emissions CSV and prepare numeric columns for plotting.

    Inputs:
        model: Model name used to select data or label output.
        velocity_class: Leader speed profile or drive-cycle name.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.

    Outputs:
        Validated dataframe read from the input file.
    """
    input_file = INSTANTANEOUS_DIR / f'{model}_{velocity_class}_{engine_type}_instantaneous_outputs.csv'
    if not input_file.exists():
        raise FileNotFoundError(f'Missing instantaneous validation file: {input_file}')

    df = pd.read_csv(input_file)
    df.columns = [column.strip().lower() for column in df.columns]

    for column in ['vehicle', 'time', 'speed', 'co2', 'fc', 'nox', 'pm', 'position', 'co2_g', 'fc_g', 'nox_g', 'pm_g']:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=['vehicle', 'time', 'speed', 'fc', 'position'])
    return keep_rows_after_vehicle_start(df)


def get_fc_label(engine_type):
    """
    Description:
        Return the fuel-consumption axis label for the selected powertrain.

    Inputs:
        engine_type: Powertrain or fuel type code used to select units and conversion factors.

    Outputs:
        Requested value for the supplied inputs.
    """
    if engine_type == 'B':
        return 'FC (kWh/h)'
    return 'FC (L/s)'


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


def get_quantity_scale_factor(plot_quantity, engine_type):
    """
    Description:
        Return the scale factor needed to display a plotted quantity in the requested units.

    Inputs:
        plot_quantity: Quantity name whose plotting scale factor is required.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.

    Outputs:
        Requested value for the supplied inputs.
    """
    if plot_quantity == 'fc' and engine_type == 'B':
        return 1
    if plot_quantity == 'fc':
        fuel_density = get_fuel_density_g_per_l(engine_type)
        if fuel_density is None:
            return 1 / 3600
        return 1 / (3600 * fuel_density)
    return 1 / 3600 if plot_quantity in ['co2', 'nox', 'pm'] else 1


def calculate_vsp(time_values, speed_values):
    """
    Description:
        Calculate vehicle specific power from speed and time series data.

    Inputs:
        time_values: Time series values used to calculate time steps.
        speed_values: Speed series values used to calculate acceleration and VSP.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    acceleration = speed_values.diff() / time_values.diff()
    acceleration = acceleration.replace([float('inf'), float('-inf')], 0).fillna(0)
    return speed_values * (1.1 * acceleration + 0.132) + 0.000302 * speed_values**3


def get_available_combinations():
    """
    Description:
        List the model, drive-cycle, and engine combinations available in the output files.

    Inputs:
        None.

    Outputs:
        Requested value for the supplied inputs.
    """
    combinations = []

    for input_file in sorted(INSTANTANEOUS_DIR.glob('*_instantaneous_outputs.csv')):
        stem = input_file.name.removesuffix('_instantaneous_outputs.csv')
        engine_type = next(
            engine for engine in ['G_HEV', 'D_HEV', 'G', 'D', 'B']
            if stem.endswith(f'_{engine}')
        )
        prefix = stem[:-(len(engine_type) + 1)]
        model, velocity_class = prefix.split('_', 1)
        combinations.append((model, velocity_class, engine_type))

    return combinations


def plot_model_quantity_histograms(
        velocity_class,
        engine_type,
        model_names=None,
        vehicle_list=None,
        bins=50,
        density=False):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.
        model_names: List of model names to include in the calculation or plot.
        vehicle_list: Follower vehicle numbers to include, excluding the leader.
        bins: Number of histogram bins.
        density: Whether histogram values should be plotted as density instead of relative frequency.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    if model_names is None:
        model_names = ['idm', 'midm']
    if vehicle_list is None:
        vehicle_list = [50, 100, 200]

    plot_quantities = [
        ('speed', 'Velocity (m/s)'),
        ('vsp', 'VSP (W/kg)'),
        ('fc', get_fc_label(engine_type)),
    ]
    # Manuscript figures use the leader plus these three follower vehicles.
    row_vehicles = [0] + vehicle_list
    row_labels = ['Leader'] + [f'Vehicle {vehicle}' for vehicle in vehicle_list]
    model_colors = {
        'idm': '#1f77b4',
        'fvdm': '#16BE16',
        'midm': '#d62728',
        'mfvdm': '#7511d3',
    }
    model_data = {
        model: read_instantaneous_data(model, velocity_class, engine_type)
        for model in model_names
        if (INSTANTANEOUS_DIR / f'{model}_{velocity_class}_{engine_type}_instantaneous_outputs.csv').exists()
    }

    if not model_data:
        raise ValueError(f'No instantaneous data found for {velocity_class} {engine_type}')

    def get_values(df, vehicle, column):
        """
        Description:
            Run the get values calculation.

        Inputs:
            df: Input dataframe used by the function.
            vehicle: Vehicle number to select from the data.
            column: Column name used to select the plotted or calculated quantity.

        Outputs:
            Requested value for the supplied inputs.
        """
        vehicle_df = df[df['vehicle'] == vehicle].sort_values('time')
        vehicle_df = vehicle_df[
            (vehicle_df['time'] >= TOTAL_EVALUATION_START_S)
            & (vehicle_df['time'] <= TOTAL_EVALUATION_END_S)
        ]
        if column == 'vsp':
            return calculate_vsp(vehicle_df['time'], vehicle_df['speed']).dropna()
        if column == 'fc':
            return (vehicle_df['fc'] * get_quantity_scale_factor('fc', engine_type)).dropna()
        return vehicle_df[column].dropna()

    bin_edges = {}
    for column, _ in plot_quantities:
        quantity_values = []
        for vehicle in row_vehicles:
            for df in model_data.values():
                values = get_values(df, vehicle, column)
                if not values.empty:
                    quantity_values.append(values)

        if not quantity_values:
            continue

        all_values = pd.concat(quantity_values, ignore_index=True)
        min_value = all_values.min()
        max_value = all_values.max()
        if min_value == max_value:
            padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
            min_value -= padding
            max_value += padding

        bin_edges[column] = [min_value + index * (max_value - min_value) / bins for index in range(bins + 1)]

    num_rows = len(row_labels)
    num_cols = len(plot_quantities)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(9, 8), sharex='col', squeeze=False)

    for row_index, (vehicle, row_label) in enumerate(zip(row_vehicles, row_labels)):
        for column_index, (column, x_label) in enumerate(plot_quantities):
            ax = axes[row_index, column_index]
            panel_label = chr(ord('a') + row_index * num_cols + column_index)

            for model, df in model_data.items():
                if column not in bin_edges:
                    continue
                values = get_values(df, vehicle, column)
                if values.empty:
                    print(f'No {x_label} data for {model.upper()} {velocity_class} {engine_type} {row_label}')
                    continue
                histogram_weights = None if density else [1 / len(values)] * len(values)

                ax.hist(
                    values,
                    bins=bin_edges[column],
                    density=density,
                    weights=histogram_weights,
                    histtype='step',
                    color=model_colors.get(model, None),
                    linewidth=1.2,
                    label=model.upper(),
                )

            ax.text(
                0.96,
                0.92,
                f'({panel_label})',
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                va='top',
                ha='right',
            )
            if row_index == 0:
                ax.set_title(x_label, fontsize=12)
            if row_index == num_rows - 1:
                ax.set_xlabel(x_label, fontsize=11)

            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=10)
            if column == 'fc':
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
                ax.xaxis.get_offset_text().set_fontsize(10)

    ylabel = 'Density' if density else 'Relative frequency'
    fig.supylabel(ylabel, fontsize=13, x=0.01)

    handles, labels = axes[-1, 0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0, 0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=11,
            frameon=False,
            loc='lower center',
            ncol=len(handles),
            bbox_to_anchor=(0.5, 0.0),
        )
    model_suffix = '_'.join(model_data.keys())
    output_path = PLOT_DIR / f'{velocity_class}_{engine_type}_{model_suffix}_velocity_vsp_fc_histogram.png'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.04, 1, 1), w_pad=0.6, h_pad=0.7)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    available_combinations = get_available_combinations()
    model_groups = [
        ['idm', 'midm'],
        ['fvdm', 'mfvdm'],
    ]

    for velocity_class, engine_type in sorted({
            (velocity_class, engine_type)
            for _, velocity_class, engine_type in available_combinations
    }):
        available_models = {
            model
            for model, current_velocity_class, current_engine_type in available_combinations
            if current_velocity_class == velocity_class and current_engine_type == engine_type
        }
        for model_group in model_groups:
            models = [model for model in model_group if model in available_models]
            if models:
                plot_model_quantity_histograms(
                    velocity_class,
                    engine_type,
                    model_names=models,
                )

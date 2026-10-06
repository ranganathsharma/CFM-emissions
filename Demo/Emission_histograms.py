"""
This file contains the codes to generate the emission rate histogram plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
START_SPEED_TOL = 1e-9
GASOLINE_DENSITY_G_PER_L = 700


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
    input_file = DEMO_DIR / f'outputs/{model}_{velocity_class}_{engine_type}_instantaneous_emissions.csv'
    if not input_file.exists():
        raise FileNotFoundError(f'Missing instantaneous validation file: {input_file}')

    df = pd.read_csv(input_file)
    df.columns = [column.strip().lower() for column in df.columns]

    for column in ['vehicle', 'time', 'speed', 'co2', 'fc', 'nox', 'pm', 'position', 'co2_g', 'fc_g', 'nox_g', 'pm_g']:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df.dropna(subset=['vehicle', 'time', 'speed', 'co2_g', 'fc_g', 'nox_g', 'pm_g', 'position'])
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


def get_plot_values(vehicle_df, column):
    """
    Description:
        Return histogram values for one vehicle and one plotted quantity.

    Inputs:
        vehicle_df: Dataframe containing rows for one vehicle.
        column: Column name used to select the plotted quantity.

    Outputs:
        Series containing values in the units shown on the x-axis.
    """
    if column == 'vsp':
        return calculate_vsp(vehicle_df['time'], vehicle_df['speed'])
    if column == 'fc_g':
        # The demo pipeline stores instantaneous gasoline fuel use in g/s.
        return vehicle_df[column] / GASOLINE_DENSITY_G_PER_L
    return vehicle_df[column]


def plot_velocity_vsp_fc_histograms(
        time_limits=(0, 500),
        bins=50):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        time_limits: Inclusive time-window limits used for plotting.
        bins: Number of histogram bins.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    model = 'idm'
    velocity_class = 'WLTC_rural'
    engine_type = 'gasoline'

    # The generated pipeline file contains vehicles 0-19.
    vehicles = [0, 10, 19]
    quantities = [
        ('speed', 'Velocity (m/s)'),
        ('vsp', 'VSP (W/kg)'),
        ('fc_g', 'FC (L/s)'),
    ]

    df = read_instantaneous_data(model, velocity_class, engine_type)
    df = df[(df['time'] >= time_limits[0]) & (df['time'] <= time_limits[1])]

    fig, axes = plt.subplots(len(vehicles), len(quantities), figsize=(9, 7), sharex='col')

    for row_index, vehicle in enumerate(vehicles):
        vehicle_df = df[df['vehicle'] == vehicle].sort_values('time')
        row_label = 'Leader' if vehicle == 0 else f'Vehicle {vehicle}'

        for column_index, (column, x_label) in enumerate(quantities):
            ax = axes[row_index, column_index]

            values = get_plot_values(vehicle_df, column)
            values = values.dropna()
            if values.empty:
                continue

            # Use relative frequency so each vehicle histogram is comparable.
            weights = [1 / len(values)] * len(values)
            ax.hist(values, bins=bins, weights=weights, histtype='step', color='#1f77b4', linewidth=1.3)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=9)

            if row_index == 0:
                ax.set_title(x_label, fontsize=12)
            if row_index == len(vehicles) - 1:
                ax.set_xlabel(x_label, fontsize=12)
            if column == 'fc_g':
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)

    fig.supylabel('Relative frequency', fontsize=12)
    output_path = DEMO_DIR / f'outputs/{model}_{velocity_class}_{engine_type}_velocity_vsp_fc_histograms.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    plot_velocity_vsp_fc_histograms()



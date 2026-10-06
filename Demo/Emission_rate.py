"""
Plot instantaneous velocity, VSP, and fuel-consumption rates for the demo case.
"""

# Short names such as df and VSP are conventional in this plotting script.
# pylint: disable=invalid-name

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_DIR / 'outputs'

MODEL_NAME = 'idm'
VELOCITY_CLASS = 'WLTC_rural'
ENGINE_TYPE = 'gasoline'
PLOT_VEHICLES = [0, 10, 19]
PLOT_QUANTITIES = [
    ('speed', 'Velocity (m/s)'),
    ('vsp', 'VSP (W/kg)'),
    ('fc_g', 'FC (g/s)'),
]
START_SPEED_TOL = 1e-9
VSP_ROLLING_TERM = 0.132
VSP_AERO_TERM = 0.000302
VSP_ACCELERATION_FACTOR = 1.1
REQUIRED_COLUMNS = [
    'vehicle',
    'time',
    'speed',
    'co2_g',
    'fc_g',
    'nox_g',
    'pm_g',
    'position',
]
NUMERIC_COLUMNS = [
    'vehicle',
    'time',
    'speed',
    'co2',
    'fc',
    'nox',
    'pm',
    'position',
    'co2_g',
    'fc_g',
    'nox_g',
    'pm_g',
]


def keep_rows_after_start(df: pd.DataFrame, speed_column: str) -> pd.DataFrame:
    """
    Description:
        Keep rows from the first nonzero-speed instant onward.

    Inputs:
        df: Input dataframe used by the function.
        speed_column: Name of the speed column used to detect when movement starts.

    Outputs:
        Dataframe containing only rows after the first movement instant.
    """
    moving_rows = df[df[speed_column].abs() > START_SPEED_TOL]
    if moving_rows.empty:
        return df.iloc[0:0].copy()

    start_time = moving_rows['time'].iloc[0]
    return df[df['time'] >= start_time].copy()


def keep_rows_after_vehicle_start(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description:
        Keep data only after each vehicle has started moving.

    Inputs:
        df: Input dataframe used by the function.

    Outputs:
        Dataframe containing post-start rows for every available vehicle.
    """
    rows = []

    for _, vehicle_df in df.groupby('vehicle', sort=False):
        vehicle_df = keep_rows_after_start(vehicle_df, 'speed')
        if not vehicle_df.empty:
            rows.append(vehicle_df)

    if not rows:
        return df.iloc[0:0].copy()

    return pd.concat(rows, ignore_index=True)


def read_instantaneous_data() -> pd.DataFrame:
    """
    Description:
        Read the demo instantaneous-emissions CSV and prepare numeric columns.

    Inputs:
        None.

    Outputs:
        Validated dataframe used by the plotting function.
    """
    input_file = (
        OUTPUT_DIR
        / f'{MODEL_NAME}_{VELOCITY_CLASS}_{ENGINE_TYPE}_instantaneous_emissions.csv'
    )
    if not input_file.exists():
        raise FileNotFoundError(f'Missing instantaneous validation file: {input_file}')

    df = pd.read_csv(input_file)
    df.columns = [column.strip().lower() for column in df.columns]

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

    return keep_rows_after_vehicle_start(df.dropna(subset=REQUIRED_COLUMNS))


def calculate_vsp(time_values: pd.Series, speed_values: pd.Series) -> pd.Series:
    """
    Description:
        Calculate vehicle specific power from speed and time series data.

    Inputs:
        time_values: Time series values used to calculate time steps.
        speed_values: Speed series values used to calculate acceleration and VSP.

    Outputs:
        Series containing calculated VSP values.
    """
    acceleration = speed_values.diff() / time_values.diff()
    acceleration = acceleration.replace([float('inf'), float('-inf')], 0).fillna(0)
    return (
        speed_values * (VSP_ACCELERATION_FACTOR * acceleration + VSP_ROLLING_TERM)
        + VSP_AERO_TERM * speed_values**3
    )


def get_plot_values(vehicle_df: pd.DataFrame, column: str) -> pd.Series:
    """
    Description:
        Return the values to plot for one vehicle and one quantity.

    Inputs:
        vehicle_df: Dataframe containing rows for one vehicle.
        column: Column name used to select the plotted quantity.

    Outputs:
        Series containing the plotted y-axis values.
    """
    if column == 'vsp':
        return calculate_vsp(vehicle_df['time'], vehicle_df['speed'])

    return vehicle_df[column]


def plot_velocity_vsp_fc(time_limits: tuple[int, int] = (0, 500)) -> Path:
    """
    Description:
        Plot velocity, VSP, and fuel consumption for the demo leader and followers.

    Inputs:
        time_limits: Inclusive time-window limits used for plotting.

    Outputs:
        Path to the saved PNG file.
    """
    df = read_instantaneous_data()
    df = df[(df['time'] >= time_limits[0]) & (df['time'] <= time_limits[1])]

    fig, axes = plt.subplots(
        len(PLOT_VEHICLES),
        len(PLOT_QUANTITIES),
        figsize=(13, 7),
        sharex=True,
    )

    for row_index, vehicle in enumerate(PLOT_VEHICLES):
        vehicle_df = df[df['vehicle'] == vehicle].sort_values('time')
        row_label = 'Leader' if vehicle == 0 else f'Vehicle {vehicle}'

        for column_index, (column, y_label) in enumerate(PLOT_QUANTITIES):
            ax = axes[row_index, column_index]
            y_values = get_plot_values(vehicle_df, column)
            ax.plot(vehicle_df['time'], y_values, linewidth=1.4, color='#1f77b4')
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=9)

            if row_index == 0:
                ax.set_title(y_label, fontsize=12)
            if column_index == 0:
                ax.set_ylabel(row_label, fontsize=12)
            if row_index == len(PLOT_VEHICLES) - 1:
                ax.set_xlabel('Time (s)', fontsize=12)
            if column == 'fc_g':
                ax.ticklabel_format(
                    axis='y',
                    style='sci',
                    scilimits=(0, 0),
                    useMathText=True,
                )

    output_path = (
        OUTPUT_DIR
        / f'{MODEL_NAME}_{VELOCITY_CLASS}_{ENGINE_TYPE}_velocity_vsp_fc.png'
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    plot_velocity_vsp_fc()

"""
This script is used to plot the total instantaneous emission for gasoline vehicles
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
    input_file = (
        DEMO_DIR
        / f'outputs/{model}_{velocity_class}_{engine_type}_instantaneous_emissions.csv'
    )
    if not input_file.exists():
        raise FileNotFoundError(f'Missing instantaneous validation file: {input_file}')

    df = pd.read_csv(input_file)
    df.columns = [column.strip().lower() for column in df.columns]

    numeric_columns = [
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
    for column in numeric_columns:
        if column not in df.columns:
            continue
        df[column] = pd.to_numeric(df[column], errors='coerce')

    required_columns = [
        'vehicle',
        'time',
        'speed',
        'co2_g',
        'fc_g',
        'nox_g',
        'pm_g',
        'position',
    ]
    df = df.dropna(subset=required_columns)
    return keep_rows_after_vehicle_start(df)


def plot_idm_wltc_rural_total_emissions(
        evaluation_duration_s=500):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        evaluation_duration_s: Evaluation duration in seconds after each vehicle starts moving.

    Outputs:
        Path to the saved total-emissions plot.
    """
    model = 'idm'
    velocity_class = 'WLTC_rural'
    engine_type = 'gasoline'
    output_dir = DEMO_DIR / 'outputs'

    df = read_instantaneous_data(model, velocity_class, engine_type)
    total_rows = []

    for vehicle, vehicle_df in df.groupby('vehicle', sort=True):
        vehicle_df = vehicle_df.sort_values('time').reset_index(drop=True)
        start_time = vehicle_df['time'].iloc[0]
        end_time = start_time + evaluation_duration_s
        vehicle_df = vehicle_df[vehicle_df['time'] <= end_time]

        if vehicle_df.empty:
            continue

        # The pipeline stores grams emitted/consumed per 1 s row in the *_g columns.
        total_rows.append({
            'vehicle': int(vehicle),
            'co2_g': vehicle_df['co2_g'].sum(),
            'nox_g': vehicle_df['nox_g'].sum(),
            'pm_g': vehicle_df['pm_g'].sum(),
            'fc_g': vehicle_df['fc_g'].sum(),
            'fc_l': vehicle_df['fc_g'].sum() / GASOLINE_DENSITY_G_PER_L,
        })

    totals = pd.DataFrame(total_rows)
    if totals.empty:
        raise ValueError('No vehicle totals could be calculated.')

    quantities = [
        ('co2_g', 'Total CO2 (g)'),
        ('nox_g', 'Total NOx (g)'),
        ('pm_g', 'Total PM (g)'),
        ('fc_l', 'Total FC (L)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    axes = axes.reshape(2, 2)

    for ax, (column, label) in zip(axes.ravel(), quantities):
        ax.scatter(totals['vehicle'], totals[column], s=18, marker='s', color='#1f77b4')
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # NOx and PM totals are small compared with CO2 and fuel consumption.
        if column in ['nox_g', 'pm_g']:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    for ax in axes[-1, :]:
        ax.set_xlabel('Vehicle number', fontsize=11)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f'{model}_{velocity_class}_{engine_type}_vehicle_totals'
    totals.to_csv(output_dir / f'{output_stem}.csv', index=False)

    output_path = output_dir / f'{output_stem}.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved IDM WLTC rural vehicle totals to {output_path}')
    return output_path


if __name__ == '__main__':
    plot_idm_wltc_rural_total_emissions()



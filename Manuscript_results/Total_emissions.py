"""
Plot total emissions and fuel consumption for each available speed profile.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
INSTANTANEOUS_DIR = RESULTS_DIR / 'outputs/instantaneous_data'
PLOT_DIR = RESULTS_DIR / 'outputs/plots/total_emissions'
TOTAL_OUTPUT_DIR = RESULTS_DIR / 'outputs/total_data'
START_SPEED_TOL = 1e-9


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
        raise FileNotFoundError(f'Missing instantaneous output file: {input_file}')

    df = pd.read_csv(input_file)
    df.columns = [column.strip().lower() for column in df.columns]

    numeric_columns = ['vehicle', 'time', 'speed', 'co2', 'nox', 'pm', 'fc', 'position']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    return df.dropna(subset=numeric_columns)


def keep_rows_after_start(vehicle_df):
    """
    Description:
        Keep rows from the first nonzero-speed instant onward.

    Inputs:
        vehicle_df: Dataframe containing rows for one vehicle.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    moving_rows = vehicle_df[vehicle_df['speed'].abs() > START_SPEED_TOL]
    if moving_rows.empty:
        return vehicle_df.iloc[0:0].copy()

    return vehicle_df[vehicle_df['time'] >= moving_rows['time'].iloc[0]].copy()


def calculate_vehicle_totals(df, model, evaluation_duration_s):
    """
    Description:
        Integrate instantaneous rates into total per-vehicle emissions and fuel consumption.

    Inputs:
        df: Input dataframe used by the function.
        model: Model name used to select data or label output.
        evaluation_duration_s: Evaluation duration in seconds after each vehicle starts moving.

    Outputs:
        Calculated dataframe or numeric values for the supplied inputs.
    """
    rows = []

    for vehicle, vehicle_df in df.groupby('vehicle', sort=True):
        vehicle_df = keep_rows_after_start(vehicle_df.sort_values('time'))
        if vehicle_df.empty:
            continue

        start_time = vehicle_df['time'].iloc[0]
        window_df = vehicle_df[vehicle_df['time'] <= start_time + evaluation_duration_s].copy()
        if window_df.empty:
            continue

        # PHEMlight rates are stored per hour, so multiply by the row time step in hours.
        dt_hours = window_df['time'].diff().fillna(0) / 3600
        rows.append({
            'model': model,
            'vehicle': int(vehicle),
            'co2_g': (window_df['co2'] * dt_hours).sum(),
            'nox_g': (window_df['nox'] * dt_hours).sum(),
            'pm_g': (window_df['pm'] * dt_hours).sum(),
            'fc_g': (window_df['fc'] * dt_hours).sum(),
        })

    if not rows:
        raise ValueError('No vehicle totals could be calculated.')

    return pd.DataFrame(rows)


def plot_total_emissions(velocity_class, engine_type, model_names, evaluation_duration_s=1800):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.
        model_names: List of model names to include in the calculation or plot.
        evaluation_duration_s: Evaluation duration in seconds after each vehicle starts moving.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    totals = []

    for model in model_names:
        input_file = INSTANTANEOUS_DIR / f'{model}_{velocity_class}_{engine_type}_instantaneous_outputs.csv'
        if input_file.exists():
            totals.append(calculate_vehicle_totals(
                read_instantaneous_data(model, velocity_class, engine_type),
                model,
                evaluation_duration_s,
            ))

    if not totals:
        raise ValueError(f'No data found for {velocity_class} {engine_type} and {model_names}')

    totals = pd.concat(totals, ignore_index=True)

    quantities = [
        ('co2_g', 'Total CO2 (g)'),
        ('nox_g', 'Total NOx (g)'),
        ('pm_g', 'Total PM (g)'),
        ('fc_g', 'Total FC (g)'),
    ]
    model_colors = {
        'idm': '#1f77b4',
        'midm': '#d62728',
        'fvdm': '#16BE16',
        'mfvdm': '#7511d3',
    }
    model_labels = {model: model.upper() for model in model_names}

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, dpi=600)
    for ax, (column, ylabel) in zip(axes.ravel(), quantities):
        for model in model_names:
            model_df = totals[totals['model'] == model]
            if model_df.empty:
                continue
            ax.scatter(
                model_df['vehicle'],
                model_df[column],
                s=18,
                marker='s',
                color=model_colors[model],
                label=model_labels[model],
            )

        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # NOx and PM totals are much smaller than CO2 and FC.
        if column in ['nox_g', 'pm_g']:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

    for ax in axes[-1, :]:
        ax.set_xlabel('Vehicle number', fontsize=11)

    TOTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    model_suffix = '_'.join(model_names)
    output_stem = f'{velocity_class}_{engine_type}_{model_suffix}_vehicle_totals'
    totals.to_csv(TOTAL_OUTPUT_DIR / f'{output_stem}.csv', index=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False)

    output_path = PLOT_DIR / f'{output_stem}.png'
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved total emissions plot to {output_path}')
    return output_path


if __name__ == '__main__':
    available_combinations = get_available_combinations()
    model_groups = [['idm', 'midm'], ['fvdm', 'mfvdm']]

    for velocity_class, engine_type in sorted({case[1:] for case in available_combinations}):
        available_models = {
            model
            for model, case_velocity_class, case_engine_type in available_combinations
            if case_velocity_class == velocity_class and case_engine_type == engine_type
        }

        for model_group in model_groups:
            models = [model for model in model_group if model in available_models]
            if models:
                plot_total_emissions(velocity_class, engine_type, models)

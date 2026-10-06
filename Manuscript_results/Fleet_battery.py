"""
This file contains the code to plot the fleet electric energy consumption
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent
PLOT_DIR = RESULTS_DIR / 'outputs/plots/fleet_battery'
FLEET_EMISSIONS_DIR = RESULTS_DIR / 'outputs/plots/fleet_emissions'


def plot_battery_energy_consumption_by_drive_cycle(
        input_file=None,
        output_dir=PLOT_DIR,
        model_order=None,
        condition_order=None,
        save_fig=True):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        input_file: CSV file to read instead of the default input location.
        output_dir: Directory where generated files are saved.
        model_order: Order in which models are displayed in the plot.
        condition_order: Order in which drive cycles are displayed.
        save_fig: Whether to write the generated figure to disk.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    if model_order is None:
        model_order = ['idm', 'fvdm', 'midm', 'mfvdm']
    if condition_order is None:
        condition_order = [
            'WLTC_urban',
            'CADC_urban',
            'FTP',
            'WLTC_rural',
            'CADC_rural',
            'WLTC_highway',
            'CADC_highway',
            'HWFET',
            'Ford_focus',
        ]

    if input_file is None:
        input_paths = sorted(FLEET_EMISSIONS_DIR.glob('*_valid_parameter_rows.csv'))
        if not input_paths:
            raise FileNotFoundError(f'Missing valid parameter rows in {FLEET_EMISSIONS_DIR}')
        parameter_df = pd.concat(
            [pd.read_csv(input_path) for input_path in input_paths],
            ignore_index=True,
        )
    else:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(
                'Missing valid parameter-combinations file needed for battery plot: '
                f'{input_path}'
            )
        parameter_df = pd.read_csv(input_path)
    required_columns = [
        'model',
        'velocity_class',
        'engine_type',
        'fc_per_km',
    ]
    missing_columns = [
        column for column in required_columns
        if column not in parameter_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f'Battery input data is missing columns: {missing_columns}'
        )

    parameter_df['fc_per_km'] = pd.to_numeric(parameter_df['fc_per_km'], errors='coerce')
    battery_df = parameter_df[
        (parameter_df['engine_type'] == 'B')
        & parameter_df['velocity_class'].isin(condition_order)
    ].dropna(subset=['fc_per_km']).copy()
    if battery_df.empty:
        raise ValueError('No battery rows found for the requested drive cycles')

    summary_df = (
        battery_df
        .groupby(['model', 'velocity_class'], as_index=False)
        .agg(
            average_fc_per_km=('fc_per_km', 'mean'),
            std_fc_per_km=('fc_per_km', lambda values: values.std(ddof=0)),
        )
    )

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

    x = np.arange(len(condition_order))
    bar_width = 0.8 / len(model_order)

    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=600)
    for model_index, model in enumerate(model_order):
        values = []
        errors = []
        for condition in condition_order:
            model_row = summary_df[
                (summary_df['model'] == model)
                & (summary_df['velocity_class'] == condition)
            ]
            if model_row.empty:
                values.append(np.nan)
                errors.append(np.nan)
            else:
                values.append(model_row.iloc[0]['average_fc_per_km'])
                errors.append(model_row.iloc[0]['std_fc_per_km'])

        x_position = x + (model_index - (len(model_order) - 1) / 2) * bar_width
        ax.bar(
            x_position,
            values,
            yerr=errors,
            width=bar_width,
            color=model_colors.get(model),
            ecolor='k',
            error_kw={'elinewidth': 1, 'capthick': 1, 'capsize': 1.5},
            label=model_labels.get(model, model.upper()),
        )

    dc_labels = [rf'DC$_{{{index}}}$' for index in range(1, len(condition_order) + 1)]
    ax.set_xticks(x)
    ax.set_xticklabels(dc_labels, fontsize=15)
    ax.set_ylabel('Battery energy\nconsumed (kWh/km)', fontsize=15)
    # ax.set_xlabel('Drive cycle', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(model_order),
        fontsize=15,
        frameon=False,
    )
    ax.set_ylim(0.1, 0.3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'battery_energy_consumption_by_drive_cycle.png'
    if save_fig:
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f'Saved battery energy consumption plot to {output_path}')
    plt.close(fig)

    dc_mapping = pd.DataFrame({
        'drive_cycle_label': dc_labels,
        'velocity_class': condition_order,
    })
    return output_path, summary_df, dc_mapping


if __name__ == '__main__':
    plot_battery_energy_consumption_by_drive_cycle()

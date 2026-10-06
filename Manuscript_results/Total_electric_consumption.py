"""
This file contains the code to plot the total electric energy consumed by all BEVs
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent
INSTANTANEOUS_DIR = RESULTS_DIR / 'outputs/instantaneous_data'
PLOT_DIR = RESULTS_DIR / 'outputs/plots/battery_totals'
START_SPEED_TOL = 1e-9


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


def plot_vehicle_total_fc_per_distance_scatter_from_instantaneous(
        engine,
        velocity_class_list=None,
        model_names=None,
        input_dir=INSTANTANEOUS_DIR,
        output_dir=PLOT_DIR,
        vehicle_count=200,
        evaluation_duration_s=1800,
        save_csv=False):
    """
    Plot per-vehicle distance-specific fuel consumption from instantaneous data.

    For each vehicle, FC is integrated over the first evaluation_duration_s
    seconds after the vehicle starts moving and divided by the distance travelled
    in the same window. Gasoline/diesel FC is converted from g to L using fuel
    density; BEV FC is kept as kWh.
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
            'Ford_focus',
        ]
    if model_names is None:
        model_names = ['idm', 'fvdm', 'midm', 'mfvdm']

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    model_labels = {
        'idm': 'IDM',
        'fvdm': 'FVDM',
        'midm': 'MIDM',
        'mfvdm': 'MFVDM',
    }
    model_colors = {
        'idm': '#1f77b4',
        'fvdm': '#16BE16',
        'midm': '#d62728',
        'mfvdm': '#7511d3',
    }
    required_columns = [
        'model',
        'velocity_class',
        'engine_type',
        'vehicle',
        'time',
        'speed',
        'fc',
        'position',
    ]
    numeric_columns = ['vehicle', 'time', 'speed', 'fc', 'position']
    fuel_density = get_fuel_density_g_per_l(engine)
    fc_distance_unit = 'kWh/km' if fuel_density is None else 'L/km'

    rows = []
    for velocity_class in velocity_class_list:
        for model in model_names:
            input_path = input_dir / f'{model}_{velocity_class}_{engine}_instantaneous_outputs.csv'
            if not input_path.exists():
                print(f'Missing instantaneous file: {input_path}')
                continue

            df = pd.read_csv(input_path, usecols=lambda column: column in required_columns)
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                raise ValueError(f'{input_path} is missing columns: {missing_columns}')

            df = df[df['engine_type'] == engine].copy()
            for column in numeric_columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            df = df.dropna(subset=numeric_columns)
            df = df[(df['vehicle'] >= 1) & (df['vehicle'] <= vehicle_count)]

            for vehicle, vehicle_df in df.groupby('vehicle', sort=True):
                vehicle_df = vehicle_df.sort_values('time').reset_index(drop=True)
                moving_rows = vehicle_df[vehicle_df['speed'].abs() > START_SPEED_TOL]
                if moving_rows.empty:
                    continue

                evaluation_start_s = moving_rows['time'].iloc[0]
                evaluation_end_s = evaluation_start_s + evaluation_duration_s
                window_df = vehicle_df[
                    (vehicle_df['time'] >= evaluation_start_s)
                    & (vehicle_df['time'] <= evaluation_end_s)
                ].copy()
                if window_df.empty:
                    continue

                dt = window_df['time'].diff().fillna(0)
                total_time_s = window_df['time'].iloc[-1] - window_df['time'].iloc[0]
                total_distance_m = window_df['position'].max() - window_df['position'].min()
                if total_distance_m <= 0:
                    continue

                total_distance_km = total_distance_m / 1000
                total_fc = ((window_df['fc'] / 3600) * dt).sum()
                if fuel_density is not None:
                    total_fc = total_fc / fuel_density

                rows.append({
                    'model': model,
                    'velocity_class': velocity_class,
                    'engine_type': engine,
                    'vehicle': int(vehicle),
                    'evaluation_start_s': evaluation_start_s,
                    'evaluation_end_s': evaluation_end_s,
                    'total_time_s': total_time_s,
                    'total_distance_m': total_distance_m,
                    'total_distance_km': total_distance_km,
                    'total_fc': total_fc,
                    'fc_distance_unit': fc_distance_unit,
                    'fc_per_km': total_fc / total_distance_km,
                })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError(f'No instantaneous vehicle FC rows found for engine {engine}')

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_'.join(velocity_class_list)
    if save_csv:
        plot_df.to_csv(
            output_dir / f'{suffix}_{engine}_instantaneous_vehicle_fc_per_distance_scatter_data.csv',
            index=False,
        )

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), dpi=600, sharex=True)
    axes = np.asarray(axes).reshape(3, 3)

    for index, velocity_class in enumerate(velocity_class_list[:9]):
        row_index = index // 3
        col_index = index % 3
        ax = axes[row_index, col_index]
        condition_df = plot_df[plot_df['velocity_class'] == velocity_class]
        panel_label = chr(ord('a') + index)

        for model in model_names:
            model_df = condition_df[condition_df['model'] == model].sort_values('vehicle')
            if model_df.empty:
                continue

            ax.scatter(
                model_df['vehicle'],
                model_df['fc_per_km'],
                s=14,
                alpha=1,
                marker='s',
                color=model_colors.get(model),
                label=model_labels.get(model, model.upper()) if index == 0 else None,
                linewidths=0,
            )

        subplot_values = pd.to_numeric(condition_df['fc_per_km'], errors='coerce').dropna()
        if not subplot_values.empty:
            y_min = subplot_values.min()
            y_max = subplot_values.max()
            if y_max == y_min:
                y_padding = 0.5 if y_max == 0 else abs(y_max) * 0.05
            else:
                y_padding = (y_max - y_min) * 0.05
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(11)
        ax.set_title(velocity_class.replace('_', ' '), fontsize=14)
        ax.text(
            0.97,
            0.93,
            f'({panel_label})',
            transform=ax.transAxes,
            fontsize=13,
            fontweight='bold',
            va='top',
            ha='right',
        )
        if row_index == 2:
            ax.set_xlabel('Vehicle number', fontsize=13)
        if col_index == 0:
            ax.set_ylabel(f'FC ({fc_distance_unit})', fontsize=13)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(handles),
            fontsize=14,
            frameon=False,
        )

    output_path = output_dir / f'{engine}_instantaneous_vehicle_fc_per_distance_scatter_3x3.png'
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved instantaneous vehicle FC-per-distance scatter plot to {output_path}')

    return output_path, plot_df


if __name__ == '__main__':
    plot_vehicle_total_fc_per_distance_scatter_from_instantaneous('B')

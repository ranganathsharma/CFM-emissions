"""
Plot fuel-consumption histograms for each powertrain.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RESULTS_DIR / 'outputs'
INSTANTANEOUS_DIR = OUTPUT_DIR / 'instantaneous_data'
PLOT_DIR = OUTPUT_DIR / 'plots/battery_histograms'
START_SPEED_TOL = 1e-9
TOTAL_EVALUATION_START_S = 800
TOTAL_EVALUATION_END_S = 2600


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

    return df[df['time'] >= moving_rows['time'].iloc[0]].copy()


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
        if not vehicle_df.empty:
            rows.append(vehicle_df)

    if not rows:
        return df.iloc[0:0].copy()

    return pd.concat(rows, ignore_index=True)


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
    return 1


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


def plot_powertrain_fc_histograms(
        velocity_class,
        model_names=None,
        vehicle_list=None,
        engine_type_list=None,
        bins=50,
        density=False):
    """
    Description:
        Create and save the requested plot from the prepared data.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        model_names: List of model names to include in the calculation or plot.
        vehicle_list: Follower vehicle numbers to include, excluding the leader.
        engine_type_list: Input value used by this function for engine type list.
        bins: Number of histogram bins.
        density: Whether histogram values should be plotted as density instead of relative frequency.

    Outputs:
        Path or tuple containing the saved plot path and any calculated data returned by the plotting function.
    """
    if model_names is None:
        model_names = ['idm', 'midm']
    if vehicle_list is None:
        vehicle_list = [50, 100, 200]
    if engine_type_list is None:
        engine_type_list = ['G', 'G_HEV', 'D', 'D_HEV', 'B']

    model_colors = {
        'idm': '#1f77b4',
        'fvdm': '#16BE16',
        'midm': '#d62728',
        'mfvdm': '#7511d3',
    }

    model_data = {}
    for engine_type in engine_type_list:
        model_data[engine_type] = {}
        for model in model_names:
            try:
                model_data[engine_type][model] = read_instantaneous_data(
                    model,
                    velocity_class,
                    engine_type,
                )
            except FileNotFoundError as exc:
                print(exc)

    if not any(model_rows for model_rows in model_data.values()):
        raise ValueError(f'No instantaneous FC data found for {velocity_class}')

    row_labels = ['Leader'] + [f'Vehicle {vehicle}' for vehicle in vehicle_list]

    def get_fc_values(engine_type, model, row_index):
        """
        Description:
            Run the get fc values calculation.

        Inputs:
            engine_type: Powertrain or fuel type code used to select units and conversion factors.
            model: Model name used to select data or label output.
            row_index: Row index of the current subplot or vehicle group.

        Outputs:
            Requested value for the supplied inputs.
        """
        df = model_data.get(engine_type, {}).get(model)
        if df is None:
            return None

        vehicle = 0 if row_index == 0 else vehicle_list[row_index - 1]
        vehicle_df = df[df['vehicle'] == vehicle]
        vehicle_df = vehicle_df[
            (vehicle_df['time'] >= TOTAL_EVALUATION_START_S)
            & (vehicle_df['time'] <= TOTAL_EVALUATION_END_S)
        ]

        if vehicle_df.empty:
            print(
                f'No FC data for {model.upper()} {engine_type} '
                f'{row_labels[row_index]}'
            )
            return None

        values = vehicle_df['fc'].dropna() * get_quantity_scale_factor('fc', engine_type)
        if values.empty:
            return None

        return values

    model_pair_rows = [
        ('IDM-MIDM', ['idm', 'midm']),
        ('FVDM-MFVDM', ['fvdm', 'mfvdm']),
    ]
    model_pair_rows = [
        (label, [model for model in pair_models if model in model_names])
        for label, pair_models in model_pair_rows
    ]
    model_pair_rows = [
        (label, pair_models)
        for label, pair_models in model_pair_rows
        if pair_models
    ]
    if not model_pair_rows:
        raise ValueError('No supported model names found for paired FC histogram rows')

    output_paths = []
    num_rows = len(model_pair_rows)
    num_cols = len(row_labels)

    for engine_type in engine_type_list:
        engine_values = []
        for column_index in range(num_cols):
            for _, pair_models in model_pair_rows:
                for model in pair_models:
                    values = get_fc_values(engine_type, model, column_index)
                    if values is not None and not values.empty:
                        engine_values.append(values)

        if not engine_values:
            continue

        all_values = pd.concat(engine_values, ignore_index=True)
        min_value = all_values.min()
        max_value = all_values.max()
        if min_value == max_value:
            padding = 0.5 if min_value == 0 else abs(min_value) * 0.05
            min_value -= padding
            max_value += padding
        bin_edges = np.linspace(min_value, max_value, bins + 1)

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(9, 5),
            sharex=True,
            sharey='col',
        )
        axes = np.asarray(axes).reshape(num_rows, num_cols)
        column_y_max = np.zeros(num_cols)

        for row_index, (pair_label, pair_models) in enumerate(model_pair_rows):
            for column_index, row_label in enumerate(row_labels):
                ax = axes[row_index, column_index]
                panel_label = chr(ord('a') + row_index * num_cols + column_index)

                for model in pair_models:
                    values = get_fc_values(engine_type, model, column_index)
                    if values is None:
                        continue

                    histogram_weights = None
                    if not density:
                        histogram_weights = np.ones(len(values)) / len(values)

                    counts, _, _ = ax.hist(
                        values,
                        bins=bin_edges,
                        density=density,
                        weights=histogram_weights,
                        histtype='step',
                        color=model_colors.get(model, None),
                        linewidth=1.3,
                        label=model.upper(),
                    )
                    if len(counts) > 0:
                        column_y_max[column_index] = max(
                            column_y_max[column_index],
                            counts.max(),
                        )

                ax.text(
                    0.96,
                    0.92,
                    f'({panel_label})',
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight='bold',
                    va='top',
                    ha='right',
                )
                if row_index == num_rows - 1:
                    ax.set_xlabel(get_fc_label(engine_type), fontsize=15)

                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
                ax.xaxis.get_offset_text().set_fontsize(10)

        for column_index, y_max in enumerate(column_y_max):
            if y_max > 0:
                for ax in axes[:, column_index]:
                    ax.set_ylim(0, y_max * 1.05)

        ylabel = 'Density' if density else 'Relative frequency'
        fig.supylabel(ylabel, fontsize=15, x=0.07)

        handles_by_label = {}
        for ax in axes.flat:
            ax_handles, ax_labels = ax.get_legend_handles_labels()
            for handle, label in zip(ax_handles, ax_labels):
                handles_by_label.setdefault(label, handle)

        if handles_by_label:
            labels = [
                model.upper()
                for model in model_names
                if model.upper() in handles_by_label
            ]
            handles = [handles_by_label[label] for label in labels]
            fig.legend(
                handles,
                labels,
                fontsize=12,
                frameon=False,
                loc='lower center',
                ncol=len(handles),
                bbox_to_anchor=(0.5, 0.0),
            )

        model_suffix = '_'.join(model_names)
        output_path = (
            PLOT_DIR
            / f'{velocity_class}_{model_suffix}_{engine_type}_fc_powertrain_histogram.png'
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0.07, 0.08, 1, 1), w_pad=0.8, h_pad=0.7)
        fig.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        output_paths.append(output_path)

    if not output_paths:
        raise ValueError(f'No instantaneous FC data found for {velocity_class}')

    if len(output_paths) == 1:
        return output_paths[0]
    return output_paths


if __name__ == '__main__':
    for velocity_class in [
            'WLTC_urban',
            'WLTC_rural',
            'WLTC_highway',
            'CADC_urban',
            'CADC_rural',
            'CADC_highway',
            'FTP',
            'HWFET']:
        plot_powertrain_fc_histograms(
            velocity_class,
            ['idm', 'midm', 'fvdm', 'mfvdm'],
            [50, 200],
            ['B'],
            bins=50,
            density=False,
        )

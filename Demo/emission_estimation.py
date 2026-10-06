"""
Demonstrate the emission-estimation workflow for CFM-generated trajectories.
"""

# Short names such as x and v are standard for trajectory arrays.
# The project import depends on adding the repository root to sys.path.
# pylint: disable=invalid-name,wrong-import-position

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(1, str(REPO_ROOT))

YAML_FILE_PATH = REPO_ROOT / 'parameters.yaml'
OUTPUT_DIR = DEMO_DIR / 'outputs'
PHEM_INPUT_DIR = DEMO_DIR / 'Saved_inputs'
PHEM_OUTPUT_DIR = DEMO_DIR / 'Saved_outputs'
PHEM_EXE = DEMO_DIR / 'PhemApp/bin/Release/net9.0/PhemApp.exe'

MODEL_NAME = 'idm'
VELOCITY_CLASS = 'WLTC_rural'
ENGINE_TYPE = 'G'
ENGINE_LABEL = 'gasoline'
STANDARD_TYPE = 'EU6ab'
SIMULATION_DT = 0.1
PHEM_TIME_STEP_RATIO = 10
PHEM_RATE_SECONDS_PER_HOUR = 3600

from cfm_solution import simulations


def load_model_parameters() -> dict:
    """
    Description:
        Read the repository parameter file and return the selected model parameters.

    Inputs:
        None.

    Outputs:
        Dictionary containing the selected model's parameter values.
    """
    with open(YAML_FILE_PATH, 'r', encoding='utf-8') as file:
        model_param_dict = yaml.safe_load(file)

    return model_param_dict['Models'][MODEL_NAME]


def run_cfm_simulation(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Run the CFM simulation and return vehicle positions and velocities.

    Inputs:
        params: Dictionary of model parameter values.

    Outputs:
        Tuple containing position and velocity arrays.
    """
    simulator = simulations(
        velocity_class=VELOCITY_CLASS,
        model_name=MODEL_NAME,
        params=params,
        dt=SIMULATION_DT,
        delay=False,
    )

    _, position, velocity = simulator.evolve()
    print('The simulation was successful')
    return position, velocity


def resample_to_one_second(
        position: np.ndarray,
        velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Average the CFM 0.1 s trajectories into 1 s samples for PHEMlight.

    Inputs:
        position: Vehicle position array from the CFM simulation.
        velocity: Vehicle velocity array from the CFM simulation.

    Outputs:
        Tuple containing 1 s position and velocity arrays.
    """
    sample_count = velocity.shape[1] // PHEM_TIME_STEP_RATIO * PHEM_TIME_STEP_RATIO
    velocity_1s = (
        velocity[:, :sample_count]
        .reshape(velocity.shape[0], -1, PHEM_TIME_STEP_RATIO)
        .mean(axis=2)
    )
    position_1s = (
        position[:, :sample_count]
        .reshape(position.shape[0], -1, PHEM_TIME_STEP_RATIO)
        .mean(axis=2)
    )
    return position_1s, velocity_1s


def write_phem_inputs(velocity_1s: np.ndarray) -> list[str]:
    """
    Description:
        Write one single-row velocity CSV per vehicle for the PHEMlight wrapper.

    Inputs:
        velocity_1s: One-second velocity array for all vehicles.

    Outputs:
        List of input file stems written for PHEMlight.
    """
    PHEM_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_stems = []
    for vehicle in range(velocity_1s.shape[0]):
        input_stem = f'{MODEL_NAME}_{VELOCITY_CLASS}_{ENGINE_LABEL}_veh_{vehicle}'
        input_stems.append(input_stem)
        with open(
                PHEM_INPUT_DIR / f'{input_stem}.csv',
                mode='w',
                newline='',
                encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(velocity_1s[vehicle])

    return input_stems


def run_phem() -> None:
    """
    Description:
        Run the PHEMlight wrapper on the generated vehicle input files.

    Inputs:
        None.

    Outputs:
        None. PHEMlight output CSV files are written as a side effect.
    """
    PHEM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [PHEM_EXE, ENGINE_TYPE, STANDARD_TYPE, PHEM_INPUT_DIR, PHEM_OUTPUT_DIR],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f'PHEMlight failed with code {result.returncode}: {result.stderr}')


def read_vehicle_emissions(
        vehicle: int,
        input_stem: str,
        position_1s: np.ndarray) -> pd.DataFrame | None:
    """
    Description:
        Read one vehicle's PHEMlight output and align it with the CFM position trace.

    Inputs:
        vehicle: Vehicle number being processed.
        input_stem: File stem used for the vehicle's PHEMlight input and output.
        position_1s: One-second position array for all vehicles.

    Outputs:
        Dataframe of instantaneous emissions for one vehicle, or None when data is missing.
    """
    output_file = PHEM_OUTPUT_DIR / f'{input_stem}_Output.csv'
    if not output_file.exists():
        print(f'Missing PHEM output for vehicle {vehicle}: {output_file}')
        return None

    df = pd.read_csv(output_file, skiprows=[0, 2], index_col=False)
    df.columns = [column.strip() for column in df.columns]
    required_columns = ['Time', 'Speed', 'CO2', 'NOx', 'PM', 'FC']
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f'Missing PHEM columns for vehicle {vehicle}: {missing_columns}')
        return None

    df = df[required_columns].apply(pd.to_numeric, errors='coerce').dropna()
    common_len = min(len(df), position_1s.shape[1])
    df = df.iloc[:common_len].copy()
    df['vehicle'] = vehicle
    df['position'] = position_1s[vehicle, :common_len]

    # PHEMlight reports rates in g/h; each row represents one second.
    df['CO2_g'] = df['CO2'] / PHEM_RATE_SECONDS_PER_HOUR
    df['NOx_g'] = df['NOx'] / PHEM_RATE_SECONDS_PER_HOUR
    df['PM_g'] = df['PM'] / PHEM_RATE_SECONDS_PER_HOUR
    df['FC_g'] = df['FC'] / PHEM_RATE_SECONDS_PER_HOUR
    return df


def combine_vehicle_emissions(input_stems: list[str], position_1s: np.ndarray) -> pd.DataFrame:
    """
    Description:
        Combine PHEMlight outputs from all vehicles into one dataframe.

    Inputs:
        input_stems: File stems used to locate each vehicle's PHEMlight output.
        position_1s: One-second position array for all vehicles.

    Outputs:
        Dataframe containing instantaneous emissions for every available vehicle.
    """
    emission_rows = []
    for vehicle, input_stem in enumerate(input_stems):
        vehicle_df = read_vehicle_emissions(vehicle, input_stem, position_1s)
        if vehicle_df is not None:
            emission_rows.append(vehicle_df)

    if not emission_rows:
        raise ValueError('No PHEMlight output rows could be read.')

    return pd.concat(emission_rows, ignore_index=True)


def save_instantaneous_emissions(emissions: pd.DataFrame) -> Path:
    """
    Description:
        Save the combined instantaneous emissions dataframe to the demo outputs folder.

    Inputs:
        emissions: Combined instantaneous emissions dataframe.

    Outputs:
        Path to the saved CSV file.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = (
        OUTPUT_DIR
        / f'{MODEL_NAME}_{VELOCITY_CLASS}_{ENGINE_LABEL}_instantaneous_emissions.csv'
    )
    emissions.to_csv(output_path, index=False)
    print('Gasoline instantaneous emissions were calculated successfully')
    return output_path


def main() -> Path:
    """
    Description:
        Run the full demo workflow from CFM simulation to saved emissions CSV.

    Inputs:
        None.

    Outputs:
        Path to the saved instantaneous emissions CSV file.
    """
    params = load_model_parameters()
    position, velocity = run_cfm_simulation(params)
    position_1s, velocity_1s = resample_to_one_second(position, velocity)
    input_stems = write_phem_inputs(velocity_1s)
    run_phem()
    emissions = combine_vehicle_emissions(input_stems, position_1s)
    return save_instantaneous_emissions(emissions)


if __name__ == '__main__':
    main()

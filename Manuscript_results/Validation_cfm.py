"""
Validate CFM emissions after examiner comments about urban driving conditions.
"""

# Import the parameters file

import csv
import os
import sys
from collections.abc import Callable
from typing import Any, cast

import yaml
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')

with open(yaml_file_path, 'r', encoding='utf-8') as file:
    model_param_dict = yaml.safe_load(file)

# Libraries import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import logging
import subprocess
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker


# Custom library import

import leader_velocity as lv
from model_equations import Model_equations
from solvers import solvers

exe_path = os.path.join("New_PhemApp", "bin", "Release", "net9.0", "New_PhemApp.exe")
TOTAL_EVALUATION_START_S = 800
TOTAL_EVALUATION_END_S = 2600
TOTAL_EVALUATION_DURATION_S = 1800
INSTANTANEOUS_VEHICLE_ROWS = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    10: 10,
    20: 20,
    30: 30,
    40: 40,
    50: 50,
    55: 55,
    60: 60,
    65: 65,
    70: 70,
    75: 75,
    80: 80,
    85: 85,
    90: 90,
    95: 95,
    100: 100,
    105: 105,
    110: 110,
    115: 115,
    120: 120,
    125: 125,
    130: 130,
    135: 135,
    140: 140,
    145: 145,
    150: 150,
    155: 155,
    160: 160,
    165: 165,
    170: 170,
    175: 175,
    180: 180,
    185: 185,
    190: 190,
    195: 195,
    200: 200
}

TRAJECTORY_COLOR_LIMITS = {
    'CO2': (0, 28800),
    'NOx': (0, 40),
    'PM': (0, 0.23),
    'FC': (0, 14500),
}

FUEL_DENSITY_G_PER_L = {
    'G': 700,
    'G_HEV': 700,
    'D': 800,
    'D_HEV': 800,
}

def get_fuel_consumption_column(df, engine_type, vehicle_label):
    """
    Description:
        Select the fuel-consumption column that matches the requested engine and vehicle.

    Inputs:
        df: Input dataframe used by the function.
        engine_type: Powertrain or fuel type code used to select units and conversion factors.
        vehicle_label: Input value used by this function for vehicle label.

    Outputs:
        Requested value for the supplied inputs.
    """
    if engine_type == 'B':
        if 'Engine Power' in df.columns:
            return 'Engine Power'

        print(
            f'Missing battery Engine Power column for vehicle {vehicle_label}; '
            'falling back to FC'
        )

    return 'FC'


def convert_total_fc(total_fc, engine_type):
    """
    Description:
        Convert total fuel consumption to the display unit used for each powertrain.

    Inputs:
        total_fc: Raw total fuel or energy value integrated from PHEMlight rates.
        engine_type: Powertrain or fuel type code used to select the conversion.

    Outputs:
        Tuple containing the converted value and its unit label.
    """
    if engine_type == 'B':
        return total_fc, 'kWh'

    fuel_density = FUEL_DENSITY_G_PER_L.get(engine_type)
    if fuel_density is None:
        return total_fc, 'g'

    return total_fc / fuel_density, 'L'


def get_callable_method(owner: object, method_name: str) -> Callable[..., Any]:
    """
    Description:
        Return a dynamically selected method after verifying that it is callable.

    Inputs:
        owner: Object that should contain the requested method.
        method_name: Name of the method to retrieve from the object.

    Outputs:
        Callable method selected from the owner object.
    """
    method = getattr(owner, method_name, None)
    if not callable(method):
        raise AttributeError(f'{method_name} is not callable')

    return cast(Callable[..., Any], method)


class simulations():
    """
    Description:
        Build and run one single-leader CFM validation simulation.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        model_name: Car-following model to validate.
        params: Parameter dictionary for the selected model.
        solver: Numerical solver name.
        dt: Simulation time step in seconds.
        spacing: Whether spacing-based initialization is used.
        delay: Whether the delayed model variant is used.

    Outputs:
        Simulation object containing time, position, and velocity arrays.
    """

    def __init__(self,
                 velocity_class: str,
                 model_name: str,
                 params: dict,
                 solver: str = 'euler',
                 dt: float = 0.1,
                 spacing: bool = True,
                 delay: bool = False) -> None:
        """
        Description:
            Initialize the object and store the parameters needed by later methods.

        Inputs:
            self: Current class instance.
            velocity_class: Leader speed profile or drive-cycle name.
            model_name: Model name used for simulation, filtering, or output naming.
            params: Dictionary of model parameter values.
            solver: Numerical solver name.
            dt: Simulation time step in seconds.
            spacing: Whether spacing-based initialization is used.
            delay: Whether the delayed model variant is used.

        Outputs:
            None. The object is configured in place.
        """
        self.model = model_name
        with open(yaml_file_path, 'r', encoding='utf-8') as file:
            self.model_order = yaml.safe_load(file)['Model_order']

        self.velocity_class = velocity_class
        self.n = 205
        self.dt = dt
        self.simulation_duration = TOTAL_EVALUATION_END_S
        self.total_evaluation_duration_s = TOTAL_EVALUATION_DURATION_S
        self.solver = solver
        self.params_dict = params

        if self.velocity_class in ['WLTC_urban', 'CADC_urban', 'FTP']:

            if self.model == 'idm':
                self.params_dict['des_vel'] = 16.67

        elif self.velocity_class in ['WLTC_rural', 'CADC_rural']:
            if self.model == 'idm':
                self.params_dict['des_vel'] = 22.22

        elif self.velocity_class in ['WLTC_highway', 'CADC_highway', 'HWFET']:
            if self.model == 'fvdm':
                self.params_dict['V2'] = 15
        else:
            pass

        self.equations = Model_equations(self.params_dict)
        self.drive_cycles = {'WLTC_urban': 1,
                            'WLTC_rural': 2,
                            'WLTC_highway': 3,
                            'CADC_urban': 4,
                            'CADC_rural': 5,
                            'CADC_highway': 6,
                            'FTP': 7,
                            'HWFET': 8,
                            'Ford_focus': 9}

        self.unpack_params()
        self.initialise_velocity()
        self.init_spacing()
        self.spacing = spacing
        self.setup_arrays()

    def unpack_params(self) -> None:
        """
        Description:
            Copy parameter values from a dictionary onto the current object.

        Inputs:
            self: Current class instance.

        Outputs:
            None. Parameters are assigned to the current object.
        """
        for key in self.params_dict.keys():
            setattr(self, key, self.params_dict[key])

    def initialise_velocity(self) -> None:

        """
        Description:
            Create the leader velocity profile for the selected drive cycle.

        Inputs:
            self: Current class instance.

        Outputs:
            None. The velocity profile is stored on the current object.
        """
        if self.velocity_class == 'WLTC_rural':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][
                int(588 / self.dt):int(1000 / self.dt)
            ]

        elif self.velocity_class == 'WLTC_urban':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][
                :int(588 / self.dt)
            ]

        elif self.velocity_class == 'WLTC_highway':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][
                int(1000 / self.dt):
            ]

        elif self.velocity_class in self.drive_cycles:
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1]

        else:
            raise IOError('The input type of velocity is not valid')

        velocity = np.asarray(velocity, dtype=float)
        if self.velocity_class == 'Ford_focus':
            self.total_evaluation_duration_s = len(velocity) * self.dt
            self.velocity_leader = np.tile(velocity, 2)
            return

        target_steps = int(round(self.simulation_duration / self.dt))
        repeat = int(np.ceil(target_steps / len(velocity)))

        if repeat > 1:
            self.velocity_leader = np.tile(velocity, repeat)[:target_steps]
        else:
            self.velocity_leader = velocity[:target_steps]

    def init_spacing(self) -> None:

        """
        Description:
            Initialize vehicle spacing from the selected model and speed profile.

        Inputs:
            self: Current class instance.

        Outputs:
            None. Initial spacing is stored on the current object.
        """
        init_vel = self.velocity_leader[0]
        method_name = f'init_spacing_{self.model}'
        init_spacing_function = get_callable_method(self.equations, method_name)
        self.init_spacing_value = init_spacing_function(init_vel)

    def setup_arrays(self) -> None:

        """
        Description:
            Create and initialize the simulation state arrays.

        Inputs:
            self: Current class instance.

        Outputs:
            None. Simulation arrays are stored on the current object.
        """
        len_sim = self.velocity_leader.shape[0]
        self.tim_array = np.arange(0, self.velocity_leader.shape[0]*self.dt, self.dt)
        self.pos_array = np.zeros((self.n, len_sim))
        self.vel_array = np.zeros((self.n, len_sim))


        self.vel_array[0] = self.velocity_leader
        self.vel_array[:,0] = self.velocity_leader[0]

        self.pos_array[:, 0] = -np.arange(self.n) * self.init_spacing_value

    def evolve(self) -> tuple:

        """
        Description:
            Run the simulation and return the generated trajectory arrays.

        Inputs:
            self: Current class instance.

        Outputs:
            Tuple containing time, position, and velocity arrays.
        """

        solver_class = solvers()

        if self.model_order[self.model] == 'second':
            v_equation = get_callable_method(
                self.equations,
                f'u_dot_{self.model}_open',
            )
            x_equation = get_callable_method(
                self.equations,
                f'x_dot_{self.model}_open',
            )

            numerical_solver = get_callable_method(solver_class, 'RK_open_second')

        else:
            v_equation = get_callable_method(
                self.equations,
                f'x_dot_{self.model}_open',
            )
            x_equation = get_callable_method(
                self.equations,
                f'x_dot_{self.model}_open',
            )

            solver_name = f'{self.solver}_open_{self.model_order[self.model]}'
            numerical_solver = get_callable_method(solver_class, solver_name)

        a_old = np.zeros((self.pos_array.shape[0]))
        for count, time in enumerate(self.tim_array[:-2]):

            if numerical_solver.__name__ == 'euler_second_TK':
                x_new, v_new, a_new = numerical_solver(self.pos_array[:,count],
                                                   self.vel_array[:,count],
                                                   a_old,
                                                   self.dt,
                                                   v_equation,
                                                   x_equation)
                a_old = a_new
                a_old[0] = 0 # Leader dynamics is always controlled

            elif numerical_solver.__name__ == 'RK_open_second':

                x_new, v_new = numerical_solver(self.pos_array[:,count],
                                                self.vel_array[:,count],
                                                self.vel_array[:,count + 1],
                                                self.dt,
                                                v_equation,
                                                x_equation)

            elif numerical_solver.__name__ == 'RK_open_first':
                x_new, v_new = numerical_solver(self.pos_array[:,count],
                                                self.vel_array[:,count],
                                                self.vel_array[:,count + 1],
                                                self.dt,
                                                v_equation,
                                                x_equation)

            else:
                raise IOError('The numerical solver is not assigned correctly')


            if np.max(v_new) > 40:
                print('The velocity has blown up at', count, np.max(v_new))
                raise OverflowError()

            if np.min(v_new) < 0:
                print('The velocity went negative', np.min(v_new), count)

                print(v_new)
                raise OverflowError()

            if np.min(x_new[:-1] - x_new[1:]) < -1e-9:

                print('The vehicles have crashed at time step', count)
                raise OverflowError()

            if np.min(x_new - self.pos_array[:,count]) < 0:
                indices = np.where(x_new - self.pos_array[:,count] < 0)[0]
                print(f'The vehicle {indices} moved backward in the iteration {count}')
                raise OverflowError()

            self.pos_array[:, count + 1] = x_new
            self.vel_array[1:, count + 1] = np.maximum(v_new[1:], v_new[1:] * 0)

        return self.tim_array, self.pos_array, self.vel_array

class variation():
    """
    Description:
        Run manuscript validation cases for one single-leader CFM model.

    Inputs:
        model: Name of the car-following model to validate.

    Outputs:
        Variation object used to generate validation trajectories and outputs.
    """

    def __init__(self, model: str) -> None:
        """
        Class to vary the parameters, engine type and driving conditions to estimate emissions

        Parameters
        ----------
        model : str
            Name of the model
        """

        self.model_name = model
        self.param_dict = model_param_dict['Models'][model]
        self.param_ranges = model_param_dict['Parameter_range'][model]

        # To accomodate models with delay

        if self.model_name in ['gipps', 'newell_delay']:
            self.delay = True
        else:
            self.delay = False

    def main_loop(self):
        """
        Description:
            Run the simulation and emissions workflow for the selected model case.

        Inputs:
            self: Current class instance.

        Outputs:
            None. The function is used for its side effects.
        """
        for self.velocity_class in [
                'WLTC_urban',
                'CADC_urban',
                'WLTC_rural',
                'CADC_rural',
                'WLTC_highway']:
            print(f'The leader velocity is {self.velocity_class}')

            self.each_param()
        pass

    def emission_calculation(self):
        """
        Description:
            Run the PHEMlight wrapper and read the generated emissions output.

        Inputs:
            self: Current class instance.

        Outputs:
            None. The function is used for its side effects.
        """
        result = subprocess.run(
            [exe_path, self.engine_type, 'EU6ab'],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print("Execution failed with code:", result.returncode)

    def each_param(self) -> None:
        """
        Description:
            Run the workflow for one parameter value.

        Inputs:
            self: Current class instance.

        Outputs:
            None. The function is used for its side effects.
        """
        temp_dict = self.param_dict.copy()

        simulator = simulations(velocity_class = self.velocity_class,
                                model_name = self.model_name,
                                params = temp_dict,
                                dt = 0.1,
                                delay = self.delay)

        self.t, self.x, self.v = simulator.evolve()
        self.num_veh = self.x.shape[0]
        self.total_evaluation_duration_s = simulator.total_evaluation_duration_s
        print('The simulation was successful')

        self.instantaneous_vehicle_labels = [
            label for label, row in INSTANTANEOUS_VEHICLE_ROWS.items()
            if row < self.num_veh
        ]
        self.instantaneous_vehicle_rows = {
            label: INSTANTANEOUS_VEHICLE_ROWS[label]
            for label in self.instantaneous_vehicle_labels
        }

        self.create_inputs()

        for self.engine_type in ['B', 'G', 'D', 'G_HEV', 'D_HEV']:

            self.emission_calculation()
            self.save_instantaneous_outputs()
            self.save_total_data()
            self.delete_files(type='outputs')

        self.delete_files(type='inputs')

    def create_inputs(self):

        """
        Description:
            Create PHEMlight input CSV files from the simulated trajectories.

        Inputs:
            self: Current class instance.

        Outputs:
            None. Input files are written as a side effect.
        """
        self.delete_files(type='inputs')
        self.delete_files(type='outputs')

        temp_vel = self.v[
            :,
            : self.v.shape[1] // 10 * 10
        ].reshape(self.v.shape[0], -1, 10).mean(axis=2)

        for label, row in self.instantaneous_vehicle_rows.items():
            file_name = f'{self.model_name}_{self.velocity_class}_veh_{label}.csv'
            with open(
                    os.path.join('Iterative_inputs', file_name),
                    mode='w',
                    newline='',
                    encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(temp_vel[row])

    def delete_files(self, type: str):

        """
        Description:
            Remove intermediate PHEM input or output files created by the workflow.

        Inputs:
            self: Current class instance.
            type: Input value used by this function for type.

        Outputs:
            None. Files are deleted as a side effect.
        """
        folder_path = f'Iterative_{type}'
        if not os.path.isdir(folder_path):
            return

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def save_instantaneous_outputs(self):

        """
        Description:
            Save instantaneous emissions and fuel-consumption data to CSV.

        Inputs:
            self: Current class instance.

        Outputs:
            None. CSV files are written as a side effect.
        """
        output_rows = []
        position_1s = self.x[
            :,
            : self.x.shape[1] // 10 * 10
        ].reshape(self.x.shape[0], -1, 10).mean(axis=2)

        for label in self.instantaneous_vehicle_labels:
            output_file = os.path.join(
                'Iterative_outputs',
                f'{self.model_name}_{self.velocity_class}_veh_{label}_Output.csv'
            )

            if not os.path.exists(output_file):
                print(f'Missing PHEM output for vehicle {label}: {output_file}')
                continue

            df = pd.read_csv(output_file, skiprows=[0, 2], index_col=False)
            df.columns = [column.strip() for column in df.columns]

            fc_column = get_fuel_consumption_column(df, self.engine_type, label)
            required_columns = ['Time', 'Speed', 'P_pos', fc_column, 'CO2', 'NOx', 'PM']
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                print(f'Missing PHEM columns for vehicle {label}: {missing_columns}')
                continue

            for column in required_columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

            df = df.dropna(subset=required_columns).reset_index(drop=True)
            if df.empty:
                continue

            simulation_row = self.instantaneous_vehicle_rows[label]
            common_len = min(len(df), position_1s.shape[1])

            vehicle_rows = pd.DataFrame({
                'model': self.model_name,
                'velocity_class': self.velocity_class,
                'engine_type': self.engine_type,
                'vehicle': label,
                'simulation_vehicle': simulation_row,
                'time': df['Time'].to_numpy()[:common_len],
                'speed': df['Speed'].to_numpy()[:common_len],
                'power_pos': df['P_pos'].to_numpy()[:common_len],
                'co2': df['CO2'].to_numpy()[:common_len],
                'nox': df['NOx'].to_numpy()[:common_len],
                'pm': df['PM'].to_numpy()[:common_len],
                'fc': df[fc_column].to_numpy()[:common_len],
                'position': position_1s[simulation_row, :common_len],
            })
            output_rows.append(vehicle_rows)

        if not output_rows:
            return

        output_df = pd.concat(output_rows, ignore_index=True)
        os.makedirs('outputs/instantaneous_data', exist_ok=True)
        output_path = (
            'outputs/instantaneous_data/'
            f'{self.model_name}_{self.velocity_class}_'
            f'{self.engine_type}_instantaneous_outputs.csv'
        )
        output_df.to_csv(
            output_path,
            index=False
        )

    def save_total_data(self):

        """
        Description:
            Save total emissions and fuel-consumption data to CSV.

        Inputs:
            self: Current class instance.

        Outputs:
            None. CSV files are written as a side effect.
        """
        output_rows = []
        position_1s = self.x[
            :,
            : self.x.shape[1] // 10 * 10
        ].reshape(self.x.shape[0], -1, 10).mean(axis=2)

        total_vehicle_rows = {
            label: label
            for label in range(0, min(200, self.num_veh - 1) + 1)
        }
        self.delete_files(type='inputs')
        self.delete_files(type='outputs')
        os.makedirs('Iterative_inputs', exist_ok=True)
        temp_vel = self.v[
            :,
            : self.v.shape[1] // 10 * 10
        ].reshape(self.v.shape[0], -1, 10).mean(axis=2)
        for label, simulation_row in total_vehicle_rows.items():
            file_name = f'{self.model_name}_{self.velocity_class}_veh_{label}.csv'
            with open(
                    os.path.join('Iterative_inputs', file_name),
                    mode='w',
                    newline='',
                    encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(temp_vel[simulation_row])
        self.emission_calculation()

        for label, simulation_row in total_vehicle_rows.items():
            output_file = os.path.join(
                'Iterative_outputs',
                f'{self.model_name}_{self.velocity_class}_veh_{label}_Output.csv'
            )

            if not os.path.exists(output_file):
                print(f'Missing PHEM output for vehicle {label}: {output_file}')
                continue

            df = pd.read_csv(output_file, skiprows=[0, 2], index_col=False)
            df.columns = [column.strip() for column in df.columns]

            fc_column = get_fuel_consumption_column(df, self.engine_type, label)
            required_columns = ['Time', 'Speed', 'P_pos', fc_column, 'CO2', 'NOx', 'PM']
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                print(f'Missing PHEM columns for vehicle {label}: {missing_columns}')
                continue

            for column in required_columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')

            df = df.dropna(subset=required_columns).reset_index(drop=True)
            if df.empty:
                continue

            common_len = min(len(df), position_1s.shape[1])
            df = df.iloc[:common_len].copy()
            df['position'] = position_1s[simulation_row, :common_len]
            moving_rows = df[df['Speed'].abs() > 1e-9]
            if moving_rows.empty:
                os.makedirs('outputs', exist_ok=True)
                with open(
                        'outputs/vehicle_start_warnings.log',
                        mode='a',
                        encoding='utf-8') as log_file:
                    log_file.write(
                        f'WARNING: {self.model_name} {self.velocity_class} '
                        f'{self.engine_type} vehicle {label} '
                        f'(simulation row {simulation_row}) never starts moving.\n'
                    )
                continue

            vehicle_start_time = moving_rows['Time'].iloc[0]
            evaluation_end_time = vehicle_start_time + self.total_evaluation_duration_s
            if evaluation_end_time > df['Time'].max():
                os.makedirs('outputs', exist_ok=True)
                with open(
                        'outputs/vehicle_start_warnings.log',
                        mode='a',
                        encoding='utf-8') as log_file:
                    log_file.write(
                        f'WARNING: {self.model_name} {self.velocity_class} '
                        f'{self.engine_type} vehicle {label} '
                        f'(simulation row {simulation_row}) has only '
                        f'{df["Time"].max() - vehicle_start_time:.3f} s after '
                        f'its start time {vehicle_start_time:.3f} s; requested '
                        f'{self.total_evaluation_duration_s} s.\n'
                    )

            time_mask = (
                (df['Time'] >= vehicle_start_time)
                & (df['Time'] <= evaluation_end_time)
            )
            df = df[time_mask].copy()
            if df.empty:
                continue
            df = df.sort_values('Time').reset_index(drop=True)

            dt = df['Time'].diff().fillna(0)
            total_distance_m = df['position'].max() - df['position'].min()
            total_time_s = df['Time'].iloc[-1] - df['Time'].iloc[0]
            total_co2_g = ((df['CO2'] / 3600) * dt).sum()
            total_nox_g = ((df['NOx'] / 3600) * dt).sum()
            total_pm_g = ((df['PM'] / 3600) * dt).sum()
            total_fc = ((df[fc_column] / 3600) * dt).sum()
            total_fc_converted, fc_converted_unit = convert_total_fc(
                total_fc,
                self.engine_type,
            )
            average_speed = total_distance_m / total_time_s if total_time_s > 0 else np.nan

            output_rows.append({
                'row_type': 'vehicle',
                'model': self.model_name,
                'velocity_class': self.velocity_class,
                'engine_type': self.engine_type,
                'vehicle': label,
                'simulation_vehicle': simulation_row,
                'evaluation_start_s': vehicle_start_time,
                'evaluation_end_s': evaluation_end_time,
                'total_distance_m': total_distance_m,
                'total_time_s': total_time_s,
                'average_speed_mps': average_speed,
                'total_co2_g': total_co2_g,
                'total_nox_g': total_nox_g,
                'total_pm_g': total_pm_g,
                'total_fc': total_fc,
                'total_fc_converted': total_fc_converted,
                'fc_unit': 'kWh' if self.engine_type == 'B' else 'g',
                'fc_converted_unit': fc_converted_unit,
            })

        if not output_rows:
            return

        output_df = pd.DataFrame(output_rows)
        total_row = {
            'row_type': 'fleet_total',
            'model': self.model_name,
            'velocity_class': self.velocity_class,
            'engine_type': self.engine_type,
            'vehicle': '1-200',
            'simulation_vehicle': '',
            'total_distance_m': output_df['total_distance_m'].sum(),
            'total_time_s': output_df['total_time_s'].sum(),
            'average_speed_mps': (
                output_df['total_distance_m'].sum() / output_df['total_time_s'].sum()
                if output_df['total_time_s'].sum() > 0 else np.nan
            ),
            'total_co2_g': output_df['total_co2_g'].sum(),
            'total_nox_g': output_df['total_nox_g'].sum(),
            'total_pm_g': output_df['total_pm_g'].sum(),
            'total_fc': output_df['total_fc'].sum(),
            'total_fc_converted': output_df['total_fc_converted'].sum(),
            'fc_unit': 'kWh' if self.engine_type == 'B' else 'g',
            'fc_converted_unit': output_df['fc_converted_unit'].iloc[0],
        }
        output_df = pd.concat([output_df, pd.DataFrame([total_row])], ignore_index=True)
        os.makedirs('outputs/total_data', exist_ok=True)
        output_path = (
            'outputs/total_data/'
            f'{self.model_name}_{self.velocity_class}_{self.engine_type}_total_data.csv'
        )
        output_df.to_csv(
            output_path,
            index=False
        )

    def save_total_emissions(self):
        """
        Description:
            Save total emissions for the configured validation case.

        Inputs:
            self: Current class instance.

        Outputs:
            None. CSV files are written as a side effect.
        """
        return self.save_total_data()

if __name__ == '__main__':
    for model in ['idm', 'fvdm']:
        emissions = variation(model = model)
        emissions.main_loop()

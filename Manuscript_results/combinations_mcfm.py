"""
Run MCFM parameter combinations and save PHEMlight emissions outputs.
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

# Custom library import

import leader_velocity as lv
from model_equations import Model_equations
from solvers import solvers

TOTAL_EVALUATION_START_S = 800
TOTAL_EVALUATION_END_S = 2600
TOTAL_EVALUATION_DURATION_S = 1800
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
        Build and run one multi-leader CFM trajectory simulation.

    Inputs:
        velocity_class: Leader speed profile or drive-cycle name.
        model_name: Multi-leader model to simulate.
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

        if self.model == 'midm':

            self.params_dict['m'] = 3
            self.params_dict['spacing_weight'] = np.array([0.7, 0.2, 0.1])
            self.params_dict['relvel_weight'] = np.array([0.7, 0.2, 0.1])

        elif self.model == 'mfvdm':

            self.params_dict['m'] = 3
            self.params_dict['weight'] = np.array([0.7, 0.2, 0.1])
            self.params_dict['p'] = 0.6
            self.params_dict['q'] = 0.4


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
        len_sim = len(self.velocity_leader)
        self.tim_array = np.arange(0, len_sim*self.dt, self.dt)
        self.pos_array = np.zeros((self.n, len_sim))
        self.vel_array = np.zeros((self.n, len_sim))


        self.vel_array[self.m - 1] = self.velocity_leader
        self.vel_array[:, 0] = self.velocity_leader[0]

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

        if self.model == 'midm':

            relvel = (self.vel_array[self.m-1, 1] - self.vel_array[self.m, 1])
            self.vel_array[0, 1] = self.vel_array[self.m-1, 1] + 2*relvel
            self.vel_array[1, 1] = self.vel_array[self.m-1, 1] + relvel
            gap = (self.pos_array[self.m-1, 1] - self.pos_array[self.m, 1])
            self.pos_array[0, 1] = self.pos_array[self.m-1, 1] + 2*gap
            self.pos_array[1, 1] = self.pos_array[self.m-1, 1] + gap

        elif self.model == 'mfvdm':
            self.vel_array[0, 1] = self.vel_array[self.m-1, 1]
            self.vel_array[1, 1] = self.vel_array[self.m-1, 1]


        for count, time in enumerate(self.tim_array[:-2]):

            if numerical_solver.__name__ == 'euler_second_TK':
                x_new, v_new, a_new = numerical_solver(self.pos_array[:,count],
                                                   self.vel_array[:,count],
                                                   a_old,
                                                   self.dt,
                                                   v_equation,
                                                   x_equation)
                a_old = a_new
                a_old[:self.m] = 0 # Leader dynamics is always controlled

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


            if np.max(v_new[2:]) > 40:
                print('The velocity has blown up at', count, np.max(v_new))
                raise OverflowError()

            if np.min(v_new) < 0:
                print('The velocity went negative', np.min(v_new), count)
                raise OverflowError()

            if np.min(x_new[self.m:-1] - x_new[1+self.m:]) < -1e-9:

                print('The vehicles have crashed at time step', count)


                raise OverflowError()

            if np.min(x_new[self.m:] - self.pos_array[self.m:,count]) < 0:
                indices = np.where(x_new - self.pos_array[:,count] < 0)[0]
                print(f'The vehicle {indices} moved backward in the iteration {count}')
                raise OverflowError()

            self.pos_array[self.m - 1:, count + 1] = x_new[self.m - 1:]
            self.vel_array[self.m:, count + 1] = np.maximum(
                v_new[self.m:],
                v_new[self.m:] * 0,
            )

            if self.model == 'midm':
                gap = (self.pos_array[self.m-1, count+1] - self.pos_array[self.m, count+1])
                relvel = (self.vel_array[self.m-1, count+1] - self.vel_array[self.m, count+1])
                self.pos_array[0, count+1] = self.pos_array[self.m-1, count+1] + 2*gap
                self.pos_array[1, count+1] = self.pos_array[self.m-1, count+1] + gap
                self.vel_array[0, count+1] = self.vel_array[self.m-1, count+1] + 2*relvel
                self.vel_array[1, count+1] = self.vel_array[self.m-1, count+1] + relvel

            elif self.model == 'mfvdm':
                gap = (self.pos_array[self.m-1, count+1] - self.pos_array[self.m, count+1])
                self.pos_array[0, count+1] = self.pos_array[self.m-1, count+1] + 2*gap
                self.pos_array[1, count+1] = self.pos_array[self.m-1, count+1] + gap
                self.vel_array[0, count+1] = self.vel_array[self.m-1, count+1]
                self.vel_array[1, count+1] = self.vel_array[self.m-1, count+1]

        return self.tim_array, self.pos_array, self.vel_array

# Construct the full path to the C# executable
exe_path = os.path.join("New_PhemApp", "bin", "Release", "net9.0", "New_PhemApp.exe")

class variation():
    """
    Description:
        Iterate through parameter, vehicle, engine, and drive-cycle
        combinations for one multi-leader CFM model.

    Inputs:
        model: Name of the multi-leader car-following model to process.

    Outputs:
        Variation object used to generate trajectories and emission files.
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
                'WLTC_rural',
                'WLTC_highway',
                'CADC_rural',
                'CADC_urban',
                'CADC_highway',
                'FTP',
                'HWFET',
                'Ford_focus']:
            print(f'The leader velocity is {self.velocity_class}')

            for param in self.param_dict.keys():
                self.each_param(param)

        pass

    def each_param(self, param_name: str) -> None:
        """
        Description:
            Run the workflow for one parameter value.

        Inputs:
            self: Current class instance.
            param_name: Input value used by this function for param name.

        Outputs:
            None. The function is used for its side effects.
        """
        temp_dict = self.param_dict.copy()

        if param_name in self.param_ranges.keys():
            # The range is defined
            print(f'Processing the parameter {param_name}')
            ind_param_range = np.linspace(
                self.param_ranges[param_name][0],
                self.param_ranges[param_name][1],
                15,
            )
            for param in ind_param_range:
                self.varying_param = param
                temp_dict[param_name] = param
                self.varying_param_value = temp_dict[param_name]

                try:
                    simulator = simulations(
                        velocity_class=self.velocity_class,
                        model_name=self.model_name,
                        params=temp_dict,
                        dt=0.1,
                        delay=self.delay,
                    )

                    self.t, self.x, self.v = simulator.evolve()
                    self.num_veh = self.x.shape[0]
                    self.total_evaluation_duration_s = simulator.total_evaluation_duration_s
                    print('The simulation was successful')

                    for self.engine_type in ['D', 'D_HEV', 'G', 'G_HEV', 'B']:
                        self.save_total_data(param_name)

                except Exception as exc:
                    print(
                        f'The simulation did not run for the parameter '
                        f'{param_name} = {param}: {exc}'
                    )

            self.delete_files(type='inputs')
            self.delete_files(type='outputs')

    def create_inputs(self):

        """
        Method to create input files for each vehicle
        """
        temp_vel = self.v[
            :,
            : self.v.shape[1] // 10 * 10
        ].reshape(self.v.shape[0], -1, 10).mean(axis=2)
        # temp_vel = self.v.reshape(self.v.shape[0], -1, int(1/0.1)).mean(axis=2)

        for i in range(self.num_veh):
            with open(
                    rf'Iterative_inputs\{self.model_name}_vel_{i}.csv',
                    mode='w',
                    newline='',
                    encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(temp_vel[i])

    def emission_calculation(self):
        # Run the executable
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
        for folder_path in [f'Iterative_{type}']:
            if not os.path.isdir(folder_path):
                return

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def save_total_data(self, param_name):

        """
        Description:
            Save total emissions and fuel-consumption data to CSV.

        Inputs:
            self: Current class instance.
            param_name: Input value used by this function for param name.

        Outputs:
            None. CSV files are written as a side effect.
        """
        output_rows = []
        position_1s = self.x[
            :,
            : self.x.shape[1] // 10 * 10
        ].reshape(self.x.shape[0], -1, 10).mean(axis=2)

        leader_count = self.param_dict.get('m', 3)
        total_vehicle_rows = {
            0: leader_count - 1,
            **{
                label: label + leader_count - 1
                for label in range(1, min(200, self.num_veh - leader_count) + 1)
            }
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
                        f'{self.engine_type} {param_name}={self.varying_param_value} '
                        f'vehicle {label} (simulation row {simulation_row}) '
                        f'never starts moving.\n'
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
                        f'{self.engine_type} {param_name}={self.varying_param_value} '
                        f'vehicle {label} (simulation row {simulation_row}) has only '
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
            max_speed_mps = df['Speed'].max()
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
                'parameter': param_name,
                'parameter_value': self.varying_param_value,
                'velocity_class': self.velocity_class,
                'engine_type': self.engine_type,
                'vehicle': label,
                'simulation_vehicle': simulation_row,
                'evaluation_start_s': vehicle_start_time,
                'evaluation_end_s': evaluation_end_time,
                'total_distance_m': total_distance_m,
                'total_time_s': total_time_s,
                'average_speed_mps': average_speed,
                'max_speed_mps': max_speed_mps,
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
            'parameter': param_name,
            'parameter_value': self.varying_param_value,
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
            'max_speed_mps': output_df['max_speed_mps'].max(),
            'total_co2_g': output_df['total_co2_g'].sum(),
            'total_nox_g': output_df['total_nox_g'].sum(),
            'total_pm_g': output_df['total_pm_g'].sum(),
            'total_fc': output_df['total_fc'].sum(),
            'total_fc_converted': output_df['total_fc_converted'].sum(),
            'fc_unit': 'kWh' if self.engine_type == 'B' else 'g',
            'fc_converted_unit': output_df['fc_converted_unit'].iloc[0],
        }
        output_df = pd.concat([output_df, pd.DataFrame([total_row])], ignore_index=True)
        os.makedirs('outputs/master_data', exist_ok=True)
        output_path = f'outputs/master_data/{self.model_name}_total_data.csv'
        output_df.to_csv(
            output_path,
            mode='a',
            header=not os.path.exists(output_path),
            index=False,
        )

    def post_processing(self, param_name):

        """
        Description:
            Read PHEMlight outputs and append processed emissions rows.

        Inputs:
            self: Current class instance.
            param_name: Input value used by this function for param name.

        Outputs:
            None. The function is used for its side effects.
        """
        cols = [' Speed', 'CO2', 'NOx', 'CO', 'PM', 'HC', 'FC']
        avg_names = [i + '_avg' for i in cols]
        a = pd.DataFrame(
            columns=[
                'model',
                'vehicle',
                'standard',
                'engine_type',
                'condition',
                'parameter',
                'value',
            ] + avg_names
        )

        if not os.path.exists(f'{self.model_name}_emission_details.csv'):
            a.to_csv(f'{self.model_name}_emission_details.csv', index=False)

        for i in range(0, self.num_veh):
            output_path = rf'Iterative_outputs\{self.model_name}_vel_{i}_Output.csv'
            df = pd.read_csv(
                output_path,
                skiprows=[0, 2],
                index_col=False,
            )
            df = df[df['Time'] > 1200]
            df = df[cols]

            df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            averages = df.mean().to_dict()

            # Build result row
            row = {
                'model': self.model_name,
                'vehicle': i,
                'standard': 'EU6ab',
                'engine_type': self.engine_type,
                'condition': self.velocity_class,
                'param': param_name,
                'param_value': self.varying_param_value
            }
            row.update({f'{k}_avg': v for k, v in averages.items()})

            pd.DataFrame([row]).to_csv(
                f'{self.model_name}_emission_details.csv',
                mode='a',
                header=False,
                index=False,
            )

if __name__ == '__main__':

    for model in ['mfvdm', 'midm']:
        emissions = variation(model = model)
        emissions.main_loop()

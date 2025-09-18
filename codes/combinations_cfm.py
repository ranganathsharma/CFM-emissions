"""
This file contains the code for varying the engine type, driving conditions and model parameters and obtaining the emissions using PHEMlight.
"""

# Import the parameters file

import os, sys, yaml, csv
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')

with open(yaml_file_path, 'r') as file:
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

class simulations():

    def __init__(self,
                 velocity_class: str,
                 model_name: str,
                 params: dict,
                 solver: str = 'euler',
                 dt: float = 0.1,
                 spacing: bool = True,
                 delay: bool = False) -> None:
        self.model = model_name
        with open(yaml_file_path, 'r') as file:
            self.model_order = yaml.safe_load(file)['Model_order']

        self.velocity_class = velocity_class
        self.n = 1000
        self.dt = dt
        self.solver = solver
        self.params_dict = params
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
        for key in self.params_dict.keys():
            setattr(self, key, self.params_dict[key])

    def initialise_velocity(self) -> None:

        if self.velocity_class == 'WLTC_rural':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][int(588/self.dt):int(1000/self.dt)]
            
        elif self.velocity_class == 'WLTC_urban':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][:int(588/self.dt)]

        elif self.velocity_class == 'WLTC_highway':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1][int(1000/self.dt):]

        elif self.velocity_class in self.drive_cycles.keys():
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1]

        else:
            raise IOError('The input type of velocity is not valid')
        
        repeat = int(np.ceil(7600/self.dt/len(velocity)))

        if repeat > 1:
            self.velocity_leader = np.tile(velocity, repeat)
        else:
            self.velocity_leader = velocity
   
    def init_spacing(self) -> None:

        init_vel = self.velocity_leader[0]
        init_spacing_function = getattr(self.equations, 'init_spacing_' + str(self.model), None)

        if callable(init_spacing_function):
            try:
                self.init_spacing_value = init_spacing_function(init_vel)
            except:
                raise OverflowError()
        else:
            raise IOError('Function error: init_spacing function is not callable')
        
    def setup_arrays(self) -> None:

        len_sim = self.velocity_leader.shape[0]
        self.tim_array = np.arange(0, self.velocity_leader.shape[0]*self.dt, self.dt)
        self.pos_array = np.zeros((self.n, len_sim))
        self.vel_array = np.zeros((self.n, len_sim))
        

        self.vel_array[0] = self.velocity_leader
        self.vel_array[:,0] = self.velocity_leader[0]

        try:
            for i in range(self.n):
                self.pos_array[i,0] = -i*self.init_spacing_value

        except:
            raise IOError('The input spacing is not valid')

    def evolve(self) -> tuple:
        
        """_summary_

        Returns
        -------
        tuple
            Three numpy arrays - time, position and velocity. 
            time - 1d array
            position - 2d array: rows - vehicle, column - time
            velocity - 2d array: rows - vehicle, column - time

        Raises
        ------
        OverflowError
            First condition for unrealistic velocity
        OverflowError
            Second condition for accidents
        """

        solver_class = solvers()

        if self.model_order[self.model] == 'second':
            v_equation = getattr(self.equations, 'u_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')

            # numerical_solver = getattr(solver_class, 'euler_second_TK')
            numerical_solver = getattr(solver_class, 'RK_open_second')

        else:
            v_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')

            numerical_solver = getattr(solver_class, self.solver + '_open_' + self.model_order[self.model]) 

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
            
            self.pos_array[:, count + 1], self.vel_array[1:, count + 1] = x_new, np.maximum(v_new[1:], v_new[1:]*0)

        return self.tim_array, self.pos_array, self.vel_array

# Construct the full path to the C# executable
exe_path = os.path.join("PhemApp", "bin", "Release", "net9.0", "PhemApp.exe")

class variation():

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
        # ['WLTC_urban', 'WLTC_rural' , 'WLTC_highway', 'CADC_rural', 'CADC_urban', 'CADC_highway', 'FTP', 'HWFET']
        for self.velocity_class in ['Ford_focus']:
            print(f'The leader velocity is {self.velocity_class}')
            
            for param in self.param_dict.keys():
                self.each_param(param)

        pass

    def each_param(self, param_name: str) -> None:
        temp_dict = self.param_dict.copy()

        if param_name in self.param_ranges.keys():
            # The range is defined
            print(f'Processing the parameter {param_name}')
            ind_param_range = np.linspace(self.param_ranges[param_name][0], self.param_ranges[param_name][1], 15)
            for param in ind_param_range:
                self.varying_param = param
                temp_dict[param_name] = param
                self.varying_param_value = temp_dict[param_name]

                try:
                    simulator = simulations(velocity_class = self.velocity_class, 
                                            model_name = self.model_name, 
                                            params = temp_dict, 
                                            dt = 0.1,
                                            delay = self.delay)
                
                
                    
                    self.t, self.x, self.v = simulator.evolve()
                    self.num_veh = self.x.shape[0]
                    print('The simulation was successful')
                    self.create_inputs()

                    for self.engine_type in ['D', 'D_HEV', 'G', 'G_HEV']:

                        try:
                            self.emission_calculation()
                            self.post_processing(param_name)
                        except:
                            print(f'The engine type {self.engine_type} could not be processed.')

                except:
                    print(f'The simulation did not run for the parameter {param_name} = {param}')

            self.delete_files(type = 'inputs')
            self.delete_files(type = 'outputs')
            
    def create_inputs(self):

        """
        Method to create input files for each vehicle
        """
        temp_vel = self.v[:, :self.v.shape[1] // 10 * 10].reshape(self.v.shape[0], -1, 10).mean(axis=2)
        # temp_vel = self.v.reshape(self.v.shape[0], -1, int(1/0.1)).mean(axis=2)

        for i in range(self.num_veh):
            with open(rf'Iterative_inputs\{self.model_name}_vel_{i}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(temp_vel[i])

    def emission_calculation(self):
        # Run the executable
        result = subprocess.run([exe_path, self.engine_type, 'EU6ab'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Execution failed with code:", result.returncode)
    
    def delete_files(self, type: str):
        for folder_path in [f'Iterative_{type}']:

            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def post_processing(self, param_name):

        cols = [' Speed', 'CO2', 'NOx', 'CO', 'PM', 'HC', 'FC']
        avg_names = [i + '_avg' for i in cols]
        a = pd.DataFrame(columns = ['model', 'vehicle', 'standard', 'engine_type', 'condition', 'parameter', 'value'] + avg_names)

        if not os.path.exists(f'{self.model_name}_emission_details.csv'):
            a.to_csv(f'{self.model_name}_emission_details.csv', index=False)

        for i in range(0, self.num_veh):
            first_row = pd.read_csv(rf'Iterative_outputs\{self.model_name}_vel_{i}_Output.csv', nrows=1, header=None)

            df = pd.read_csv(rf'Iterative_outputs\{self.model_name}_vel_{i}_Output.csv', skiprows = [0,2], index_col=False)
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

            pd.DataFrame([row]).to_csv(f'{self.model_name}_emission_details.csv', mode = 'a', header = False, index = False)

if __name__ == '__main__':

    for model in ['fvdm', 'idm', 'gfm', 'tsh', 'bando']:                      
        emissions = variation(model = model)
        emissions.main_loop()
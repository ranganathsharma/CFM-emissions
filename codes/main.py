# File with the class simulations whose input are the conditions of run and the output is the trajectory data for the model

# Accessing files in the master folder
import os, sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Importing parameters file
try:
    yaml_file_path = os.path.join(parent_dir, 'parameters.yaml')
except:
    raise SystemExit('Path Error: The parameters.yaml file could not be found')

# Library imports

import numpy as np
import pandas as pd
import yaml
import leader_velocity as lv
from model_equations import Model_equations
from solvers import solvers

class simulations():


    def __init__(self,
                 velocity_class: int,
                 model_name :str,
                 params: dict,
                 solver: str = 'euler',
                 dt: float = 0.1, 
                 spacing: bool = True,
                 delay: bool = False) -> None:
        """_summary_

        Parameters
        ----------
        velocity_class : int
            velocity class 1: laboratory drive cycle
            2: wltc 2
            3: wltc 3
            4: laboratory temporary speed increase
            5: laboratory temporary speed decrease
            6: custom speed profile
        model_name : str
            Name of the model
        params : dict
            dictionary of the model parameters
        solver : str, optional
            name of the numerical solver, by default 'euler'
        dt : float, optional
            simulation time step, by default 0.1
        spacing : bool, optional
            initial spacing, by default True
        """


        self.model = model_name

        with open(yaml_file_path, 'r') as file:
            self.model_order = yaml.safe_load(file)['Model_order']

        self.velocity_class = velocity_class
        self.n = 2000
        self.dt = dt
        self.num_rep = 4 # Number of times the leader cycle is repeated to simulate the system for 2 hours.
        self.solver = solver
        self.params_dict = params
        self.equations = Model_equations(self.params_dict)
        self.unpack_params()
        self.initialise_velocity()
        self.init_spacing()
        self.spacing = spacing
        self.setup_arrays()
        
    def unpack_params(self) -> None:
        for key in self.params_dict.keys():
            setattr(self, key, self.params_dict[key])

    def initialise_velocity(self) -> None:

        if self.velocity_class == 1:
            # Laboratory velocity profile with fluctuations
            self.velocity_leader = lv.vel_leader_rand(1, 1, 0.5, self.dt)[1]
        elif self.velocity_class in [2, 3]:
            # WLTC velocity profiles
            velocity = lv.wltc(self.velocity_class, self.dt)[1]
            self.velocity_leader = np.tile(velocity, self.num_rep)
        elif self.velocity_class == 4:
            # Laboratory velocity profile with temporary increase
            self.velocity_leader = lv.vel_lead_acc(self.dt)[1]
        elif self.velocity_class == 5:
            # Laboratory velocity profile with temporary decrease
            self.velocity_leader = lv.vel_lead_dec(self.dt)[1]
        elif self.velocity_class == 6:
            # Custom velocity profile
            pass
        else:
            raise SystemExit('Input Error: Invalid leader velocity profile')

    def init_spacing(self) -> None:

        init_vel = self.velocity_leader[0]
        init_spacing_function = getattr(self.equations, 'init_spacing_' + str(self.model), None)

        if callable(init_spacing_function):
            self.init_spacing_value = init_spacing_function(init_vel)
        else:
            raise SystemExit('Function error: init_spacing function is not callable')
        
    def setup_arrays(self) -> None:

        len_sim = self.velocity_leader.shape[0]
        self.tim_array = np.arange(0, self.velocity_leader.shape[0]*self.dt, self.dt)
        self.pos_array = np.zeros((self.n, len_sim))
        self.vel_array = np.zeros((self.n, len_sim))
        

        self.vel_array[0] = self.velocity_leader
        self.vel_array[:,0] = self.velocity_leader[0]

        if self.spacing:
            for i in range(self.n):
                self.pos_array[i,0] = -i*self.init_spacing_value

        else:
            for i in range(1, self.n):
                self.pos_array[i,0] = self.pos_array[i-1,0] - np.random.uniform(6, 100)

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

        if self.model_order[self.model] == 'second':
            v_equation = getattr(self.equations, 'u_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')
        else:
            v_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')
            x_equation = getattr(self.equations, 'x_dot_' + self.model + '_open')

        solver_class = solvers()
        
        numerical_solver = getattr(solver_class, self.solver + '_open_' + self.model_order[self.model])

        for count, time in enumerate(self.tim_array[:-2]):
            
            x_new, v_new = numerical_solver(self.pos_array[:,count],
                                            self.vel_array[:,count],
                                            self.vel_array[:,count + 1],
                                            self.dt,
                                            v_equation,
                                            x_equation)
            
            if np.max(v_new) > 40:
                print('The velocity has blown up at', count, np.max(v_new))
                raise OverflowError()
            
            if np.min(v_new) < 0:
                print('The velocity went negative', np.min(v_new))
                raise OverflowError()

            if np.min(x_new[:-1] - x_new[1:]) < -1e-9:

                # indices = np.where(x_new[:-1] - x_new[1:] < -1e-9)
                # fig, ax = plt.subplots(figsize=(8, 6), dpi = 600)
                # # Main plot
                # ax.plot(self.tim_array[:count], self.pos_array[0, :count], label='Leader')
                # ax.plot(self.tim_array[:count], self.pos_array[1, :count], label='Follower')
                # ax.set_xlabel('Time (s)', fontsize=15)
                # ax.set_ylabel('Position (m)', fontsize=15)
                # ax.legend(fontsize=10)
                # ax.tick_params(axis='both', which='major', labelsize=15)
                # ax.grid(True)

                # # Inset zoom
                # x_zoom = (465, 480)
                # y_zoom = (2400, 2410)

                # inset_ax = inset_axes(
                #     ax,
                #     width="40%",
                #     height="40%",
                #     loc="lower right",
                #     bbox_to_anchor=(-0.05, 0.1, 1, 1),  # Move inset upwards with bbox_to_anchor
                #     bbox_transform=ax.transAxes,
                #     borderpad=0,
                # )
                # inset_ax.plot(self.tim_array[:count], self.pos_array[0, :count], label='Leader')
                # inset_ax.plot(self.tim_array[:count], self.pos_array[1, :count], label='Follower')
                # inset_ax.set_xlim(x_zoom)
                # inset_ax.set_ylim(y_zoom)
                # inset_ax.tick_params(axis='both', which='major', labelsize=10)
                # inset_ax.grid(True)

                # # Connect the inset to the main plot
                # mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5", linestyle="--")

                # # plt.tight_layout()
                # plt.savefig('Shamoto_crash.png', bbox_inches = 'tight')
                # plt.close()

                print('The vehicles have crashed at time step', count)
                raise OverflowError()
            
            if np.min(x_new - self.pos_array[:,count]) < -1e-9:
                indices = np.where(x_new - self.pos_array[:,count] < 0)[0]
                print(f'The vehicle {indices} moved backward in the iteration {count}')
                
                raise OverflowError()
                
            self.pos_array[:, count + 1], self.vel_array[1:, count + 1] = x_new, np.maximum(v_new[1:], v_new[1:]*0)

        return self.tim_array, self.pos_array, self.vel_array

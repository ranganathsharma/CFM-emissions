"""
This file contains a demonstration of solving the car following model equations using the
parameters defined in the parameters.yaml file.
"""

# Short names such as x, v, and dt are standard in this simulation code.
# The project imports also depend on adding the repository root to sys.path.
# pylint: disable=invalid-name,wrong-import-position

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

YAML_FILE_PATH = REPO_ROOT / 'parameters.yaml'

import leader_velocity as lv
from model_equations import Model_equations
from solvers import solvers


with open(YAML_FILE_PATH, 'r', encoding='utf-8') as file:
    MODEL_PARAM_DICT = yaml.safe_load(file)

TOTAL_EVALUATION_END_S = 500
TOTAL_EVALUATION_DURATION_S = 500


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


class simulations:  # pylint: disable=too-many-instance-attributes,too-many-arguments
    """
    Description:
        Run one demonstration car-following simulation for a selected model.

    Inputs:
        None. Configure instances through the constructor arguments.

    Outputs:
        Simulation object with generated time, position, and velocity arrays.
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
        with open(YAML_FILE_PATH, 'r', encoding='utf-8') as file:
            self.model_order = yaml.safe_load(file)['Model_order']

        self.velocity_class = velocity_class
        self.n = 20
        self.dt = dt
        self.simulation_duration = TOTAL_EVALUATION_END_S
        self.total_evaluation_duration_s = TOTAL_EVALUATION_DURATION_S
        self.solver = solver
        self.params_dict = params
        self.delay = delay
        self.equations = Model_equations(self.params_dict)
        self.drive_cycles = {
            'WLTC_urban': 1,
            'WLTC_rural': 2,
            'WLTC_highway': 3,
            'CADC_urban': 4,
            'CADC_rural': 5,
            'CADC_highway': 6,
            'FTP': 7,
            'HWFET': 8,
            'Ford_focus': 9,
        }
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
        for key in self.params_dict:
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
            velocity = lv.wltc(self.drive_cycles[self.velocity_class],
                               self.dt)[1][int(588/self.dt):int(1000/self.dt)]

        elif self.velocity_class == 'WLTC_urban':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class],
                               self.dt)[1][:int(588/self.dt)]

        elif self.velocity_class == 'WLTC_highway':
            velocity = lv.wltc(self.drive_cycles[self.velocity_class],
                               self.dt)[1][int(1000/self.dt):]

        elif self.velocity_class in self.drive_cycles:
            velocity = lv.wltc(self.drive_cycles[self.velocity_class], self.dt)[1]

        else:
            raise OSError('The input type of velocity is not valid')

        velocity = np.asarray(velocity, dtype=float)
        if self.velocity_class == 'Ford_focus':
            self.total_evaluation_duration_s = len(velocity) * self.dt
            self.velocity_leader = np.tile(velocity, 2)
            return

        target_steps = int(round(self.simulation_duration / self.dt))
        repeat = int(np.ceil(target_steps / len(velocity)))

        if repeat > 1:
            if self.velocity_class != 'Ford_focus':
                self.velocity_leader = np.tile(velocity, repeat)[:target_steps]
            else:
                self.velocity_leader = np.tile(velocity, repeat)
        else:
            if self.velocity_class != 'Ford_focus':
                self.velocity_leader = velocity[:target_steps]
            else:
                self.velocity_leader = np.tile(velocity, repeat)

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

        self.vel_array[0] = self.velocity_leader
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

        for count in range(len(self.tim_array[:-2])):
            x_new, v_new = numerical_solver(
                self.pos_array[:, count],
                self.vel_array[:, count],
                self.vel_array[:, count + 1],
                self.dt,
                v_equation,
                x_equation,
            )

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

            if np.min(x_new - self.pos_array[:, count]) < 0:
                indices = np.where(x_new - self.pos_array[:, count] < 0)[0]
                print(f'The vehicle {indices} moved backward in the iteration {count}')
                raise OverflowError()

            self.pos_array[:, count + 1] = x_new
            self.vel_array[1:, count + 1] = np.maximum(v_new[1:], v_new[1:] * 0)

        return self.tim_array, self.pos_array, self.vel_array


if __name__ == '__main__':

    VELOCITY_CLASS = 'WLTC_rural'
    MODEL_NAME = 'idm'
    PARAM_DICT = MODEL_PARAM_DICT['Models'][MODEL_NAME]

    simulator = simulations(
        velocity_class=VELOCITY_CLASS,
        model_name=MODEL_NAME,
        params=PARAM_DICT,
        dt=0.1,
        delay=False,
    )

    t, x, v = simulator.evolve()
    print('The simulation was successful')

    plt.plot(t, v[0], label='Leader')
    plt.plot(t, v[1], label='Follower 1')
    plt.plot(t, v[10], label='Follower 10')
    plt.xlabel('Time (s)', fontsize=15)
    plt.ylabel('Velocity (m/s)', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlim([0, 200])
    (DEMO_DIR / 'outputs').mkdir(exist_ok=True)
    plt.savefig(DEMO_DIR / 'outputs/idm_wltc_rural_velocity.png', dpi=600, bbox_inches='tight')
    plt.close()

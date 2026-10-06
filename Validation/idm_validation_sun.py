"""Compare a local IDM Euler update with the solver-class Euler update."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_DIR = Path(__file__).resolve().parent.parent
PARAMETER_FILE = REPO_DIR / "parameters.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_TIME_STEP = 0.01
MODEL_NAME = "idm"
SOLVER_NAME = "euler"

# Allow this validation file to be run directly from inside the Validation folder.
sys.path.insert(0, str(REPO_DIR))

from model_equations import Model_equations  # noqa: E402
from solvers import solvers  # noqa: E402


def get_callable_method(owner, method_name):
    """Return a named method after checking that it is callable.

    Parameters
    ----------
    owner : object
        Object that should contain the requested method.
    method_name : str
        Name of the method to retrieve.

    Returns
    -------
    Callable
        Method selected from the owner object.
    """
    method = getattr(owner, method_name, None)
    if not callable(method):
        raise AttributeError(f"{method_name} is not callable")

    return method


def load_parameters(parameter_file=PARAMETER_FILE):
    """Load model parameters and model-order metadata from parameters.yaml.

    Parameters
    ----------
    parameter_file : str or pathlib.Path
        Path to the repository ``parameters.yaml`` file.

    Returns
    -------
    tuple[dict, dict]
        IDM parameter dictionary and model-order dictionary.
    """
    with open(parameter_file, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    return parameters["Models"][MODEL_NAME], parameters["Model_order"]


def create_leader_velocity(time_array):
    """Create the imposed leader velocity used by both Euler solutions.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.

    Returns
    -------
    numpy.ndarray
        Leader velocity in m/s at each time stamp.
    """
    velocity = np.zeros_like(time_array)

    # The same smooth leader profile is used in both methods, so any difference
    # comes from the numerical update and not from the input trajectory.
    velocity += np.where(time_array <= 30, time_array / 30 * 15, 0)
    velocity += np.where((time_array > 30) & (time_array <= 80), 15, 0)
    velocity += np.where(
        (time_array > 80) & (time_array <= 120),
        15 - (time_array - 80) / 40 * 8,
        0,
    )
    velocity += np.where(time_array > 120, 7, 0)

    return velocity


class IDMSunValidationSimulation:
    """Small simulation-flow wrapper for IDM Euler validation.

    Parameters
    ----------
    params : dict
        IDM parameter dictionary.
    model_order : dict
        Mapping from model name to numerical order.
    dt : float
        Simulation time step in seconds.
    duration : float
        Simulation duration in seconds.
    n_vehicles : int
        Number of vehicles, including the leader.
    solver : str
        Solver prefix used to build the solver method name.

    Returns
    -------
    None
        The object stores the simulation arrays and model functions.
    """

    def __init__(
        self,
        params,
        model_order,
        dt=DEFAULT_TIME_STEP,
        duration=150.0,
        n_vehicles=20,
        solver=SOLVER_NAME,
    ):
        """Configure the IDM validation simulation and allocate arrays.

        Parameters
        ----------
        params : dict
            IDM parameter dictionary loaded from ``parameters.yaml``.
        model_order : dict
            Mapping from model name to numerical order.
        dt : float
            Simulation time step in seconds.
        duration : float
            Simulation duration in seconds.
        n_vehicles : int
            Number of vehicles in the validation platoon.
        solver : str
            Solver prefix used to select the method from ``solvers.py``.

        Returns
        -------
        None
            The object is initialized with parameters, arrays, and model methods.
        """
        self.model = MODEL_NAME
        self.params_dict = params.copy()
        self.model_order = model_order
        self.dt = dt
        self.duration = duration
        self.n = n_vehicles
        self.solver = solver
        self.equations = Model_equations(self.params_dict)

        self.unpack_params()
        self.initialise_velocity()
        self.init_spacing()
        self.setup_arrays()

    def unpack_params(self):
        """Copy IDM parameter values onto the simulation object.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Parameters are assigned to the current object.
        """
        for key, value in self.params_dict.items():
            setattr(self, key, value)

    def initialise_velocity(self):
        """Create the imposed leader velocity profile.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Time and leader-velocity arrays are stored on the object.
        """
        self.tim_array = np.arange(0, self.duration + self.dt, self.dt)
        self.velocity_leader = create_leader_velocity(self.tim_array)

    def init_spacing(self):
        """Calculate initial IDM equilibrium spacing.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initial equilibrium spacing is stored on the object.
        """
        init_spacing = get_callable_method(
            self.equations,
            f"init_spacing_{self.model}",
        )
        self.init_spacing_value = init_spacing(self.velocity_leader[0])

    def setup_arrays(self):
        """Create and initialize position and velocity arrays.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Position and velocity arrays are stored on the object.
        """
        n_steps = self.velocity_leader.shape[0]
        self.pos_array = np.zeros((self.n, n_steps))
        self.vel_array = np.zeros((self.n, n_steps))

        self.pos_array[:, 0] = -np.arange(self.n) * self.init_spacing_value
        self.vel_array[:, 0] = self.velocity_leader[0]
        self.vel_array[0, :] = self.velocity_leader

    def local_euler_step(self, position, velocity, next_velocity, acceleration, x_dot):
        """Advance one step using the local Euler update to be validated.

        Parameters
        ----------
        position : numpy.ndarray
            Vehicle positions at the current time step.
        velocity : numpy.ndarray
            Vehicle velocities at the current time step.
        next_velocity : numpy.ndarray
            Velocity array containing the imposed next leader velocity.
        acceleration : Callable
            IDM acceleration function.
        x_dot : Callable
            IDM position-rate function.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Updated position and velocity arrays.
        """
        position_change = x_dot(position, velocity) * self.dt
        velocity_change = acceleration(position, velocity) * self.dt

        velocity_change[0] = next_velocity[0] - velocity[0]
        new_position = position + np.maximum(position_change, 0)
        new_velocity = np.maximum(velocity + velocity_change, 0)

        return new_position, new_velocity

    def evolve(self, solution_name):
        """Run the selected Euler solution through the simulation-class flow.

        Parameters
        ----------
        solution_name : {"local_euler", "solver_euler"}
            Selects either the local Euler update or the solver-class Euler
            update from ``solvers.py``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Time, position, and velocity arrays from the selected solution.
        """
        position = self.pos_array.copy()
        velocity = self.vel_array.copy()

        acceleration = get_callable_method(
            self.equations,
            f"u_dot_{self.model}_open",
        )
        x_dot = get_callable_method(self.equations, f"x_dot_{self.model}_open")
        solver_method = self.get_solver_method()

        for step in range(len(self.tim_array) - 1):
            if solution_name == "local_euler":
                next_position, next_velocity = self.local_euler_step(
                    position[:, step],
                    velocity[:, step],
                    velocity[:, step + 1],
                    acceleration,
                    x_dot,
                )
            elif solution_name == "solver_euler":
                next_position, next_velocity = solver_method(
                    position[:, step],
                    velocity[:, step],
                    velocity[:, step + 1],
                    self.dt,
                    acceleration,
                    x_dot,
                )
            else:
                raise ValueError("solution_name must be local_euler or solver_euler")

            position[:, step + 1] = next_position
            velocity[1:, step + 1] = np.maximum(next_velocity[1:], 0)

        return self.tim_array, position, velocity

    def get_solver_method(self):
        """Select the Euler solver from solvers.py without hard-coding it.

        Parameters
        ----------
        None

        Returns
        -------
        Callable
            Solver method named from ``solver``, road type, and model order.
        """
        solver_class = solvers()
        model_order = self.model_order[self.model]
        solver_method_name = f"{self.solver}_open_{model_order}"

        return get_callable_method(solver_class, solver_method_name)


def build_output_paths(dt):
    """Create output paths tagged with the selected simulation time step.

    Parameters
    ----------
    dt : float
        Simulation time step in seconds.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Plot path and CSV path for the selected time step.
    """
    dt_label = f"{dt:g}".replace(".", "p")
    output_stem = f"idm_validation_sun_dt_{dt_label}"

    return OUTPUT_DIR / f"{output_stem}.png", OUTPUT_DIR / f"{output_stem}.csv"


def build_comparison_table(time_array, local_velocity, solver_velocity):
    """Create a time series of differences between both Euler solutions.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    local_velocity : numpy.ndarray
        Velocity array from the local Euler update.
    solver_velocity : numpy.ndarray
        Velocity array from the solver-class Euler update.

    Returns
    -------
    pandas.DataFrame
        Total, mean, and maximum absolute velocity differences over time.
    """
    absolute_difference = np.abs(local_velocity - solver_velocity)

    return pd.DataFrame(
        {
            "time_s": time_array,
            "total_abs_velocity_difference_mps": absolute_difference.sum(axis=0),
            "mean_abs_velocity_difference_mps": absolute_difference.mean(axis=0),
            "max_abs_velocity_difference_mps": absolute_difference.max(axis=0),
        }
    )


def plot_comparison(
    time_array,
    local_velocity,
    solver_velocity,
    comparison_table,
    output_file,
):
    """Plot local Euler results against solver-class Euler results.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    local_velocity : numpy.ndarray
        Velocity array from the local Euler update.
    solver_velocity : numpy.ndarray
        Velocity array from the solver-class Euler update.
    comparison_table : pandas.DataFrame
        Difference table created by ``build_comparison_table``.
    output_file : str or pathlib.Path
        Path where the comparison PNG is saved.

    Returns
    -------
    pathlib.Path
        Path to the saved plot.
    """
    vehicle_indices = [0, 5, 10, 19]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, dpi=300)
    fig.suptitle("IDM Euler Validation: Local Update vs solvers.py")

    for vehicle in vehicle_indices:
        axes[0].plot(
            time_array,
            local_velocity[vehicle],
            linestyle="-",
            label=f"Local Euler vehicle {vehicle}",
        )
        axes[0].plot(
            time_array,
            solver_velocity[vehicle],
            linestyle="--",
            label=f"solvers.py Euler vehicle {vehicle}",
        )

    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Velocity Traces from Both Euler Implementations")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["total_abs_velocity_difference_mps"],
        color="black",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Total |velocity difference| (m/s)")
    axes[1].set_title("Absolute Difference: Local Euler vs solvers.py Euler")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()

    return output_path


def parse_arguments():
    """Read command-line options for the Sun-style IDM validation.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the externally supplied time step.
    """
    parser = argparse.ArgumentParser(
        description="Compare local IDM Euler update with solvers.py Euler."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_TIME_STEP,
        help="Simulation time step in seconds. Default: 0.01",
    )

    args = parser.parse_args()
    if args.dt <= 0:
        raise ValueError("The time step --dt must be positive.")

    return args


def main():
    """Run the IDM Sun-style Euler validation and save the outputs.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Euler comparison table written to the time-step-tagged CSV file.
    """
    args = parse_arguments()
    plot_file, csv_file = build_output_paths(args.dt)
    idm_params, model_order = load_parameters()

    simulation = IDMSunValidationSimulation(
        idm_params,
        model_order,
        dt=args.dt,
    )
    time_array, _, local_velocity = simulation.evolve("local_euler")
    _, _, solver_velocity = simulation.evolve("solver_euler")

    comparison_table = build_comparison_table(
        time_array,
        local_velocity,
        solver_velocity,
    )
    comparison_table.to_csv(csv_file, index=False)
    plot_comparison(
        time_array,
        local_velocity,
        solver_velocity,
        comparison_table,
        output_file=plot_file,
    )

    max_difference = comparison_table["max_abs_velocity_difference_mps"].max()
    mean_difference = comparison_table["mean_abs_velocity_difference_mps"].mean()

    print("Running IDM Euler validation against solvers.py.")
    print("Reference solver is selected dynamically from model order.")
    print(f"Using externally supplied time step dt = {args.dt:g} s.\n")
    print("Final five time steps of the comparison:")
    print(comparison_table.tail().to_string(index=False))
    print("\nEuler comparison summary:")
    print(f"Maximum vehicle-level velocity difference: {max_difference:.6e} m/s")
    print(f"Mean vehicle-level velocity difference: {mean_difference:.6e} m/s")
    print(f"\nSaved validation plot to: {plot_file}")
    print(f"Saved comparison table to: {csv_file}")

    return comparison_table


if __name__ == "__main__":
    main()

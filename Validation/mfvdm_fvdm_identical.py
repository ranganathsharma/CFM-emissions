"""Validate MFVDM against FVDM when the first three MFVDM leaders match."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_DIR = Path(__file__).resolve().parent.parent
PARAMETER_FILE = REPO_DIR / "parameters.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent
TIME_STEP = 0.1
SIMULATION_DURATION = 150.0
MFVDM_LOOKAHEAD = 3
MFVDM_WEIGHT = np.array([0.7, 0.2, 0.1])
MFVDM_P = 0.6
MFVDM_Q = 0.4
PLOT_FILE = OUTPUT_DIR / "mfvdm_fvdm_identical_dt_0p1.png"
CSV_FILE = OUTPUT_DIR / "mfvdm_fvdm_identical_dt_0p1.csv"

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


def load_model_parameters(parameter_file=PARAMETER_FILE):
    """Load FVDM, MFVDM, and model-order data from parameters.yaml.

    Parameters
    ----------
    parameter_file : str or pathlib.Path
        Path to the repository ``parameters.yaml`` file.

    Returns
    -------
    tuple[dict, dict, dict]
        FVDM parameters, MFVDM parameters, and model-order dictionary.
    """
    with open(parameter_file, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    return (
        parameters["Models"]["fvdm"],
        parameters["Models"]["mfvdm"],
        parameters["Model_order"],
    )


def create_leader_velocity(time_array):
    """Create the speed profile used by FVDM and the first three MFVDM vehicles.

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
    velocity += np.where(time_array <= 30, time_array / 30 * 13, 0)
    velocity += np.where((time_array > 30) & (time_array <= 80), 13, 0)
    velocity += np.where(
        (time_array > 80) & (time_array <= 120),
        13 - (time_array - 80) / 40 * 7,
        0,
    )
    velocity += np.where(time_array > 120, 6, 0)

    return velocity


class Simulation:
    """Run FVDM or MFVDM using the same flow as Validation_mcfm.py.

    Parameters
    ----------
    model : {"fvdm", "mfvdm"}
        Model name to simulate.
    params : dict
        Model parameter dictionary.
    model_order : dict
        Mapping from model name to numerical order.

    Returns
    -------
    None
        Simulation arrays and model functions are stored on the object.
    """

    def __init__(self, model, params, model_order):
        """Configure the FVDM or MFVDM identical-profile simulation.

        Parameters
        ----------
        model : str
            Model name, either ``fvdm`` or ``mfvdm``.
        params : dict
            Parameter dictionary for the selected model.
        model_order : dict
            Mapping from model name to numerical order.

        Returns
        -------
        None
            The object is initialized with model parameters and simulation arrays.
        """
        self.model = model
        self.params_dict = params.copy()
        self.model_order = model_order
        self.dt = TIME_STEP
        self.duration = SIMULATION_DURATION
        self.solver = "euler"
        self.n = 2

        if self.model == "mfvdm":
            self.n = MFVDM_LOOKAHEAD + 1
            self.params_dict["m"] = MFVDM_LOOKAHEAD
            self.params_dict["weight"] = MFVDM_WEIGHT.copy()
            self.params_dict["p"] = MFVDM_P
            self.params_dict["q"] = MFVDM_Q

        self.equations = Model_equations(self.params_dict)
        self.unpack_params()
        self.initialise_velocity()
        self.init_spacing()
        self.setup_arrays()

    def unpack_params(self):
        """Copy parameter values onto the simulation object.

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
        """Create the imposed speed profile.

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
        """Calculate the initial equilibrium spacing.

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
        """Create and initialize simulation arrays.

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

        if self.model == "mfvdm":
            # MFVDM virtual leaders use identical speeds, but unlike MIDM their
            # virtual velocities are not extrapolated from relative velocity.
            self.vel_array[:MFVDM_LOOKAHEAD, :] = self.velocity_leader
        else:
            self.vel_array[0, :] = self.velocity_leader

    def evolve(self):
        """Run the simulation and return time, position, and velocity arrays.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Time, position, and velocity arrays.
        """
        solver_class = solvers()

        if self.model_order[self.model] == "second":
            acceleration = get_callable_method(
                self.equations,
                f"u_dot_{self.model}_open",
            )
            x_dot = get_callable_method(self.equations, f"x_dot_{self.model}_open")
            numerical_solver = get_callable_method(solver_class, "RK_open_second")
        else:
            acceleration = get_callable_method(
                self.equations,
                f"x_dot_{self.model}_open",
            )
            x_dot = get_callable_method(self.equations, f"x_dot_{self.model}_open")
            solver_name = f"{self.solver}_open_{self.model_order[self.model]}"
            numerical_solver = get_callable_method(solver_class, solver_name)

        for count, _ in enumerate(self.tim_array[:-2]):
            next_position, next_velocity = numerical_solver(
                self.pos_array[:, count],
                self.vel_array[:, count],
                self.vel_array[:, count + 1],
                self.dt,
                acceleration,
                x_dot,
            )

            if self.model == "mfvdm":
                self.pos_array[MFVDM_LOOKAHEAD - 1:, count + 1] = next_position[
                    MFVDM_LOOKAHEAD - 1:
                ]
                self.vel_array[MFVDM_LOOKAHEAD:, count + 1] = np.maximum(
                    next_velocity[MFVDM_LOOKAHEAD:],
                    0,
                )
                self.rebuild_identical_virtual_leaders(count + 1)
            else:
                self.pos_array[:, count + 1] = next_position
                self.vel_array[1:, count + 1] = np.maximum(next_velocity[1:], 0)

        return self.tim_array, self.pos_array, self.vel_array

    def rebuild_identical_virtual_leaders(self, step):
        """Rebuild MFVDM virtual leaders using the Validation_mcfm.py logic.

        Parameters
        ----------
        step : int
            Time-step index to update.

        Returns
        -------
        None
            Virtual leader rows are overwritten in the stored arrays.
        """
        leader = MFVDM_LOOKAHEAD - 1
        follower = MFVDM_LOOKAHEAD
        gap = self.pos_array[leader, step] - self.pos_array[follower, step]

        self.pos_array[1, step] = self.pos_array[leader, step] + gap
        self.pos_array[0, step] = self.pos_array[leader, step] + 2 * gap
        self.vel_array[:MFVDM_LOOKAHEAD, step] = self.vel_array[leader, step]


def build_comparison_table(time_array, fvdm_position, fvdm_velocity,
                           mfvdm_position, mfvdm_velocity):
    """Create residuals between FVDM vehicle 2 and MFVDM vehicle 4.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    fvdm_position : numpy.ndarray
        FVDM second-vehicle position, zero-based row 1.
    fvdm_velocity : numpy.ndarray
        FVDM second-vehicle velocity, zero-based row 1.
    mfvdm_position : numpy.ndarray
        MFVDM fourth-vehicle position, zero-based row 3.
    mfvdm_velocity : numpy.ndarray
        MFVDM fourth-vehicle velocity, zero-based row 3.

    Returns
    -------
    pandas.DataFrame
        Residual table for the requested vehicle comparison.
    """
    return pd.DataFrame(
        {
            "time_s": time_array,
            "fvdm_vehicle_2_position_m": fvdm_position,
            "mfvdm_vehicle_4_position_m": mfvdm_position,
            "position_residual_m": mfvdm_position - fvdm_position,
            "fvdm_vehicle_2_velocity_mps": fvdm_velocity,
            "mfvdm_vehicle_4_velocity_mps": mfvdm_velocity,
            "velocity_residual_mps": mfvdm_velocity - fvdm_velocity,
        }
    )


def plot_comparison(comparison_table):
    """Plot the requested FVDM/MFVDM vehicle comparison.

    Parameters
    ----------
    comparison_table : pandas.DataFrame
        Residual table created by ``build_comparison_table``.

    Returns
    -------
    pathlib.Path
        Path to the saved plot.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, dpi=300)
    fig.suptitle("MFVDM Identical-Speed Leaders Against FVDM")

    axes[0].plot(
        comparison_table["time_s"],
        comparison_table["fvdm_vehicle_2_velocity_mps"],
        label="FVDM vehicle 2",
    )
    axes[0].plot(
        comparison_table["time_s"],
        comparison_table["mfvdm_vehicle_4_velocity_mps"],
        linestyle="--",
        label="MFVDM vehicle 4",
    )
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Generated Follower Speed")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["velocity_residual_mps"],
        label="velocity residual",
    )
    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["position_residual_m"],
        label="position residual",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("MFVDM Vehicle 4 Minus FVDM Vehicle 2")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE, bbox_inches="tight", dpi=600)
    plt.close()

    return PLOT_FILE


def main():
    """Run the MFVDM/FVDM identical-speed-leader validation.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Residual table for MFVDM vehicle 4 and FVDM vehicle 2.
    """
    fvdm_params, mfvdm_params, model_order = load_model_parameters()

    fvdm_simulation = Simulation("fvdm", fvdm_params, model_order)
    mfvdm_simulation = Simulation("mfvdm", mfvdm_params, model_order)

    time_array, fvdm_position, fvdm_velocity = fvdm_simulation.evolve()
    _, mfvdm_position, mfvdm_velocity = mfvdm_simulation.evolve()

    comparison_table = build_comparison_table(
        time_array,
        fvdm_position[1],
        fvdm_velocity[1],
        mfvdm_position[3],
        mfvdm_velocity[3],
    )
    comparison_table.to_csv(CSV_FILE, index=False)
    plot_comparison(comparison_table)

    max_velocity_residual = comparison_table["velocity_residual_mps"].abs().max()
    max_position_residual = comparison_table["position_residual_m"].abs().max()

    print("Running MFVDM identical-speed-leader validation.")
    print("dt = 0.1 s")
    print("MFVDM vehicles 1, 2, and 3 have identical speed profiles.")
    print("MFVDM vehicle 4 is compared with FVDM vehicle 2.\n")
    print("Final five residual rows:")
    print(comparison_table.tail().to_string(index=False))
    print("\nMFVDM/FVDM residual summary:")
    print(f"Maximum velocity residual: {max_velocity_residual:.6e} m/s")
    print(f"Maximum position residual: {max_position_residual:.6e} m")
    print(f"\nSaved validation plot to: {PLOT_FILE}")
    print(f"Saved residual table to: {CSV_FILE}")

    return comparison_table


if __name__ == "__main__":
    main()

"""Run a two-step MFVDM validation using hard-coded boundary trajectories."""

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
N_FOLLOWERS_FIRST_RUN = 100
N_FOLLOWERS_SECOND_RUN = 50
CONTROLLED_PHYSICAL_VEHICLE = 52
BOUNDARY_PHYSICAL_VEHICLES = (50, 51, 52)
FIRST_RUN_TARGET_PHYSICAL_VEHICLE = 100
SECOND_RUN_TARGET_ROW = 50
PLOT_FILE = OUTPUT_DIR / "mfvdm_two_step_validation_dt_0p1.png"
CSV_FILE = OUTPUT_DIR / "mfvdm_two_step_validation_dt_0p1.csv"

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


def load_mfvdm_parameters(parameter_file=PARAMETER_FILE):
    """Load MFVDM parameters and model-order metadata.

    Parameters
    ----------
    parameter_file : str or pathlib.Path
        Path to the repository ``parameters.yaml`` file.

    Returns
    -------
    tuple[dict, dict]
        MFVDM parameter dictionary and model-order dictionary.
    """
    with open(parameter_file, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    return parameters["Models"]["mfvdm"], parameters["Model_order"]


def create_leader_velocity(time_array):
    """Create the imposed leader speed profile for the first run.

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
    velocity += np.where(time_array <= 30, time_array / 30 * 15, 0)
    velocity += np.where((time_array > 30) & (time_array <= 80), 15, 0)
    velocity += np.where(
        (time_array > 80) & (time_array <= 120),
        15 - (time_array - 80) / 40 * 8,
        0,
    )
    velocity += np.where(time_array > 120, 7, 0)

    return velocity


def physical_vehicle_to_first_run_row(physical_vehicle):
    """Map physical vehicle number to first-run MFVDM row.

    Parameters
    ----------
    physical_vehicle : int
        Physical vehicle number, with 0 as the controlled leader.

    Returns
    -------
    int
        Row index in the MFVDM array, including two virtual leader rows.
    """
    return physical_vehicle + MFVDM_LOOKAHEAD - 1


class TwoStepMFVDMSimulation:
    """MFVDM simulation copied in spirit from Validation_mcfm.py.

    Parameters
    ----------
    params : dict
        MFVDM parameter dictionary.
    model_order : dict
        Mapping from model name to numerical order.
    n_followers : int
        Number of generated followers after the controlled leader.
    controlled_position : numpy.ndarray or None
        Optional hard-coded positions for boundary rows 0, 1, and 2.
    controlled_velocity : numpy.ndarray or None
        Optional hard-coded velocities for boundary rows 0, 1, and 2.
    initial_position : numpy.ndarray or None
        Optional initial positions for every second-run row.
    initial_velocity : numpy.ndarray or None
        Optional initial velocities for every second-run row.

    Returns
    -------
    None
        The simulation arrays and model functions are stored on the object.
    """

    def __init__(
        self,
        params,
        model_order,
        n_followers,
        controlled_position=None,
        controlled_velocity=None,
        initial_position=None,
        initial_velocity=None,
    ):
        """Configure one MFVDM run used by the two-step validation.

        Parameters
        ----------
        params : dict
            MFVDM parameter dictionary loaded from ``parameters.yaml``.
        model_order : dict
            Mapping from model name to numerical order.
        n_followers : int
            Number of generated followers after the controlled boundary.
        controlled_position : numpy.ndarray or None
            Optional full position history for hard-coded boundary rows.
        controlled_velocity : numpy.ndarray or None
            Optional full velocity history for hard-coded boundary rows.
        initial_position : numpy.ndarray or None
            Optional initial positions copied from the first run.
        initial_velocity : numpy.ndarray or None
            Optional initial velocities copied from the first run.

        Returns
        -------
        None
            The object is initialized with model parameters and simulation arrays.
        """
        self.params_dict = params.copy()
        self.model_order = model_order
        self.dt = TIME_STEP
        self.duration = SIMULATION_DURATION
        self.n = MFVDM_LOOKAHEAD + n_followers
        self.controlled_position = controlled_position
        self.controlled_velocity = controlled_velocity
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity

        self.params_dict["m"] = MFVDM_LOOKAHEAD
        self.params_dict["weight"] = np.array([0.7, 0.2, 0.1])
        self.params_dict["p"] = 0.6
        self.params_dict["q"] = 0.4
        self.equations = Model_equations(self.params_dict)

        self.initialise_velocity()
        self.init_spacing()
        self.setup_arrays()

    def initialise_velocity(self):
        """Create the leader velocity profile.

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
        """Calculate initial MFVDM equilibrium spacing.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Initial equilibrium spacing is stored on the object.
        """
        init_spacing = get_callable_method(self.equations, "init_spacing_mfvdm")
        self.init_spacing_value = init_spacing(self.velocity_leader[0])

    def setup_arrays(self):
        """Create and initialize the simulation arrays.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Position and velocity arrays are stored on the object.
        """
        n_steps = self.tim_array.shape[0]
        self.pos_array = np.zeros((self.n, n_steps))
        self.vel_array = np.zeros((self.n, n_steps))
        self.pos_array[:, 0] = -np.arange(self.n) * self.init_spacing_value
        self.vel_array[:, 0] = self.velocity_leader[0]

        if self.initial_position is not None:
            n_initial = len(self.initial_position)
            self.pos_array[:n_initial, 0] = self.initial_position
        if self.initial_velocity is not None:
            n_initial = len(self.initial_velocity)
            self.vel_array[:n_initial, 0] = self.initial_velocity

        if self.controlled_position is None or self.controlled_velocity is None:
            self.vel_array[MFVDM_LOOKAHEAD - 1] = self.velocity_leader
        else:
            n_boundary = self.controlled_position.shape[0]
            self.pos_array[:n_boundary] = self.controlled_position
            self.vel_array[:n_boundary] = self.controlled_velocity

    def evolve(self):
        """Run the MFVDM simulation using the Validation_mcfm.py flow.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Time, position, and velocity arrays.
        """
        solver_class = solvers()
        acceleration = get_callable_method(self.equations, "u_dot_mfvdm_open")
        x_dot = get_callable_method(self.equations, "x_dot_mfvdm_open")
        numerical_solver = get_callable_method(solver_class, "RK_open_second")

        if self.controlled_position is None:
            self.rebuild_virtual_leaders(1)
        else:
            self.apply_hard_coded_boundary(0)

        for count, _ in enumerate(self.tim_array[:-2]):
            next_position, next_velocity = numerical_solver(
                self.pos_array[:, count],
                self.vel_array[:, count],
                self.vel_array[:, count + 1],
                self.dt,
                acceleration,
                x_dot,
            )

            self.pos_array[MFVDM_LOOKAHEAD - 1:, count + 1] = next_position[
                MFVDM_LOOKAHEAD - 1:
            ]
            self.vel_array[MFVDM_LOOKAHEAD:, count + 1] = np.maximum(
                next_velocity[MFVDM_LOOKAHEAD:],
                0,
            )

            if self.controlled_position is None:
                self.rebuild_virtual_leaders(count + 1)
            else:
                self.apply_hard_coded_boundary(count + 1)

        return self.tim_array, self.pos_array, self.vel_array

    def rebuild_virtual_leaders(self, step):
        """Rebuild virtual leaders using the MFVDM logic in Validation_mcfm.py.

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

        self.pos_array[0, step] = self.pos_array[leader, step] + 2 * gap
        self.pos_array[1, step] = self.pos_array[leader, step] + gap
        self.vel_array[0, step] = self.vel_array[leader, step]
        self.vel_array[1, step] = self.vel_array[leader, step]

    def apply_hard_coded_boundary(self, step):
        """Restore hard-coded boundary rows for the second run.

        Parameters
        ----------
        step : int
            Time-step index to update.

        Returns
        -------
        None
            Rows 0, 1, and 2 are overwritten with first-run trajectories.
        """
        self.pos_array[:MFVDM_LOOKAHEAD, step] = self.controlled_position[
            :MFVDM_LOOKAHEAD,
            step,
        ]
        self.vel_array[:MFVDM_LOOKAHEAD, step] = self.controlled_velocity[
            :MFVDM_LOOKAHEAD,
            step,
        ]


def extract_boundary_profiles(first_position, first_velocity):
    """Extract vehicles 50, 51, and 52 as hard-coded second-run rows.

    Parameters
    ----------
    first_position : numpy.ndarray
        First-run MFVDM position array.
    first_velocity : numpy.ndarray
        First-run MFVDM velocity array.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Position and velocity arrays for second-run rows 0, 1, and 2.
    """
    rows = [
        physical_vehicle_to_first_run_row(vehicle)
        for vehicle in BOUNDARY_PHYSICAL_VEHICLES
    ]

    return first_position[rows], first_velocity[rows]


def extract_second_run_initial_state(first_position, first_velocity):
    """Copy first-run physical vehicles 50 through 100 into the second run.

    Parameters
    ----------
    first_position : numpy.ndarray
        First-run MFVDM position array.
    first_velocity : numpy.ndarray
        First-run MFVDM velocity array.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Initial position and velocity arrays for second-run rows 0 through 50.
    """
    physical_vehicles = range(
        CONTROLLED_PHYSICAL_VEHICLE - (MFVDM_LOOKAHEAD - 1),
        FIRST_RUN_TARGET_PHYSICAL_VEHICLE + 1,
    )
    rows = [
        physical_vehicle_to_first_run_row(vehicle)
        for vehicle in physical_vehicles
    ]

    return first_position[rows, 0], first_velocity[rows, 0]


def build_comparison_table(time_array, first_velocity, second_velocity):
    """Create residuals for the mapped first-run and second-run vehicles.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    first_velocity : numpy.ndarray
        First-run target vehicle speed profile.
    second_velocity : numpy.ndarray
        Second-run mapped vehicle speed profile.

    Returns
    -------
    pandas.DataFrame
        Velocity residual table for the two-step verification.
    """
    return pd.DataFrame(
        {
            "time_s": time_array,
            "first_run_physical_100_velocity_mps": first_velocity,
            "second_run_row_50_velocity_mps": second_velocity,
            "velocity_residual_mps": second_velocity - first_velocity,
        }
    )


def calculate_normalized_residual(comparison_table):
    """Calculate the residual norm relative to the first-run speed norm.

    Parameters
    ----------
    comparison_table : pandas.DataFrame
        Table containing the first-run speed and residual columns.

    Returns
    -------
    float
        L2 norm of the residual divided by the L2 norm of the first-run speed.
    """
    residual = comparison_table["velocity_residual_mps"].to_numpy()
    first_speed = comparison_table[
        "first_run_physical_100_velocity_mps"
    ].to_numpy()
    first_speed_norm = np.linalg.norm(first_speed)

    if first_speed_norm == 0:
        return np.nan

    return np.linalg.norm(residual) / first_speed_norm


def plot_comparison(comparison_table):
    """Plot the target speed profiles and residual.

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
    fig.suptitle("MFVDM Two-Step Virtual-Leader Verification")

    axes[0].plot(
        comparison_table["time_s"],
        comparison_table["first_run_physical_100_velocity_mps"],
        label="first run physical vehicle 100",
    )
    axes[0].plot(
        comparison_table["time_s"],
        comparison_table["second_run_row_50_velocity_mps"],
        linestyle="--",
        label="second run row 50",
    )
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Mapped Speed Profiles")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["velocity_residual_mps"],
        color="black",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Velocity residual (m/s)")
    axes[1].set_title("Second Run Row 50 Minus First Run Physical Vehicle 100")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE, bbox_inches="tight", dpi=600)
    plt.close()

    return PLOT_FILE


def main():
    """Run the two-step MFVDM virtual-leader verification.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Velocity residual table for the mapped vehicle profiles.
    """
    mfvdm_params, model_order = load_mfvdm_parameters()

    first_run = TwoStepMFVDMSimulation(
        mfvdm_params,
        model_order,
        n_followers=N_FOLLOWERS_FIRST_RUN,
    )
    time_array, first_position, first_velocity = first_run.evolve()

    controlled_position, controlled_velocity = extract_boundary_profiles(
        first_position,
        first_velocity,
    )
    initial_position, initial_velocity = extract_second_run_initial_state(
        first_position,
        first_velocity,
    )
    second_run = TwoStepMFVDMSimulation(
        mfvdm_params,
        model_order,
        n_followers=N_FOLLOWERS_SECOND_RUN,
        controlled_position=controlled_position,
        controlled_velocity=controlled_velocity,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    _, _, second_velocity = second_run.evolve()

    first_target_row = physical_vehicle_to_first_run_row(
        FIRST_RUN_TARGET_PHYSICAL_VEHICLE
    )
    comparison_table = build_comparison_table(
        time_array,
        first_velocity[first_target_row],
        second_velocity[SECOND_RUN_TARGET_ROW],
    )
    comparison_table.to_csv(CSV_FILE, index=False)
    plot_comparison(comparison_table)

    max_velocity_residual = comparison_table["velocity_residual_mps"].abs().max()
    normalized_residual = calculate_normalized_residual(comparison_table)

    print("Running MFVDM two-step virtual-leader verification.")
    print("dt = 0.1 s")
    print("First run: generated physical vehicles 0 through 100.")
    print(
        "Second run: first-run physical vehicles 50, 51, and 52 are "
        "hard-coded as rows 0, 1, and 2."
    )
    print("Second-run initial rows 1-50 match first-run physical vehicles 51-100.")
    print("Mapping check: first-run physical 100 maps to second-run row 50.\n")
    print("Final five residual rows:")
    print(comparison_table.tail().to_string(index=False))
    print("\nTwo-step residual summary:")
    print(f"Maximum velocity residual: {max_velocity_residual:.6e} m/s")
    print(
        "Normalized residual: "
        f"{normalized_residual:.6e} "
        "(||residual|| / ||first-run speed||)"
    )
    print(f"\nSaved validation plot to: {PLOT_FILE}")
    print(f"Saved residual table to: {CSV_FILE}")

    return comparison_table


if __name__ == "__main__":
    main()

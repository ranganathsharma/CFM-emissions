"""Validate that MFVDM with weights [1, 0, 0] collapses to FVDM."""

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
N_FVDM_VEHICLES = 20
MFVDM_LOOKAHEAD = 3
COLLAPSE_WEIGHTS = np.array([1.0, 0.0, 0.0])
PLOT_FILE = OUTPUT_DIR / "mfvdm_factor_validation_dt_0p1.png"
CSV_FILE = OUTPUT_DIR / "mfvdm_factor_validation_dt_0p1.csv"

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
    """Create the imposed leader velocity used by FVDM and MFVDM.

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


class FactorValidationSimulation:
    """Run FVDM or MFVDM through the same Euler simulation flow.

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
        """Configure an FVDM or MFVDM collapse-validation simulation.

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
        self.n = N_FVDM_VEHICLES
        self.solver = "euler"

        if self.model == "mfvdm":
            self.n += MFVDM_LOOKAHEAD - 1
            self.params_dict["m"] = MFVDM_LOOKAHEAD
            self.params_dict["weight"] = COLLAPSE_WEIGHTS.copy()
            self.params_dict["p"] = 1.0
            self.params_dict["q"] = 0.0

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
        """Create the common imposed leader velocity profile.

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
        """Calculate initial equilibrium spacing for the selected model.

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

        if self.model == "mfvdm":
            leader_index = MFVDM_LOOKAHEAD - 1
            index_offsets = leader_index - np.arange(self.n)
            self.pos_array[:, 0] = index_offsets * self.init_spacing_value
            self.vel_array[:MFVDM_LOOKAHEAD, :] = self.velocity_leader
        else:
            self.pos_array[:, 0] = -np.arange(self.n) * self.init_spacing_value
            self.vel_array[0, :] = self.velocity_leader

        self.vel_array[:, 0] = self.velocity_leader[0]

    def evolve(self):
        """Run the selected model using the Euler method from solvers.py.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Time, position, and velocity arrays.
        """
        solver_class = solvers()
        model_order = self.model_order[self.model]
        solver_name = f"{self.solver}_open_{model_order}"
        numerical_solver = get_callable_method(solver_class, solver_name)
        acceleration = get_callable_method(
            self.equations,
            f"u_dot_{self.model}_open",
        )
        x_dot = get_callable_method(self.equations, f"x_dot_{self.model}_open")

        for step in range(len(self.tim_array) - 1):
            next_position, next_velocity = numerical_solver(
                self.pos_array[:, step],
                self.vel_array[:, step],
                self.vel_array[:, step + 1],
                self.dt,
                acceleration,
                x_dot,
            )

            self.pos_array[:, step + 1] = next_position
            self.vel_array[:, step + 1] = np.maximum(next_velocity, 0)
            self.apply_controlled_leaders(step)

        return self.tim_array, self.pos_array, self.vel_array

    def apply_controlled_leaders(self, step):
        """Keep the imposed leader and MFVDM virtual leaders in the state.

        Parameters
        ----------
        step : int
            Current time-step index whose next state has just been calculated.

        Returns
        -------
        None
            Leader rows are overwritten in the stored arrays.
        """
        next_step = step + 1
        if self.model == "fvdm":
            self.vel_array[0, next_step] = self.velocity_leader[next_step]
            return

        leader_index = MFVDM_LOOKAHEAD - 1
        spacing = self.init_spacing_value
        self.vel_array[:MFVDM_LOOKAHEAD, next_step] = self.velocity_leader[next_step]
        self.pos_array[leader_index, next_step] = self.pos_array[
            leader_index,
            step,
        ] + self.velocity_leader[step] * self.dt

        for index in range(leader_index):
            offset = leader_index - index
            self.pos_array[index, next_step] = (
                self.pos_array[leader_index, next_step] + offset * spacing
            )


def align_mfvdm_with_fvdm(mfvdm_position, mfvdm_velocity):
    """Align MFVDM physical vehicles with the equivalent FVDM vehicles.

    Parameters
    ----------
    mfvdm_position : numpy.ndarray
        MFVDM position array, including virtual leaders.
    mfvdm_velocity : numpy.ndarray
        MFVDM velocity array, including virtual leaders.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Position and velocity arrays corresponding to FVDM vehicle numbering.
    """
    leader_index = MFVDM_LOOKAHEAD - 1
    return mfvdm_position[leader_index:], mfvdm_velocity[leader_index:]


def build_comparison_table(
    time_array,
    fvdm_position,
    fvdm_velocity,
    mfvdm_position,
    mfvdm_velocity,
):
    """Create residuals between FVDM and MFVDM after vehicle alignment.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    fvdm_position : numpy.ndarray
        FVDM position array.
    fvdm_velocity : numpy.ndarray
        FVDM velocity array.
    mfvdm_position : numpy.ndarray
        Aligned MFVDM position array.
    mfvdm_velocity : numpy.ndarray
        Aligned MFVDM velocity array.

    Returns
    -------
    pandas.DataFrame
        Position and velocity residual summaries over time.
    """
    velocity_residual = mfvdm_velocity - fvdm_velocity
    position_residual = mfvdm_position - fvdm_position

    return pd.DataFrame(
        {
            "time_s": time_array,
            "max_abs_velocity_residual_mps": np.abs(velocity_residual).max(axis=0),
            "mean_abs_velocity_residual_mps": np.abs(velocity_residual).mean(axis=0),
            "max_abs_position_residual_m": np.abs(position_residual).max(axis=0),
            "mean_abs_position_residual_m": np.abs(position_residual).mean(axis=0),
        }
    )


def save_vehicle_residuals(time_array, fvdm_velocity, mfvdm_velocity, output_file):
    """Save per-vehicle velocity residuals in long format.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    fvdm_velocity : numpy.ndarray
        FVDM velocity array.
    mfvdm_velocity : numpy.ndarray
        Aligned MFVDM velocity array.
    output_file : str or pathlib.Path
        CSV path where residuals are saved.

    Returns
    -------
    pandas.DataFrame
        Long-format table of velocity residuals by time and vehicle.
    """
    rows = []
    residual = mfvdm_velocity - fvdm_velocity
    for vehicle in range(residual.shape[0]):
        rows.extend(
            {
                "time_s": time,
                "vehicle": vehicle,
                "fvdm_velocity_mps": fvdm_velocity[vehicle, time_id],
                "mfvdm_velocity_mps": mfvdm_velocity[vehicle, time_id],
                "velocity_residual_mps": residual[vehicle, time_id],
            }
            for time_id, time in enumerate(time_array)
        )

    residual_table = pd.DataFrame(rows)
    residual_table.to_csv(output_file, index=False)
    return residual_table


def plot_comparison(
    time_array,
    fvdm_velocity,
    mfvdm_velocity,
    comparison_table,
    output_file,
):
    """Plot FVDM/MFVDM velocity traces and residuals.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    fvdm_velocity : numpy.ndarray
        FVDM velocity array.
    mfvdm_velocity : numpy.ndarray
        Aligned MFVDM velocity array.
    comparison_table : pandas.DataFrame
        Residual summary table created by ``build_comparison_table``.
    output_file : str or pathlib.Path
        Path where the comparison PNG is saved.

    Returns
    -------
    pathlib.Path
        Path to the saved plot.
    """
    vehicle_indices = [0, 5, 10, 19]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, dpi=300)
    fig.suptitle("MFVDM [1, 0, 0] Factor Validation Against FVDM")

    for vehicle in vehicle_indices:
        axes[0].plot(
            time_array,
            fvdm_velocity[vehicle],
            linestyle="-",
            label=f"FVDM vehicle {vehicle}",
        )
        axes[0].plot(
            time_array,
            mfvdm_velocity[vehicle],
            linestyle="--",
            label=f"MFVDM vehicle {vehicle}",
        )

    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Aligned FVDM and MFVDM Vehicle Velocities")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["max_abs_velocity_residual_mps"],
        label="max velocity residual",
    )
    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["max_abs_position_residual_m"],
        label="max position residual",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual After Accounting for Virtual Leaders")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()

    return output_path


def main():
    """Run the MFVDM factor-collapse validation and save the outputs.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Residual summary table for the FVDM/MFVDM comparison.
    """
    fvdm_params, mfvdm_params, model_order = load_model_parameters()

    fvdm_simulation = FactorValidationSimulation("fvdm", fvdm_params, model_order)
    mfvdm_simulation = FactorValidationSimulation("mfvdm", mfvdm_params, model_order)

    time_array, fvdm_position, fvdm_velocity = fvdm_simulation.evolve()
    _, mfvdm_position, mfvdm_velocity = mfvdm_simulation.evolve()
    aligned_mfvdm_position, aligned_mfvdm_velocity = align_mfvdm_with_fvdm(
        mfvdm_position,
        mfvdm_velocity,
    )

    comparison_table = build_comparison_table(
        time_array,
        fvdm_position,
        fvdm_velocity,
        aligned_mfvdm_position,
        aligned_mfvdm_velocity,
    )
    comparison_table.to_csv(CSV_FILE, index=False)
    save_vehicle_residuals(
        time_array,
        fvdm_velocity,
        aligned_mfvdm_velocity,
        CSV_FILE.with_name("mfvdm_factor_validation_vehicle_residuals_dt_0p1.csv"),
    )
    plot_comparison(
        time_array,
        fvdm_velocity,
        aligned_mfvdm_velocity,
        comparison_table,
        PLOT_FILE,
    )

    max_velocity_residual = comparison_table["max_abs_velocity_residual_mps"].max()
    max_position_residual = comparison_table["max_abs_position_residual_m"].max()

    print("Running self-contained MFVDM factor validation.")
    print("dt = 0.1 s")
    print("MFVDM weight is set to [1, 0, 0], with p = 1 and q = 0.")
    print("MFVDM virtual leaders remain in the simulation and are aligned out only")
    print("when comparing the physical FVDM-equivalent vehicles.\n")
    print("Final five residual rows:")
    print(comparison_table.tail().to_string(index=False))
    print("\nMFVDM collapse residual summary:")
    print(f"Maximum velocity residual: {max_velocity_residual:.6e} m/s")
    print(f"Maximum position residual: {max_position_residual:.6e} m")
    print(f"\nSaved validation plot to: {PLOT_FILE}")
    print(f"Saved residual summary to: {CSV_FILE}")

    return comparison_table


if __name__ == "__main__":
    main()

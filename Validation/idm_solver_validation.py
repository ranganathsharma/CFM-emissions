"""Compare IDM trajectories produced by Euler and Runge-Kutta solvers."""

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
PLOT_FILE = OUTPUT_DIR / "idm_solver_comparison.png"
CSV_FILE = OUTPUT_DIR / "idm_solver_comparison.csv"
DEFAULT_TIME_STEP = 0.01

# Allow this validation file to be run directly from inside the Validation folder.
sys.path.insert(0, str(REPO_DIR))

from model_equations import Model_equations  # noqa: E402
from solvers import solvers  # noqa: E402


def load_idm_parameters(parameter_file=PARAMETER_FILE):
    """Load the IDM parameters used by the simulation code.

    Parameters
    ----------
    parameter_file : str or pathlib.Path
        Path to the repository ``parameters.yaml`` file.

    Returns
    -------
    dict
        IDM parameter dictionary from the ``Models`` section.
    """
    with open(parameter_file, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    return parameters["Models"]["idm"]


def create_leader_velocity(time_array):
    """Create the imposed leader velocity used by both numerical solvers.

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

    # Smoothly accelerate, cruise, then decelerate so both solvers see
    # meaningful transient behaviour.
    velocity += np.where(time_array <= 30, time_array / 30 * 15, 0)
    velocity += np.where((time_array > 30) & (time_array <= 80), 15, 0)
    velocity += np.where(
        (time_array > 80) & (time_array <= 120),
        15 - (time_array - 80) / 40 * 8,
        0,
    )
    velocity += np.where(time_array > 120, 7, 0)

    return velocity


def initialise_traffic_state(idm_model, leader_velocity, n_vehicles, n_steps):
    """Initialise vehicle positions and velocities at equilibrium spacing.

    Parameters
    ----------
    idm_model : model_equations.Model_equations
        IDM equation object used to calculate the initial equilibrium spacing.
    leader_velocity : numpy.ndarray
        Imposed leader velocity profile in m/s.
    n_vehicles : int
        Number of vehicles, including the leader.
    n_steps : int
        Number of time steps in the simulation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Position and velocity arrays with shape ``(n_vehicles, n_steps)``.
    """
    initial_velocity = leader_velocity[0]
    initial_spacing = idm_model.init_spacing_idm(initial_velocity)

    position = np.zeros((n_vehicles, n_steps))
    velocity = np.zeros((n_vehicles, n_steps))

    position[:, 0] = -np.arange(n_vehicles) * initial_spacing
    velocity[:, 0] = initial_velocity
    velocity[0, :] = leader_velocity

    return position, velocity


def run_idm_simulation(
    solver_name,
    dt=DEFAULT_TIME_STEP,
    duration=150.0,
    n_vehicles=20,
):
    """Run one IDM simulation with the selected numerical solver.

    Parameters
    ----------
    solver_name : {"euler_open_second", "RK_open_second"}
        Numerical solver method to use from ``solvers.py``.
    dt : float
        Simulation time step in seconds.
    duration : float
        Simulation duration in seconds.
    n_vehicles : int
        Number of vehicles, including the leader.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Time, position, and velocity arrays from the simulation.
    """
    time_array = np.arange(0, duration + dt, dt)
    leader_velocity = create_leader_velocity(time_array)

    idm_model = Model_equations(load_idm_parameters())
    position, velocity = initialise_traffic_state(
        idm_model,
        leader_velocity,
        n_vehicles,
        len(time_array),
    )

    solver_method = getattr(solvers(), solver_name)
    acceleration = idm_model.u_dot_idm_open
    position_rate = idm_model.x_dot_idm_open

    for step in range(len(time_array) - 1):
        next_position, next_velocity = solver_method(
            position[:, step],
            velocity[:, step],
            velocity[:, step + 1],
            dt,
            acceleration,
            position_rate,
        )

        position[:, step + 1] = next_position
        velocity[1:, step + 1] = np.maximum(next_velocity[1:], 0)

    return time_array, position, velocity


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
    output_stem = f"idm_solver_comparison_dt_{dt_label}"

    return OUTPUT_DIR / f"{output_stem}.png", OUTPUT_DIR / f"{output_stem}.csv"


def build_comparison_table(time_array, euler_velocity, rk_velocity):
    """Create a compact table of solver differences over time.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    euler_velocity : numpy.ndarray
        Velocity array from the Euler simulation.
    rk_velocity : numpy.ndarray
        Velocity array from the Runge-Kutta simulation.

    Returns
    -------
    pandas.DataFrame
        Time series of total, mean, and maximum absolute velocity differences.
    """
    absolute_difference = np.abs(euler_velocity - rk_velocity)

    return pd.DataFrame(
        {
            "time_s": time_array,
            "total_abs_velocity_difference_mps": absolute_difference.sum(axis=0),
            "mean_abs_velocity_difference_mps": absolute_difference.mean(axis=0),
            "max_abs_velocity_difference_mps": absolute_difference.max(axis=0),
        }
    )


def plot_solver_comparison(
    time_array,
    euler_velocity,
    rk_velocity,
    comparison_table,
    output_file=PLOT_FILE,
):
    """Plot IDM velocity traces and solver differences.

    Parameters
    ----------
    time_array : numpy.ndarray
        Simulation time stamps in seconds.
    euler_velocity : numpy.ndarray
        Velocity array from the Euler simulation.
    rk_velocity : numpy.ndarray
        Velocity array from the Runge-Kutta simulation.
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
    fig.suptitle("IDM Numerical Solver Comparison: Euler vs Runge-Kutta")

    for vehicle in vehicle_indices:
        axes[0].plot(
            time_array,
            euler_velocity[vehicle],
            linestyle="-",
            label=f"Euler vehicle {vehicle}",
        )
        axes[0].plot(
            time_array,
            rk_velocity[vehicle],
            linestyle="--",
            label=f"RK vehicle {vehicle}",
        )

    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title("Velocity Traces from Both Numerical Methods")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(
        comparison_table["time_s"],
        comparison_table["total_abs_velocity_difference_mps"],
        color="black",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Total |velocity difference| (m/s)")
    axes[1].set_title("Euler vs Runge-Kutta Absolute Velocity Difference")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()

    return output_path


def parse_arguments():
    """Read command-line options for the solver validation.

    Parameters
    ----------
    None

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the externally supplied time step.
    """
    parser = argparse.ArgumentParser(
        description="Compare IDM Euler and Runge-Kutta solvers."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_TIME_STEP,
        help="Simulation time step in seconds. Default: 0.1",
    )

    args = parser.parse_args()
    if args.dt <= 0:
        raise ValueError("The time step --dt must be positive.")

    return args


def main():
    """Run the IDM solver comparison and save the plot and CSV table.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Solver-difference table written to ``idm_solver_comparison.csv``.
    """
    args = parse_arguments()
    plot_file, csv_file = build_output_paths(args.dt)

    print("Running IDM numerical-solver validation.")
    print("Comparing euler_open_second against RK_open_second.\n")
    print(f"Using externally supplied time step dt = {args.dt:g} s.\n")

    time_array, _, euler_velocity = run_idm_simulation(
        "euler_open_second",
        dt=args.dt,
    )
    _, _, rk_velocity = run_idm_simulation("RK_open_second", dt=args.dt)

    comparison_table = build_comparison_table(
        time_array,
        euler_velocity,
        rk_velocity,
    )
    comparison_table.to_csv(csv_file, index=False)
    plot_solver_comparison(
        time_array,
        euler_velocity,
        rk_velocity,
        comparison_table,
        output_file=plot_file,
    )

    max_difference = comparison_table["max_abs_velocity_difference_mps"].max()
    mean_difference = comparison_table["mean_abs_velocity_difference_mps"].mean()

    print("Final five time steps of the Euler vs Runge-Kutta comparison:")
    print(comparison_table.tail().to_string(index=False))
    print("\nSolver comparison summary:")
    print(f"Maximum vehicle-level velocity difference: {max_difference:.6e} m/s")
    print(f"Mean vehicle-level velocity difference: {mean_difference:.6e} m/s")
    print(f"\nSaved Euler vs Runge-Kutta plot to: {plot_file}")
    print(f"Saved comparison table to: {csv_file}")

    return comparison_table


if __name__ == "__main__":
    main()

"""Validate IDM equilibrium spacing against the project implementation."""

from pathlib import Path
import sys

import pandas as pd
import yaml


REPO_DIR = Path(__file__).resolve().parent.parent
PARAMETER_FILE = REPO_DIR / "parameters.yaml"
OUTPUT_FILE = Path(__file__).resolve().parent / "idm_equilibrium_spacing_validation.csv"

# Allow this validation file to be run directly from inside the Validation folder.
sys.path.insert(0, str(REPO_DIR))

from model_equations import Model_equations  # noqa: E402


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


def manual_idm_equilibrium_spacing(velocity, idm_parameters):
    """Calculate IDM equilibrium spacing from the closed-form equation.

    Parameters
    ----------
    velocity : float
        Equilibrium velocity in m/s. The value must be lower than the IDM
        desired velocity.
    idm_parameters : dict
        IDM parameters containing ``jam_dis_0``, ``safe_head``, ``des_vel``,
        ``acc_exp``, and ``len_veh``.

    Returns
    -------
    float
        Equilibrium front-to-front spacing in metres, matching the convention
        used by ``Model_equations.init_spacing_idm``.
    """
    jam_distance = idm_parameters["jam_dis_0"]
    safe_headway = idm_parameters["safe_head"]
    desired_velocity = idm_parameters["des_vel"]
    acceleration_exponent = idm_parameters["acc_exp"]
    vehicle_length = idm_parameters["len_veh"]

    free_flow_term = 1 - (velocity / desired_velocity) ** acceleration_exponent
    net_spacing = (jam_distance + safe_headway * velocity) / free_flow_term**0.5

    # The model code returns front-to-front spacing, so vehicle length is added.
    return net_spacing + vehicle_length


def compare_manual_and_code_spacing(velocities=None):
    """Compare manual IDM spacings with values returned by the code.

    Parameters
    ----------
    velocities : list[float] or None
        Equilibrium velocities in m/s. When omitted, a small set of manually
        chosen low, medium, and high speeds is used.

    Returns
    -------
    pandas.DataFrame
        Table with manual spacing, code spacing, spacing difference, and the
        velocity recovered by ``Model_equations.init_vel_idm``.
    """
    if velocities is None:
        velocities = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    idm_parameters = load_idm_parameters()
    idm_model = Model_equations(idm_parameters)
    rows = []

    for velocity in velocities:
        if velocity >= idm_parameters["des_vel"]:
            raise ValueError("IDM equilibrium spacing is finite only below des_vel.")

        manual_spacing = manual_idm_equilibrium_spacing(velocity, idm_parameters)
        code_spacing = idm_model.init_spacing_idm(velocity)
        recovered_velocity = idm_model.init_vel_idm(manual_spacing)

        rows.append(
            {
                "velocity_mps": velocity,
                "manual_spacing_m": manual_spacing,
                "code_spacing_m": code_spacing,
                "spacing_difference_m": code_spacing - manual_spacing,
                "recovered_velocity_mps": recovered_velocity,
                "velocity_difference_mps": recovered_velocity - velocity,
            }
        )

    return pd.DataFrame(rows)


def main():
    """Run the IDM equilibrium-spacing validation and save the comparison.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Validation table also written to ``idm_equilibrium_spacing_validation.csv``.
    """
    comparison = compare_manual_and_code_spacing()
    comparison.to_csv(OUTPUT_FILE, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved validation table to: {OUTPUT_FILE}")

    return comparison


if __name__ == "__main__":
    main()

"""Validate FVDM equilibrium spacing against the project implementation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


REPO_DIR = Path(__file__).resolve().parent.parent
PARAMETER_FILE = REPO_DIR / "parameters.yaml"
OUTPUT_FILE = (
    Path(__file__).resolve().parent / "fvdm_equilibrium_spacing_validation.csv"
)

# Allow this validation file to be run directly from inside the Validation folder.
sys.path.insert(0, str(REPO_DIR))

from model_equations import Model_equations  # noqa: E402


def load_fvdm_parameters(parameter_file=PARAMETER_FILE):
    """Load the FVDM parameters used by the simulation code.

    Parameters
    ----------
    parameter_file : str or pathlib.Path
        Path to the repository ``parameters.yaml`` file.

    Returns
    -------
    dict
        FVDM parameter dictionary from the ``Models`` section.
    """
    with open(parameter_file, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)

    return parameters["Models"]["fvdm"]


def manual_fvdm_equilibrium_spacing(velocity, fvdm_parameters):
    """Calculate FVDM equilibrium spacing from the optimal-velocity equation.

    Parameters
    ----------
    velocity : float
        Equilibrium velocity in m/s. The value must lie within the valid
        inverse-tanh range ``V1 - V2 < velocity < V1 + V2``.
    fvdm_parameters : dict
        FVDM parameters containing ``V1``, ``V2``, ``C1``, ``C2``, and
        ``len_veh``.

    Returns
    -------
    float
        Equilibrium front-to-front spacing in metres, matching the convention
        used by ``Model_equations.init_spacing_fvdm``.
    """
    normalized_velocity = (
        (velocity - fvdm_parameters["V1"]) / fvdm_parameters["V2"]
    )
    net_spacing = (
        fvdm_parameters["C2"] + np.arctanh(normalized_velocity)
    ) / fvdm_parameters["C1"]

    # The model code returns front-to-front spacing, so vehicle length is added.
    return fvdm_parameters["len_veh"] + net_spacing


def compare_manual_and_code_spacing(velocities=None):
    """Compare manual FVDM spacings with values returned by the code.

    Parameters
    ----------
    velocities : list[float] or None
        Equilibrium velocities in m/s. When omitted, representative values
        inside the valid FVDM inverse range are used.

    Returns
    -------
    pandas.DataFrame
        Table with manual spacing, code spacing, spacing difference, and the
        velocity recovered by ``Model_equations.init_vel_fvdm``.
    """
    fvdm_parameters = load_fvdm_parameters()
    fvdm_model = Model_equations(fvdm_parameters)

    if velocities is None:
        velocities = [0.5, 2.0, 5.0, 8.0, 11.0, 14.0]

    rows = []
    min_velocity = fvdm_parameters["V1"] - fvdm_parameters["V2"]
    max_velocity = fvdm_parameters["V1"] + fvdm_parameters["V2"]

    for velocity in velocities:
        if not min_velocity < velocity < max_velocity:
            raise ValueError(
                "FVDM equilibrium spacing is finite only for "
                f"{min_velocity} < velocity < {max_velocity}."
            )

        manual_spacing = manual_fvdm_equilibrium_spacing(velocity, fvdm_parameters)
        code_spacing = fvdm_model.init_spacing_fvdm(velocity)
        recovered_velocity = fvdm_model.init_vel_fvdm(manual_spacing)

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
    """Run the FVDM equilibrium-spacing validation and save the comparison.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.DataFrame
        Validation table also written to ``fvdm_equilibrium_spacing_validation.csv``.
    """
    comparison = compare_manual_and_code_spacing()
    comparison.to_csv(OUTPUT_FILE, index=False)

    print(comparison.to_string(index=False))
    print(f"\nSaved validation table to: {OUTPUT_FILE}")

    return comparison


if __name__ == "__main__":
    main()

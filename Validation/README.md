# Validation workflow

This folder contains small, self-contained validation scripts for checking the
car-following model equations, numerical solvers, and multi-anticipation model
setup used in the repository. The scripts generate `.csv` tables and `.png`
figures in this same folder.

Run the scripts from the repository root or from inside the `Validation/`
folder. Each script inserts the repository root into `sys.path`, so both usage
patterns work.

```bash
python Validation/<script_name>.py
```

or, from inside `Validation/`:

```bash
python <script_name>.py
```

The validation scripts read model parameters from `parameters.yaml`.

## 1. Equilibrium Spacing Checks

These scripts verify that the coded equilibrium spacing relationships match the
same relationships calculated manually.

| Script | Model | Main output |
|---|---|---|
| `idm_equilibrium_spacing_validation.py` | IDM | `idm_equilibrium_spacing_validation.csv` |
| `fvdm_equilibrium_spacing_validation.py` | FVDM | `fvdm_equilibrium_spacing_validation.csv` |

Each script compares the manual equilibrium spacing calculation with the
corresponding `Model_equations.init_spacing_*()` method and inverse velocity
relation.

Expected result: the manual values and code-generated values should match to
numerical precision.

## 2. Solver Comparison Checks

These scripts compare speed profiles generated using Euler and RK4 numerical
integration. The difference should decrease as the time step is reduced.

| Script | Model | Outputs |
|---|---|---|
| `idm_solver_validation.py` | IDM | `idm_solver_comparison_dt_<dt>.*` |
| `fvdm_solver_validation.py` | FVDM | `fvdm_solver_comparison_dt_<dt>.*` |

The time step can be supplied externally:

```bash
python Validation/idm_solver_validation.py --dt 0.1
python Validation/idm_solver_validation.py --dt 0.01
python Validation/fvdm_solver_validation.py --dt 0.1
python Validation/fvdm_solver_validation.py --dt 0.01
```

The output files include the chosen time step in the file name, with decimal
points written as `p`.

## 3. Independent IDM Euler Check

`idm_validation_sun.py` compares a local Euler implementation against the Euler
method called through `solvers.py`. This confirms that the solver flow used by
the simulation class produces the same result as the explicit Euler update.

Example:

```bash
python Validation/idm_validation_sun.py --dt 0.01
```

Outputs:

| Output | Description |
|---|---|
| `idm_validation_sun_dt_<dt>.csv` | Time history and Euler residuals. |
| `idm_validation_sun_dt_<dt>.png` | Plot of the compared speed profiles and residual. |

Expected result: the residual should be close to numerical round-off.

## 4. MIDM Collapse-To-IDM Checks

The MIDM validation files check whether the multi-anticipation model behaves as
expected in limiting cases.

### `midm_factor_validation.py`

This script sets:

```text
spacing_weight = [1, 0, 0]
relvel_weight  = [1, 0, 0]
dt             = 0.1 s
```

With these weights, MIDM only responds to the immediate leader and should
collapse to IDM. The script compares all corresponding IDM and MIDM vehicle
profiles.

Outputs:

| Output | Description |
|---|---|
| `midm_factor_validation_dt_0p1.csv` | Summary residuals. |
| `midm_factor_validation_vehicle_residuals_dt_0p1.csv` | Per-vehicle residuals. |
| `midm_factor_validation_dt_0p1.png` | Plot of the comparison. |

### `midm_idm_identical.py`

This script sets the first three MIDM speed profiles to be identical and checks
that the generated MIDM follower matches the equivalent IDM follower. Because
MIDM has two virtual leader rows, the comparison is between:

```text
MIDM row 3  <->  IDM row 1
```

Outputs:

| Output | Description |
|---|---|
| `midm_idm_identical_dt_0p1.csv` | Residual table. |
| `midm_idm_identical_dt_0p1.png` | Plot of the compared trajectories. |

## 5. MIDM Two-Step Convective Check

`midm_two_step_validation.py` checks whether a downstream section of the MIDM
platoon can be reproduced by starting a second simulation from an internal
boundary of the first simulation.

The procedure is:

1. Run MIDM for physical vehicles `0` through `100`.
2. Start a second MIDM run with `50` generated followers.
3. Hard-code second-run rows `0`, `1`, and `2` using the full saved position and
   velocity histories of first-run physical vehicles `50`, `51`, and `52`.
4. Initialize second-run rows `1` through `50` from first-run physical vehicles
   `51` through `100`.
5. Compare first-run physical vehicle `100` with second-run row `50`.

Outputs:

| Output | Description |
|---|---|
| `midm_two_step_validation_dt_0p1.csv` | Speed profiles and velocity residual. |
| `midm_two_step_validation_dt_0p1.png` | Plot of the compared speed profiles and residual. |

The final printed error indicators are:

```text
maximum velocity residual
||residual|| / ||first-run speed||
```

The second value is the normalized L2 residual and is the main scale-independent
indicator of the two-step error.

## 6. MFVDM Collapse-To-FVDM Checks

The MFVDM validation files mirror the MIDM validations for the FVDM family.

### `mfvdm_factor_validation.py`

This script sets:

```text
weight = [1, 0, 0]
p      = 1
q      = 0
dt     = 0.1 s
```

With these settings, MFVDM only responds to the immediate leader and should
collapse to FVDM. The script compares all corresponding FVDM and MFVDM vehicle
profiles.

Outputs:

| Output | Description |
|---|---|
| `mfvdm_factor_validation_dt_0p1.csv` | Summary residuals. |
| `mfvdm_factor_validation_vehicle_residuals_dt_0p1.csv` | Per-vehicle residuals. |
| `mfvdm_factor_validation_dt_0p1.png` | Plot of the comparison. |

### `mfvdm_fvdm_identical.py`

This script sets the first three MFVDM speed profiles to be identical and checks
that the generated MFVDM follower matches the equivalent FVDM follower. Because
MFVDM has two virtual leader rows, the comparison is between:

```text
MFVDM row 3  <->  FVDM row 1
```

Outputs:

| Output | Description |
|---|---|
| `mfvdm_fvdm_identical_dt_0p1.csv` | Residual table. |
| `mfvdm_fvdm_identical_dt_0p1.png` | Plot of the compared trajectories. |

## 7. MFVDM Two-Step Convective Check

`mfvdm_two_step_validation.py` is the MFVDM equivalent of the MIDM two-step
check.

The procedure is:

1. Run MFVDM for physical vehicles `0` through `100`.
2. Start a second MFVDM run with `50` generated followers.
3. Hard-code second-run rows `0`, `1`, and `2` using the full saved position and
   velocity histories of first-run physical vehicles `50`, `51`, and `52`.
4. Initialize second-run rows `1` through `50` from first-run physical vehicles
   `51` through `100`.
5. Compare first-run physical vehicle `100` with second-run row `50`.

Outputs:

| Output | Description |
|---|---|
| `mfvdm_two_step_validation_dt_0p1.csv` | Speed profiles and velocity residual. |
| `mfvdm_two_step_validation_dt_0p1.png` | Plot of the compared speed profiles and residual. |

The final printed error indicators are:

```text
maximum velocity residual
||residual|| / ||first-run speed||
```

The normalized L2 residual is the main scale-independent indicator of the
two-step error.

## Suggested Run Order

For a clean validation pass, run the scripts in this order:

```bash
python Validation/idm_equilibrium_spacing_validation.py
python Validation/fvdm_equilibrium_spacing_validation.py
python Validation/idm_solver_validation.py --dt 0.1
python Validation/idm_solver_validation.py --dt 0.01
python Validation/fvdm_solver_validation.py --dt 0.1
python Validation/fvdm_solver_validation.py --dt 0.01
python Validation/idm_validation_sun.py --dt 0.01
python Validation/midm_factor_validation.py
python Validation/midm_idm_identical.py
python Validation/midm_two_step_validation.py
python Validation/mfvdm_factor_validation.py
python Validation/mfvdm_fvdm_identical.py
python Validation/mfvdm_two_step_validation.py
```

## Reading The Results

Use the `.csv` files for exact residual values and the `.png` files for a visual
check of the profiles. For identity and collapse checks, residuals should be
near numerical precision. For solver comparisons, residuals should reduce as
`dt` is reduced. For the two-step checks, use the reported normalized residual:

```text
||residual|| / ||first-run speed||
```

This value makes the error easier to compare across models and speed profiles
because it scales the residual by the magnitude of the reference first-run
speed profile.

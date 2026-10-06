# CSV descriptors for `Manuscript_results/`

This file describes the generated CSV outputs used by the manuscript plotting
and summary scripts. The same schemas are repeated across many model, drive
cycle, engine, and parameter combinations.

Common categorical fields:

| Column | Description |
|---|---|
| `model` | Car-following model name, such as `idm`, `midm`, `fvdm`, or `mfvdm`. |
| `velocity_class` | Leader speed profile or drive-cycle name. |
| `engine_type` | Powertrain or fuel code, such as `G`, `D`, `G_HEV`, `D_HEV`, or `B`. |
| `parameter` | Model parameter varied in the sensitivity or uncertainty run. |
| `parameter_value` | Numeric value used for the varied parameter. |
| `vehicle` | Reported vehicle index. |
| `simulation_vehicle` | Vehicle row used in the simulation arrays. |

## `outputs/instantaneous_data/*_instantaneous_outputs.csv`

Instantaneous vehicle-level PHEMlight outputs joined with simulation states.

Rows are vehicle-time observations. Each row is one vehicle at one sampled time
step.

| Column | Unit | Description |
|---|---:|---|
| `model` | - | Car-following model. |
| `velocity_class` | - | Leader speed profile or drive cycle. |
| `engine_type` | - | Fuel or powertrain type. |
| `vehicle` | - | Reported vehicle index. |
| `simulation_vehicle` | - | Vehicle row used internally by the simulation. |
| `time` | s | Time stamp. |
| `speed` | m/s | Vehicle speed. |
| `power_pos` | kW | Positive traction power from PHEMlight. |
| `co2` | g/h | Instantaneous CO2 rate. |
| `nox` | g/h | Instantaneous NOx rate. |
| `pm` | g/h | Instantaneous PM rate. |
| `fc` | g/h or kWh/h | Fuel consumption rate for fuel vehicles; battery energy rate for `B`. |
| `position` | m | Vehicle longitudinal position. |

For plotting, emissions rates are often converted from `g/h` to `g/s`.
Fuel-consumption plotting converts fuel vehicles to `L/s` using fuel density,
while battery files keep `fc` as `kWh/h`.

## `outputs/total_data/*_total_data.csv`

Vehicle totals for one model, drive cycle, and engine type.

Rows are mostly individual vehicles. Some workflows may also include aggregate
rows, identified by `row_type`.

| Column | Unit | Description |
|---|---:|---|
| `row_type` | - | Row category, commonly `vehicle` or an aggregate row. |
| `model` | - | Car-following model. |
| `velocity_class` | - | Leader speed profile or drive cycle. |
| `engine_type` | - | Fuel or powertrain type. |
| `vehicle` | - | Reported vehicle index. |
| `simulation_vehicle` | - | Vehicle row used internally by the simulation. |
| `evaluation_start_s` | s | Start time used for total calculation. |
| `evaluation_end_s` | s | End time used for total calculation. |
| `total_distance_m` | m | Distance travelled during the evaluation window. |
| `total_time_s` | s | Evaluation duration. |
| `average_speed_mps` | m/s | Mean speed over the evaluation window. |
| `total_co2_g` | g | Total CO2 over the evaluation window. |
| `total_nox_g` | g | Total NOx over the evaluation window. |
| `total_pm_g` | g | Total PM over the evaluation window. |
| `total_fc` | see `fc_unit` | Total fuel or battery energy over the evaluation window. |
| `fc_unit` | - | Unit for `total_fc`; commonly `g`, `L`, or `kWh` depending on file and workflow. |

## `outputs/master_data/<model>_total_data.csv`

Master total files concatenating many parameter, drive-cycle, and engine
combinations for one model.

These files contain all columns from `outputs/total_data/*_total_data.csv` plus
parameter metadata and `max_speed_mps`.

| Column | Unit | Description |
|---|---:|---|
| `row_type` | - | Row category, commonly `vehicle` or an aggregate row. |
| `model` | - | Car-following model. |
| `parameter` | - | Varied model parameter. |
| `parameter_value` | parameter unit | Value used for the varied model parameter. |
| `velocity_class` | - | Leader speed profile or drive cycle. |
| `engine_type` | - | Fuel or powertrain type. |
| `vehicle` | - | Reported vehicle index. |
| `simulation_vehicle` | - | Vehicle row used internally by the simulation. |
| `evaluation_start_s` | s | Start time used for total calculation. |
| `evaluation_end_s` | s | End time used for total calculation. |
| `total_distance_m` | m | Distance travelled during the evaluation window. |
| `total_time_s` | s | Evaluation duration. |
| `average_speed_mps` | m/s | Mean speed over the evaluation window. |
| `max_speed_mps` | m/s | Maximum speed over the evaluation window. |
| `total_co2_g` | g | Total CO2 over the evaluation window. |
| `total_nox_g` | g | Total NOx over the evaluation window. |
| `total_pm_g` | g | Total PM over the evaluation window. |
| `total_fc` | see `fc_unit` | Total fuel or battery energy over the evaluation window. |
| `fc_unit` | - | Unit for `total_fc`. |

## `outputs/master_data/filtered_cov_by_parameter.csv`

Filtered coefficient-of-variation summary by model, driving condition, speed
profile, engine type, and varied parameter.

Rows are parameter groups after filtering to valid parameter combinations.

| Column group | Unit | Description |
|---|---:|---|
| Group identifiers | - | Model, driving condition, speed profile, engine, and parameter. |
| `n_parameter_values` | count | Number of parameter values retained in the group. |
| `<quantity>_mean_per_km` | g/km, L/km, or kWh/km | Mean normalized result. |
| `<quantity>_std_per_km` | g/km, L/km, or kWh/km | Standard deviation of the per-km result. |
| `<quantity>_mean_total` | g, L, or kWh | Mean total result for the group. |
| `<quantity>_std_total` | g, L, or kWh | Standard deviation of total result. |
| `<quantity>_cov` | - | Coefficient of variation, `std / mean`. |
| `<quantity>_cov_percent` | % | Coefficient of variation expressed as a percentage. |

The `<quantity>` prefix is one of `co2`, `nox`, `pm`, or `fc`.

## `outputs/plots/*_vehicle_totals.csv`

Vehicle totals written by plotting scripts for a selected drive-cycle and model
pair/group.

Rows are vehicles.

| Column | Unit | Description |
|---|---:|---|
| `model` | - | Car-following model. |
| `vehicle` | - | Vehicle index. |
| `co2_g` | g | Total CO2. |
| `nox_g` | g | Total NOx. |
| `pm_g` | g | Total PM. |
| `fc_g` | g | Total fuel consumption when fuel-based data are plotted. |

## `outputs/plots/fleet_emissions/*_valid_parameter_rows.csv`

Filtered parameter rows used in fleet-emissions plots.

Rows are valid model, drive-cycle, engine, and parameter combinations.

| Column | Unit | Description |
|---|---:|---|
| `model` | - | Car-following model. |
| `velocity_class` | - | Leader speed profile or drive cycle. |
| `engine_type` | - | Fuel or powertrain type. |
| `parameter` | - | Varied model parameter. |
| `parameter_value` | parameter unit | Value used for the varied parameter. |
| `max_speed_kmh` | km/h | Maximum speed reached by the selected vehicle/run. |
| `co2_g_per_km` | g/km | CO2 normalized by travelled distance. |
| `nox_g_per_km` | g/km | NOx normalized by travelled distance. |
| `pm_g_per_km` | g/km | PM normalized by travelled distance. |
| `fc_per_km` | g/km, L/km, or kWh/km | Fuel or battery energy normalized by travelled distance. |

## `outputs/plots/fleet_emissions/*_fleet_emissions_summary.csv`

Fleet-emissions summary over valid parameter combinations.

Rows are model, drive-cycle, and engine groups.

| Column | Unit | Description |
|---|---:|---|
| `model` | - | Car-following model. |
| `velocity_class` | - | Leader speed profile or drive cycle. |
| `engine_type` | - | Fuel or powertrain type. |
| `valid_parameter_combinations` | count | Number of valid parameter rows included in the group. |
| `<quantity>_g_per_km_mean` | g/km | Mean per-km emissions for `co2`, `nox`, or `pm`. |
| `<quantity>_g_per_km_std` | g/km | Standard deviation of per-km emissions. |
| `fc_per_km_mean` | g/km, L/km, or kWh/km | Mean fuel or battery energy normalized by distance. |
| `fc_per_km_std` | g/km, L/km, or kWh/km | Standard deviation of normalized fuel or energy. |

For emissions columns, `<quantity>` is one of `co2`, `nox`, or `pm`.

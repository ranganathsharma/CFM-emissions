# CSV descriptors for `Demo/`

This file describes the CSV files generated or consumed by the demo workflow.
The demo focuses on one IDM, WLTC rural, gasoline example.

## `Saved_inputs/*_veh_<n>.csv`

Headerless input files passed to the PHEMlight wrapper.

Example:

- `Saved_inputs/idm_WLTC_rural_gasoline_veh_0.csv`

Structure:

| Dimension | Unit | Description |
|---|---:|---|
| Row | - | One row containing the full speed profile for one vehicle. |
| Column | s | Each column is one successive time step, sampled at 1 s intervals. |
| Cell value | m/s | Vehicle speed at that time step. |

The vehicle number is encoded in the file name. For example, `veh_0` is the
leader and higher numbers are followers.

## `Saved_outputs/*_veh_<n>_Output.csv`

Raw PHEMlight output files for one vehicle.

Example:

- `Saved_outputs/idm_WLTC_rural_gasoline_veh_0_Output.csv`

File layout:

| Row | Description |
|---:|---|
| 1 | Vehicle type metadata, such as `Vehicletype: ,PC_EU6ab_G`. |
| 2 | PHEMlight output column names. |
| 3 | PHEMlight output units. |
| 4 onward | One time step per row. |

Main columns:

| Column | Unit | Description |
|---|---:|---|
| `Time` | s | Time stamp. |
| `Speed` | m/s | Vehicle speed. |
| `Gradient` | % | Road gradient. |
| `Accelaration` | m/s^2 | Vehicle acceleration. The spelling follows the PHEMlight output. |
| `Engine power raw` | kW | Raw engine power. |
| `P_pos` | kW | Positive traction power. |
| `P_norm_rated` | - | Power normalized by rated power. |
| `P_norm_drive` | - | Power normalized by drive-cycle reference. |
| `FC` | g/h | Fuel consumption rate for fuel vehicles. |
| `Engine Power` | kWh/h | Battery energy rate when electric output is used. |
| `CO2` | g/h | Carbon dioxide emission rate. |
| `NOx` | g/h | Nitrogen oxides emission rate. |
| `CO` | g/h | Carbon monoxide emission rate. |
| `HC` | g/h | Hydrocarbon emission rate. |
| `PM` | g/h | Particulate matter emission rate. |
| `PN` | #/h | Particle number rate. |

## `outputs/*_instantaneous_emissions.csv`

Combined instantaneous demo output for all vehicles.

Example:

- `outputs/idm_WLTC_rural_gasoline_instantaneous_emissions.csv`

Rows are vehicle-time observations. Each row describes one vehicle at one time
step.

| Column | Unit | Description |
|---|---:|---|
| `Time` | s | Time stamp. |
| `Speed` | m/s | Vehicle speed. |
| `CO2` | g/h | Instantaneous PHEMlight CO2 rate. |
| `NOx` | g/h | Instantaneous PHEMlight NOx rate. |
| `PM` | g/h | Instantaneous PHEMlight PM rate. |
| `FC` | g/h | Instantaneous PHEMlight fuel consumption rate. |
| `vehicle` | - | Vehicle index; `0` is the leader. |
| `position` | m | Vehicle longitudinal position. |
| `CO2_g` | g | CO2 emitted during this time step. |
| `NOx_g` | g | NOx emitted during this time step. |
| `PM_g` | g | PM emitted during this time step. |
| `FC_g` | g | Fuel consumed during this time step. |

The per-step mass columns are obtained by converting rates from per hour to the
mass emitted or consumed during the sampled time interval.

## `outputs/*_vehicle_totals.csv`

Total demo emissions and fuel consumption by vehicle.

Example:

- `outputs/idm_WLTC_rural_gasoline_vehicle_totals.csv`

Rows are vehicles.

| Column | Unit | Description |
|---|---:|---|
| `vehicle` | - | Vehicle index; `0` is the leader. |
| `co2_g` | g | Total CO2 over the evaluated duration. |
| `nox_g` | g | Total NOx over the evaluated duration. |
| `pm_g` | g | Total PM over the evaluated duration. |
| `fc_g` | g | Total fuel consumption over the evaluated duration. |

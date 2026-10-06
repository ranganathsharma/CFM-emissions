This README file provides step by step instructions to run the demonstration workflow for CFM simulation and emission
estimation. The Python demo files and the demo PHEMlight C# wrapper are in this `Demo/` folder. Shared source files,
input data, and parameter files remain in the parent repository folder.

============================================================================================================================
# SETUP OF PYTHON ENVIRONMENT
============================================================================================================================

1. Open a terminal or command prompt.

2. Move to the parent repository folder, where `requirements.txt` is located:

    cd ..

3. Create a Python virtual environment:

    python -m venv .venv

4. Activate the virtual environment.

    On Windows:

        .venv\Scripts\activate

        Note: On Windows PowerShell, if script execution is blocked, run:

            Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

        Then activate the environment with:

            .\.venv\Scripts\Activate.ps1

    On macOS/Linux:

        source .venv/bin/activate

5. Install the required Python packages:

    python -m pip install -r requirements.txt

6. Return to the demo folder before running the demo scripts:

    cd Demo

============================================================================================================================
# CFM SIMULATION
============================================================================================================================

A demonstration of solving IDM using the WLTC_rural leader speed profile is provided in `cfm_solution.py`. From this `Demo/`
folder, run:

    python cfm_solution.py

Generation of CFM trajectories uses `model_equations.py`, `solvers.py`, `parameters.yaml`, and the `data/` folder from the
parent repository. This creates `Demo/outputs/idm_wltc_rural_velocity.png`.

============================================================================================================================
# EMISSION CALCULATION SETUP
============================================================================================================================

The calculation of emission and fuel consumption rates requires a licensed installation of the proprietary PHEMlight software. A C# wrapper for PHEMlight is provided in `Demo/PhemApp/`. From this `Demo/` folder, build the wrapper with:

    dotnet build PhemApp/PhemApp.csproj -c Release

This creates the executable used by the Python pipeline to run PHEMlight calculations.

Note: This step requires access to a licensed PHEMlight installation and cannot be completed without it.

============================================================================================================================
# EMISSION RATE ESTIMATION 
============================================================================================================================  
The trajectories generated using the car-following model are used to calculate instantaneous CO2, NOx, PM, and fuel consumption rates. A detailed demonstration of the workflow is provided in `emission_estimation.py`. From this `Demo/` folder, run:

    python emission_estimation.py

This creates `Demo/outputs/idm_WLTC_rural_gasoline_instantaneous_emissions.csv`. The output file contains position, velocity, instantaneous emissions, and instantaneous fuel consumption rates for all vehicles.

Note: This step requires access to a licensed PHEMlight installation and cannot be completed without it.

============================================================================================================================
# EMISSION RATE PLOTTING 
============================================================================================================================ 
The instantaneous velocity, fuel consumption rate, and vehicle specific power (VSP) are plotted for the leader, 10th vehicle, and 19th vehicle.

VSP is calculated as:

    VSP = speed * (1.1 * acceleration + rolling_resistance_coefficient * gravitational_acceleration)
          + (air_density * drag_coefficient * frontal_area * speed^3) / (2 * vehicle_mass)

The constants used in this calculation are taken from the Euro 6ab gasoline passenger car parameters in PHEMlight.

The script uses `Demo/outputs/idm_WLTC_rural_gasoline_instantaneous_emissions.csv`. From this `Demo/` folder, run:

    python Emission_rate.py

This generates `Demo/outputs/idm_WLTC_rural_gasoline_velocity_vsp_fc.png`.

============================================================================================================================
# EMISSION RATE HISTOGRAM PLOTTING 
============================================================================================================================ 
Histograms of instantaneous velocity, VSP, and fuel consumption rate are plotted for the leader, 10th vehicle, and 19th vehicle. The script uses `Demo/outputs/idm_WLTC_rural_gasoline_instantaneous_emissions.csv`. From this `Demo/` folder, run:

    python Emission_histograms.py

This generates `Demo/outputs/idm_WLTC_rural_gasoline_velocity_vsp_fc_histograms.png`.

============================================================================================================================
# TOTAL EMISSION PLOTTING 
============================================================================================================================ 
Total emissions are calculated for each vehicle over a fixed duration, starting from the time the vehicle first begins moving.

From this `Demo/` folder, run:

    python Total_emissions.py

This generates `Demo/outputs/idm_WLTC_rural_gasoline_vehicle_totals.png`.

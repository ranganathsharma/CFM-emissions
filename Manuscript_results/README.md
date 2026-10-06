The data and figures presented in the manuscript are generated using the same estimation process presented in the `demo/`
folder. 

============================================================================================================================
# PARAMETER VALUES
============================================================================================================================

Starting with the parameter values listed in the `parameters.yaml` file, depending on the leader speed profile, some 
parameters of CFMs and MCFMs are modified to fit the driving condition. The changed parameter values are provided in the 
table below:

|---|---|---|---|
| Model | Urban | Rural | Highway |
|---|---|---|---|
| IDM | v_0 = 16.67 m/s | v_0 =  22.22 m/s | NA |
| MIDM | v_0 = 16.67 m/s | v_0 =  22.22 m/s | NA |
|---|---|---|---|
| FVDM | NA | NA | V_2 = 15 m/s |
| MFVDM | NA | NA | V_2 = 15 m/s |
|---|---|---|---|

Note: NA corresponds to all parameter values of the model identical to those provided in the `parameters.yaml` file. The 
parameters are kept unchanged for the case of real driving speed profile as it comprises of urban, rural and highway driving.  

============================================================================================================================
# INSTANTANEOUS EMISSION RATE DATA GENERATION
============================================================================================================================

The files `Validation_cfm.py` and `Validation_mcfm.py` calculate the instantaneous emission rates and fuel consumption rate 
for a single set of CFM parameter values. Instantaneous rates for selected vehicles are saved in 
`outputs/instantaneous_data/`, and total emission and fuel consumption data are saved in `outputs/total_data/`. Running the 
file with the command 

    python Validation_cfm.py

generates the emission and fuel consumption data for ICE, HEV and BEVs for IDM and FVDM. For each model, 45 different files 
corresponding to different powertrain-fuel-leader speed profile combinations are generated and saved with the format 
`outputs/instantaneous_data/{model name}_{leader speed profile}_{engine type}_instantaneous_outputs.csv`. The corresponding 
total-data files are saved with the format 
`outputs/total_data/{model name}_{leader speed profile}_{engine type}_total_data.csv`. Similarly, the data for MCFMs is 
generated using the command 

    python Validation_mcfm.py

Note: This step requires access to a licensed PHEMlight installation and cannot be completed without it.

============================================================================================================================
# PLOTTING
============================================================================================================================

The file `Emission_rate.py` uses the instantaneous emission rate data from 
`outputs/instantaneous_data/{model name}_{leader speed profile}_{engine type}_instantaneous_outputs.csv` and plots the 
velocity, VSP and fuel consumption rates for all leader speed profile-fuel-powertrain combinations. Run the file using the 
command: 

    python Emission_rate.py

The resulting .png files are saved in the folder `outputs/plots/time_series`. 

The file `Emission_histograms.py` uses the instantaneous emission rate data from 
`outputs/instantaneous_data/{model name}_{leader speed profile}_{engine type}_instantaneous_outputs.csv` and plots the
velocity, VSP and fuel consumption rate histograms for all leader speed profile-fuel-powertrain combinations. Run the file 
using the command: 

    python Emission_histograms.py

The resulting .png files are saved in the folder `outputs/plots/histograms`.

The file Total_emissions.py uses the instantaneous emission rate data from 
`outputs/instantaneous_data/{model name}_{leader speed profile}_{engine type}_instantaneous_outputs.csv` and plots the 
total CO2, NOx, PM emission levels and fuel consumption rates for all leader speed profile-fuel-powertrain combinations. 
Run the file using the command: 

    python Total_emissions.py

The resulting .png files are saved in the folder `outputs/plots/total_emissions`.

============================================================================================================================
# FLEET EMISSION DATA GENERATION
============================================================================================================================

The files `combinations_cfm.py` and `combinations_mcfm.py` calculate the total emission and fuel consumption per unit 
distance for a wide range of parameter combinations and save the master parameter-sweep data in the format 
`outputs/master_data/{model name}_total_data.csv`. Run the files with the command 

    python combinations_cfm.py
    python combinations_mcfm.py

One master file is generated for each model. Each master file contains all parameter combinations, powertrain-fuel cases, 
and leader speed profiles for that model. 

Note: This step requires access to a licensed PHEMlight installation and cannot be completed without it.
Note: The master data sets are provided in the google drive link: https://drive.google.com/drive/folders/1wbXLQRo8ghCqzylmd7u0iMHrsYw85SEM?usp=sharing. In case the folder cannot be downloaded, please contact brra@tcd.ie for 
access. 

============================================================================================================================
# FLEET EMISSIONS PLOTTING
============================================================================================================================

The average fleet emissions of a model is calculated separately over different driving conditions. Only CFM parameter 
combinations that generate speed profiles that describe urban, rural or highway driving are considered. The file 
`Fleet_emissions.py` checks whether each parameter combination is applicable to each driving condition and plots the average 
fleet emissions for each leader speed profile.

Run the file using the command: 

    python Fleet_emissions.py

This code generates `{leader speed profile}_valid_parameter_emission_distance_bars.png` for each leader speed profile in 
`outputs/plots/fleet_emissions`.

============================================================================================================================
# FLEET BATTERY ENERGY CONSUMPTION PLOTTING
============================================================================================================================

The battery electric energy consumption rate histograms of 50th and 200th vehicle in the platoon are plotted using the 
instantaneous energy consumption rate data files 
`outputs/instantaneous_data/{model name}_{leader speed profile}_{engine type}_instantaneous_outputs.csv`. Run the file using 
the command 

    python Battery_histograms.py

This generates the files `{leader speed profile}_idm_midm_fvdm_mfvdm_B_fc_powertrain_histogram.png` in 
`outputs/plots/battery_histograms`.

The file `Total_electric_consumption.py` uses the instantaneous BEV energy consumption data and plots the total electric 
energy consumed per unit distance by all vehicles. Run the file using the command:

    python Total_electric_consumption.py

This generates `B_instantaneous_vehicle_fc_per_distance_scatter_3x3.png` in `outputs/plots/battery_totals`.

The file `Fleet_battery.py` uses the valid parameter rows generated by `Fleet_emissions.py` and plots the average fleet 
battery energy consumption by drive cycle. Run the file using the command:

    python Fleet_battery.py

This generates `battery_energy_consumption_by_drive_cycle.png` in `outputs/plots/fleet_battery`.

============================================================================================================================
# SENSITIVITY ANALYSIS PLOTTING
============================================================================================================================

The change in average fleet emissions when parameter values are changed under each leader speed profile is plotted using the
file Sensitivity_analysis.py. To limit the parameter values within the range allowed for each driving condition, the function
uses the data file `filtered_cov_by_parameter.csv`. Run the file with the command 

    python Sensitivity_analysis.py

This command generates `outputs\plots\sensitivity_analysis\{model}_{engine}_{pollutant}_filtered_cov_sensitivity.png` which 
plots the coefficient of variation of fleet emissions and fuel consumption for fuel-powertrain-model combinations. In 
addition, the figures 
`outputs\plots\sensitivity_analysis\{model}_{fuel}_{pollutant}_all_profiles_filtered_emission_parameters.png` 
show the change in fleet emissions as the parameter value is changed. 
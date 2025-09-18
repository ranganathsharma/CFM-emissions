
import subprocess
import os, csv, sys, yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Construct the full path to the C# executable
exe_path = os.path.join("PhemApp", "bin", "Release", "net9.0", "PhemApp.exe")

# Run the phemlight software to get the emissions for all the files that are in the Iterative_inputs folder

def emission_calculation(engine, standard):
    # Run the executable
    result = subprocess.run([exe_path, engine, standard], capture_output=True, text=True)
    if result.returncode != 0:
        print("Execution failed with code:", result.returncode)



def create_inputs(model: str, vc: int, num: int):

    """Method to create the input files of velocity

    Parameters
    ----------
    model : str
        Name of the model
    vc : int
        Name of the velocity class whose data is being processed
    num : int
        Number of vehicles whose velocity profiles are to be created separately
    """
    v_values = np.loadtxt(rf'C:\Users\Administrator\OneDrive - Trinity College Dublin\Academic\PhD\NCFM\codes\nonlinear\Data\{model}_original_velocity_class_velocity{vc}.csv', delimiter=',')
    for i in range(num):
        with open(rf'Iterative_inputs\{model}_vel_vc_{vc}_{i}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(v_values[i])

def delete_files(type: str):
    for folder_path in [f'Iterative_{type}']:

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def post_processing(model: str, vc: int = 2):

    cols = [' Speed', 'CO2', 'NOx', 'CO', 'PM', 'HC', 'FC']
    avg_names = [i + '_avg' for i in cols]
    a = pd.DataFrame(columns = ['model', 'vehicle', 'fuel'] + avg_names)

    if not os.path.exists('Emission_details.csv'):
        a.to_csv('Emission_details.csv', index=False)

    for i in range(0, num):
        first_row = pd.read_csv(rf'Iterative_outputs\{model}_vel_vc_{vc}_{i}_Output.csv', nrows=1, header=None)
        fuel_type = first_row.iat[0, 1][-1]  # 0th row, 2nd column (index 1)

        df = pd.read_csv(rf'Iterative_outputs\{model}_vel_vc_{vc}_{i}_Output.csv', skiprows = [0,2], index_col=False)
        df = df[df['Time'] > 1200]
        df = df[cols]

        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        averages = df.mean().to_dict()

        # Build result row
        row = {
            'model': model,
            'vehicle': i,
            'fuel': fuel_type
        }
        row.update({f'{k}_avg': v for k, v in averages.items()})

        # Append to CSV
        pd.DataFrame([row]).to_csv('Emission_details.csv', mode='a', header=False, index=False)

if __name__ == '__main__':

    for model in ['newell_delay', 'tsh', 'gfm']:
        vc = 2
        num = 1000

        create_inputs(model = model,
                    vc = vc,
                    num = num)
        
        emission_calculation()

        delete_files('inputs')

        post_processing(model = model, vc=vc)

        delete_files('outputs')

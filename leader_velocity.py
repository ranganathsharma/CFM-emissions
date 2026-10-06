import os

import numpy as np
import pandas as pd
from scipy import interpolate

LOCATION = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def wltc(velocity_class: int, dt: float = 0.1) -> list[np.ndarray]:
    """
    Method yielding wltc velocity profile

    Parameters
    ----------
    velocity_class : int
        velocity class
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    list[np.ndarray]
        time_array: 1d array of time
        velocity_array: 1d array of leader velocity

    Raises
    ------
    SystemExit
        Input Error if the velocity class is not correct
    """

    if velocity_class in [1, 2, 3]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\WLTP_class_2_DC.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)
        
    elif velocity_class in [4]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\Artemis_urban.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [5]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\Artemis_rural.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [6]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\Artemis_mw.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [7]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\FTP.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)
        
    elif velocity_class in [8]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\HWFET.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [9]:
        dataframe_dc = pd.read_csv(os.path.join(LOCATION, f'data\\Ford_focus.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [10]:
        time = np.arange(0, 100)
        velocity = np.array([20 for t in time])

        time_array = np.arange(0, time[-1] + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    else:
        raise SystemExit(f'Input Error: The class: {velocity_class} of wltc does not exist')
    
    velocity_array = interpolation_function(time_array)/3.6 # The 3.6 factor is considered to convert velocity units from km/h to m/s
    velocity_array = [0 if v < 0 else v for v in velocity_array]
    return [time_array, velocity_array]

import sys, os
import numpy as np
from scipy import interpolate
import pandas as pd

location = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def vel_increase(vel_initial: float,
                 vel_final: float,
                 dt: float = 0.1) -> np.ndarray:
    """
    Method to increase the velocity from vel_initial to vel_final with a cos curve

    Parameters
    ----------
    vel_initial : float
        Initial velocity
    vel_final : float
        Final velocity
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    np.ndarray
        1d velocity array
    """

    if vel_initial < vel_final:
        acc_max = 2
    else:
        acc_max = -3

    time_dur = np.pi*(vel_final - vel_initial)/(2*acc_max)
    time = np.arange(0, time_dur, dt)
    velocity = vel_initial + (vel_final-vel_initial)*0.5*(1 - np.cos(np.pi * time/time_dur))

    return velocity

def vel_constant(vel_mean: float, time_duration: float, dt: float = 0.1) -> np.ndarray:
    """
    Method to maintain constant velocity

    Parameters
    ----------
    vel_mean : float
        constant velocity maintained
    time_duration : float
        Total time duration the velocity is maintained constant for
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    np.ndarray
        1d velocity
    """

    time_array = np.arange(0, time_duration, dt)
    velocity = vel_mean*np.ones((time_array.shape[0]))

    return velocity

def vel_lead_acc(dt: float = 0.1) -> list[np.ndarray]:
    """
    Method to temporarily increase leader velocity

    Parameters
    ----------
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    list[np.ndarray]
        time_array: 1d array of time
        velocity_array: 1d array of leader velocity
    """
    # Acceleration set to 12 m/s
    segments = {
        1: {'method': vel_constant, 'inputs': (8, 600)},
        2: {'method': vel_increase, 'inputs': (8, 12)},
        3: {'method': vel_constant, 'inputs': (12, 5)},
        4: {'method': vel_increase, 'inputs': (12, 8)},
        5: {'method': vel_constant, 'inputs': (8, 3000)}
    }

    main_velocity_array = [_segment['method'](*_segment['inputs']) for _segment in segments.values()]
    velocity_array = np.concatenate(main_velocity_array)
    time_max = len(velocity_array)*dt
    time_array = np.arange(0, time_max, dt)

    return [time_array, velocity_array]

def vel_lead_dec(dt:float = 0.1) -> list[np.ndarray]:
    """
    Method to temporarily decrease leader velocity

    Parameters
    ----------
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    list[np.ndarray]
        time_array: 1d array of time
        velocity_array: 1d array of leader velocity
    """
    # Deceleration set to 1 m/s
    segments = {
        1: {'method': vel_constant, 'inputs': (8, 600)},
        2: {'method': vel_increase, 'inputs': (8, 1)},
        3: {'method': vel_constant, 'inputs': (1, 5)},
        4: {'method': vel_increase, 'inputs': (1, 8)},
        5: {'method': vel_constant, 'inputs': (8, 3000)}
    }

    main_velocity_array = [_segment['method'](*_segment['inputs']) for _segment in segments.values()]
    velocity_array = np.concatenate(main_velocity_array)
    time_max = len(velocity_array)*dt
    time_array = np.arange(0, time_max, dt)

    return [time_array, velocity_array]

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
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\WLTP_class_2_DC.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)
        
    elif velocity_class in [4]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\Artemis_urban.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [5]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\Artemis_rural.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [6]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\Artemis_mw.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [7]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\FTP.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)
        
    elif velocity_class in [8]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\HWFET.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    elif velocity_class in [9]:
        dataframe_dc = pd.read_csv(os.path.join(location, f'data\\Ford_focus.csv'))
        time = dataframe_dc['Time in s'].to_numpy()
        velocity = dataframe_dc['Speed in kmph'].to_numpy()
        
        time_array = np.arange(0, len(time)-1 + dt, dt)
        interpolation_function = interpolate.PchipInterpolator(time, velocity)

    else:
        raise SystemExit(f'Input Error: The class: {velocity_class} of wltc does not exist')
    
    velocity_array = interpolation_function(time_array)/3.6 # The 3.6 factor is considered to convert velocity units from km/h to m/s
    velocity_array = [0 if v < 0 else v for v in velocity_array]
    return [time_array, velocity_array]

def vel_rand(amplitude: float, velocity_mean: float, time_duration: int, cubic = True, dt: float = 0.1) -> np.ndarray:
    """
    Method yielding fluctuating velocity around velocity_mean

    Parameters
    ----------
    amplitude : float
        Amplitude of velocity variation around velocity_mean
    velocity_mean : float
        Mean velocity around with the values fluctuate
    time_duration : int
        Time duration
    cubic : bool, optional
        interpolation method if True: cubic, if false: linear, by default True
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    np.ndarray
        1d velocity array
    """

    _time = np.arange(0, time_duration + 1, 10) # 10 is the gap between the readings
    _velocity = np.random.normal(velocity_mean, amplitude, (len(_time)))
    _velocity[0] = _velocity[-1] = velocity_mean
    
    _velocity = [0 if _v < 0 else _v for _v in _velocity]

    if cubic:
        interpolation_function = interpolate.CubicSpline(_time, _velocity)
    else:
        interpolation_function = interpolate.interp1d(_time, _velocity, kind = 'linear')

    time_array = np.arange(0, len(_time)*10 - 10 + dt, dt)
    velocity_array = interpolation_function(time_array)
    
    return velocity_array

def vel_leader_rand(velocity_initial: float,
                    velocity_final: float,
                    amplitude: float,
                    dt: float = 0.1) -> list[np.ndarray]:
    """
    Method yielding velocity profile with temporary increase in velocity and then a dip

    Parameters
    ----------
    velocity_initial : float
        starting velocity
    velocity_final : float
        ending velocity
    amplitude : float
        amplitude of fluctuation
    dt : float, optional
        simulation time step, by default 0.1

    Returns
    -------
    list[np.ndarray]
        time array: 1d time
        velocity array: 1d velocity

    Raises
    ------
    ValueError
        If the velocity exceeds 100 m/s
    """

    segments = {
        1: {'method': vel_increase,'inputs': (velocity_initial, 16, np.round((16 - velocity_initial)*2), dt) }, # The leader goes with velocity 16 m/s only
        2: {'method': vel_rand,'inputs': (amplitude, 16, 2400, dt)},
        3: {'method': vel_increase,'inputs': (16, velocity_final, np.round((16 - velocity_final)*0.8), dt)},
        4: {'method': vel_rand,'inputs': (amplitude, velocity_final, 200, dt)}
    }
    
    # main_velocity_array = np.array([[_method(*_inputs)] for _no, _segment in segments.items() for _method, _inputs in _segment.items()])
    main_velocity_array = [_segment['method'](*_segment['inputs']) for _segment in segments.values()]
    velocity_array = np.concatenate(main_velocity_array)
    time_max = len(velocity_array)*dt
    time_array = np.arange(0, time_max, dt)

    if np.max(velocity_array) > 100:
        raise ValueError('The velocity value is unrealistic')
    
    return [time_array, velocity_array]
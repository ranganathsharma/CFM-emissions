import numpy as np
from typing import Callable

class solvers():

    def __init__(self,
                 road_len: float = 1000) -> None:
        """
        Class of solver methods

        Euler, Runge Kutta and others. 

        {solver}_{road}_{order} is the format of the methods

        solver: euler / RK (runge kutta 4th order)
        road: closed/open 
        order: first/second

        Parameters
        ----------
        road_len : float, optional
            Length of the road for ring road condition, by default 0

        Summary of each solver method:

        Parameters
        ----------
        x : np.ndarray
            position array of all vehicles at time 0
        v : np.ndarray
            velocity array of all vehicles at time 0
        v_next : np.ndarray
            velocity of the next time step (applicable only when leader velocity is known)
        dt : float
            simulation time step
        u_dot : function
            velocity rate equation
        x_dot : _type_
            position rate equation

        Returns
        -------
        tuple of np.ndarray
            updated position x_new and velocity v_new
        """

        self.road_len = road_len

    def euler_closed_second(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        dx = u_dot(x, v)*dt
        dv = x_dot(x, v)*dt + 0.5*u_dot(x, v)*dt**2

        x_new = (x + dx)%self.road_len
        v_new = np.maximum(v + dv, v*0)

        return x_new, v_new
    
    # def euler_open_second(self, x, v, v_next, dt, u_dot, x_dot):

    #     dv = u_dot(x, v)*dt
    #     dx = x_dot(x, v)*dt + 0.5*u_dot(x, v)*dt**2

    #     # if np.min(dx) < -1e-9:
    #     #     print('Negative movement was predicted')
    #     #     raise OverflowError()

    #     # new_dx = np.where((-1e-9 < dx) & (dx < 0), 0, dx)
    #     new_dx = np.where(dx < 0, 0, dx)
    #     x_new = x + new_dx

    #     # if np.min(new_dx) < -1e-9:
    #     #     print(f'The vehicle {np.where(new_dx < 0)} moved back. There is a problem with the solver')
    #     #     print(u_dot(x, v)[1], x[0], x[1], v[0], v[1], np.min(new_dx))
    #     #     raise OverflowError('Vehicle is set to move back')
        
    #     v_new = np.maximum(v + dv, v*0)
    #     # v_new = np.where(v + dv < 0, 0, v + dv)
    #     return x_new, v_new

    def euler_open_second(self, x, v, v_next, dt, u_dot, x_dot):
        

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        threshold = 1e-9

        dv = u_dot(x, v)*dt
        dx = x_dot(x, v)*dt

        dv = np.where(np.abs(dv) < threshold, 0, dv)
        dx = np.where(dx < threshold, 0, dx)

        dv[0] = v_next[0] - v[0]

        x_new = x + dx
        v_new = np.maximum(v + dv, v*0)

        return x_new, v_new
            
    def euler_closed_first(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        x_new = x + x_dot(x, v)*dt
        v_new = (x_new - x)/dt
        x_new = x_new%self.road_len

        return x_new, v_new
    
    def euler_open_first(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        x_new = x + x_dot(x, v)*dt
        v_new = (x_new - x)/dt


        # v_new = np.maximum(v_new, v*0)

        return x_new, v_new
    
    def euler_delay_open_first(self, x, v, v_next, dt, u_dot, x_dot):
        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        v_new = x_dot(x, v)
        return v_new
    
    def RK_closed_second(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        x_temp = x.copy()
        v_temp = v.copy()

        k1 = dt*x_dot(x, v)
        l1 = dt*u_dot(x, v)

        k2 = dt/2*x_dot((x + k1)%self.road_len, v + l1)
        l2 = dt/2*u_dot((x + k1)%self.road_len, v + l1)

        k3 = dt/2*x_dot((x + k2)%self.road_len, v + l2)
        l3 = dt/2*u_dot((x + k2)%self.road_len, v + l2)

        k4 = dt*x_dot((x + k3)%self.road_len, v + l3)
        l4 = dt*u_dot((x + k3)%self.road_len, v + l3)

        dx = 1/6*(k1 + 2*(k2 + k3) + k4)
        dv = 1/6*(l1 + 2*(l2 + l3) + l4)

        x_new = (x + dx)%self.road_len
        v_new = v + dv

        return x_new, v_new
    
    def RK_open_second(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        threshold = 1e-9

        leader_index = 0
        multi_leader_model = u_dot.__name__.startswith(('u_dot_midm', 'u_dot_mfvdm'))
        if multi_leader_model:
            leader_index = u_dot.__self__.m - 1

        leader_v_next = v_next[leader_index]
        a_leader = (leader_v_next - v[leader_index])/dt
        leader_indices = slice(0, leader_index + 1)

        def apply_leader_velocity(v_stage, stage_dt):
            """
            Description:
                Run the apply leader velocity calculation.
            
            Inputs:
                v_stage: Input value used by this function for v stage.
                stage_dt: Input value used by this function for stage dt.
            
            Outputs:
                None. The function is used for its side effects.
            """
            if multi_leader_model:
                v_stage[leader_indices] = v[leader_indices] + a_leader*stage_dt
            else:
                v_stage[leader_index] = v[leader_index] + a_leader*stage_dt

        def reconstruct_virtual_leaders(x_stage, v_stage):
            """
            Description:
                Run the reconstruct virtual leaders calculation.
            
            Inputs:
                x_stage: Input value used by this function for x stage.
                v_stage: Input value used by this function for v stage.
            
            Outputs:
                None. The function is used for its side effects.
            """
            if not multi_leader_model:
                return

            first_follower = leader_index + 1
            gap = x_stage[leader_index] - x_stage[first_follower]

            if u_dot.__name__.startswith('u_dot_midm'):
                relvel = v_stage[leader_index] - v_stage[first_follower]
                for i in range(leader_index):
                    multiplier = leader_index - i
                    x_stage[i] = x_stage[leader_index] + multiplier*gap
                    v_stage[i] = v_stage[leader_index] + multiplier*relvel
            elif u_dot.__name__.startswith('u_dot_mfvdm'):
                for i in range(leader_index):
                    multiplier = leader_index - i
                    x_stage[i] = x_stage[leader_index] + multiplier*gap
                    v_stage[i] = v_stage[leader_index]

        x_base = x.copy()
        v_base = v.copy()
        reconstruct_virtual_leaders(x_base, v_base)

        k1 = x_dot(x_base, v_base)
        l1 = u_dot(x_base, v_base)

        v_temp = np.maximum(v_base + l1/2*dt, v_base*0)
        apply_leader_velocity(v_temp, dt/2)

        x_temp = x_base + k1*dt/2
        reconstruct_virtual_leaders(x_temp, v_temp)
        k2 = x_dot(x_temp, v_temp)
        l2 = u_dot(x_temp, v_temp)

        v_temp = np.maximum(v_base + l2/2*dt, v_base*0)
        apply_leader_velocity(v_temp, dt/2)

        x_temp = x_base + k2*dt/2
        reconstruct_virtual_leaders(x_temp, v_temp)
        k3 = x_dot(x_temp, v_temp)
        l3 = u_dot(x_temp, v_temp)

        v_temp = np.maximum(v_base + l3*dt, v_base*0)
        apply_leader_velocity(v_temp, dt)

        x_temp = x_base + k3*dt
        reconstruct_virtual_leaders(x_temp, v_temp)
        k4 = x_dot(x_temp, v_temp)
        l4 = u_dot(x_temp, v_temp)

        v_new = v + dt/6*(l1 + 2*l2 + 2*l3 + l4)

        v_new = np.maximum(v_new, v_new*0)
        x_new = x + np.maximum(dt/6*(k1 + 2*k2 + 2*k3 + k4), x*0)

        x_new = np.where(np.abs(x_new - x) > threshold, x_new, x)
        v_new = np.where(np.abs(v_new - v) > threshold, v_new, v)

        return x_new, v_new
    
    def RK8_open_second(self, x, v, v_next, dt, u_dot, x_dot):
        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        threshold = 1e-9  # Convergence threshold

        v_next = v_next[0]
        a_leader = (v_next - v[0]) / dt

        # RK8 coefficients (Dormand-Prince 8(7))
        c = np.array([0, 1/18, 1/12, 1/8, 5/16, 3/8, 59/400, 93/200, 5490023248/9719169821, 13/20, 1201146811/1299019798, 1, 1])
        
        a = np.array([
            [],  # k1
            [1/18],  # k2
            [1/48, 1/16],  # k3
            [1/32, 0, 3/32],  # k4
            [5/16, 0, -75/64, 75/64],  # k5
            [3/80, 0, 0, 3/16, 3/20],  # k6
            [29443841/614563906, 0, 0, 77736538/692538347, -28693883/1125000000, 23124283/1800000000],  # k7
            [16016141/946692911, 0, 0, 61564180/158732637, 22789713/633445777, 545815736/2771057229, -180193667/1043307555],  # k8
            [39632708/573591083, 0, 0, -433636366/683701615, -421739975/2616292301, 100302831/723423059, 790204164/839813087, 800635310/3783071287],  # k9
            [246121993/1340847787, 0, 0, -37695042795/15268766246, -309121744/1061227803, -12992083/490766935, 6005943493/2108947869, 393006217/1396673457, 123872331/1001029789],  # k10
            [-1028468189/846180014, 0, 0, 8478235783/508512852, 1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560, 15336726248/1032824649, -45442868181/3398467696, 3065993473/597172653],  # k11
            [185892177/718116043, 0, 0, -3185094517/667107341, -477755414/1098053517, -703635378/230739211, 5731566787/1027545527, 5232866602/850066563, -4093664535/808688257, 3962137247/1805957418, 65686358/487910083],  # k12
            [403863854/491063109, 0, 0, -5068492393/434740067, -411421997/543043805, 652783627/914296604, 11173962825/1121059268, -13158990841/618572847, 3936647629/1978049680, -160528059/685178525, 248638103/1413531060, 0]  # k13
        ])

        b = np.array([14005451/335480064, 0, 0, 0, 0, -59238493/1068277825, 181606767/758867731, 561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4])

        # Compute k-values
        k = np.zeros((13, len(x)))
        l = np.zeros((13, len(v)))

        for i in range(13):
            x_temp = x + dt * np.sum([a[i][j] * k[j] for j in range(i)], axis=0)
            v_temp = v + dt * np.sum([a[i][j] * l[j] for j in range(i)], axis=0)
            v_temp = np.maximum(v_temp, v * 0)
            if i > 0:
                v_temp[0] += a_leader * c[i] * dt

            k[i] = x_dot(x_temp, v_temp)
            l[i] = u_dot(x_temp, v_temp)

        # Update position and velocity
        x_new = x + dt * np.sum([b[i] * k[i] for i in range(13)], axis=0)
        v_new = v + dt * np.sum([b[i] * l[i] for i in range(13)], axis=0)

        x_new = np.maximum(x_new, x * 0)
        v_new = np.maximum(v_new, v * 0)

        # Apply thresholding
        x_new = np.where(np.abs(x_new - x) > threshold, x_new, x)
        v_new = np.where(np.abs(v_new - v) > threshold, v_new, v)

        return x_new, v_new
    
    def RK_closed_first(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        k1 = x_dot(x, v)*dt/2
        k2 = x_dot((x + k1)%self.road_len, v)*dt/2
        k3 = x_dot((x + k2)%self.road_len, v)*dt
        k4 = x_dot((x + k3)%self.road_len, v)*dt

        dx = 1/6*(k1 + 2*(k2 + k3) + k4)
        dv = dx/dt
        x_new = (x + dx)%self.road_len
        v_new = v + dv

        return x_new, v_new
    
    def RK_open_first(self, x, v, v_next, dt, u_dot, x_dot):

        """
        Description:
            Advance the vehicle state by one time step using this numerical solver.
        
        Inputs:
            self: Current class instance.
            x: Vehicle position array at the current step.
            v: Vehicle velocity array at the current step.
            v_next: Leader velocity value for the next time step.
            dt: Simulation time step in seconds.
            u_dot: Velocity-rate function used by the solver.
            x_dot: Position-rate function used by the solver.
        
        Outputs:
            Updated position and velocity arrays for the next simulation step.
        """
        k1 = dt*x_dot(x, v)
        x_temp = x.copy()
        x_temp[0] = x_temp[0] + v[0]*self.dt/2
        x_temp[1:] = x[1:] + k1/2 

        k2 = dt*x_dot(x_temp)
        x_temp[1:] = x[1:] + k2/2

        k3 = dt*x_dot(x_temp)
        x_temp[1:] = x[1:] + k3
        x_temp[0] = x_temp[0] + v[0]*dt/2

        k4 = dt*x_dot(x_temp)

        dx = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        dx = np.append(np.array(v[0]*self.dt), dx)

        dv = dx/dt
        x_new = x + dx
        v_new = v + dv

        return x_new, v_new
    
    def euler_second_TK(self, 
                 x: np.ndarray, 
                 v: np.ndarray, 
                 a_prev: np.ndarray, 
                 dt: float, 
                 u_dot: Callable, 
                 x_dot: Callable) -> tuple[np.ndarray]:
        """
        https://www.sciencedirect.com/science/article/pii/S0191261517308536?via%3Dihub
        Numerical method with first order approximation for a second order ode

        Parameters
        ----------
        x : np.ndarray
            position 1d array
        v : np.ndarray
            velocity 1d array
        a_prev : np.ndarray
            acceleration 1d array
        dt : float
            simulation time step
        u_dot : function
            method that dictates the acceleration
        x_dot : function
            method that dictates the velocity

        Returns
        -------
        tuple[np.ndarray]
            updated position x_new, velocities v_new and calculated acceleration a_curr
        """

        threshold = 1e-9
        # threshold = 1e-18

        a_cur = u_dot(x, v)

        v_new = v + 0.5*(a_prev + a_cur)*dt
        x_new = x + v*dt + 0.5*a_prev*dt**2

        x_new = np.where(np.abs(x_new - x) > threshold, x_new, x)
        v_new = np.where(np.abs(v_new - v) > threshold, v_new, v)

        x_new = np.where(x_new > x, x_new, x)

        # Ensure velocity is non-negative
        v_new = np.maximum(np.zeros_like(v_new), v_new)

        return x_new, v_new, a_cur
    

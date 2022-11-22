import numpy as np
import numba
from numba.experimental import jitclass
from numba import float64, int64
from numba_fluid import Fluid
from PIL import Image

_simulation_spec = [
    ('_domain_length_x', float64),
    ('_domain_length_y', float64),
    ('_nx', int64),
    ('_ny', int64),
    ('_dx', float64),
    ('_dy', float64),
    ('_obstacle_temperture', float64),
    ('_obstacle_pressure', float64),
    ('_obstacle', float64[:, :]),
    ('_u0', float64[:, :]),
    ('_v0', float64[:, :]),
    ('_p0', float64[:, :]),
    ('_t0', float64[:, :]),
    ('_c0', float64[:, :]),
]

@jitclass(_simulation_spec)
class Simulation (object):
    def __init__(self, width, height, n_points_x=41, n_points_y=41):
        # Domain
        self._domain_length_x = width
        self._domain_length_y = height
        self._nx = n_points_x
        self._ny = n_points_y
        self._dx = self._domain_length_x/(self._nx-1)
        self._dy = self._domain_length_y/(self._ny-1)

        # Obstacle parameters
        self._obstacle_temperture = 0
        self._obstacle_pressure = 0
        self._obstacle = np.ones((self._ny, self._nx))
        self._obstacle[:, 0] = 0
        self._obstacle[:, -1] = 0
        self._obstacle[0, :] = 0
        self._obstacle[-1, :] = 0

        # Initial conditions
        self._u0 = np.zeros((self._ny, self._nx))
        self._v0 = np.zeros((self._ny, self._nx))
        self._p0 = np.zeros((self._ny, self._nx))
        self._t0 = np.zeros((self._ny, self._nx))
        self._c0 = np.zeros((self._ny, self._nx))

    @property
    def grid_size(self):
        return self._ny, self._nx
    
    @property
    def domain_length(self):
        return self._domain_length_y, self._domain_length_x

    @property
    def obstacle(self):
        return self._obstacle
    
    @property
    def obstacle_temperture(self):
        return self._obstacle_temperture
    
    @obstacle_temperture.setter
    def obstacle_temperture(self, value):
        self._obstacle_temperture = value

    @property
    def obstacle_pressure(self):
        return self._obstacle_pressure
    
    @obstacle_pressure.setter
    def obstacle_pressure(self, value):
        self._obstacle_pressure = value

    @property
    def initial_conditions(self):
        return self._u0, self._v0, self._p0, self._t0, self._c0

    @property
    def initial_velocity_u(self):
        return self._u0

    @property
    def initial_velocity_v(self):
        return self._v0

    @initial_velocity_u.setter
    def initial_velocity_u(self, value):
        print('setting initial velocity u')
        if value.shape == self._u0.shape:
            self._u0 = value.copy()
        else:
            raise ValueError(
                'Initial velocity u must have the same shape as the grid')
    
    @initial_velocity_v.setter
    def initial_velocity_v(self, value):
        if value.shape == self._v0.shape:
            self._v0 = value.copy()
        else:
            raise ValueError(
                'Initial velocity v must have the same shape as the grid')
    
    def diff_x(self, field):
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 2:]-field[1:-1, 0:-2]) / (2 * self._dx)
        return diff

    def diff_y(self, field):
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[2:, 1:-1]-field[0:-2, 1:-1]) / (2 * self._dy)
        return diff

    def diff_laplace(self, field):
        ''' Calculate laplace operator '''
        laplace = np.zeros_like(field)
        laplace[1:-1, 1:-1] = (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, 0:-2]) / (self._dx**2) + \
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] +
             field[0:-2, 1:-1]) / (self._dy**2)
        return laplace

    def apply_velocity_BC(self, u, v):
        for y in range(self._ny):
            for x in range(self._nx):
                if self._obstacle[y, x] == 0:
                    u[y, x] = 0
                    v[y, x] = 0

    def apply_pressure_BC(self, p):
        ''' Apply boundary conditions for pressure '''
        for i in range(self._ny):
            for j in range(self._nx):
                if self._obstacle[i, j] == 0:
                    p[i, j] = self._obstacle_pressure

        # todo implement boundary conditions for pressure
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:,  0] = p[:,  1]
        p[-1, :] = 0

    def solve_poisson(self,b):
        ''' Solve poission equation '''
        # copied from https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/13_Step_10.ipynb
        p = np.zeros((self._ny, self._nx))  # initial guess
        for _ in range(50):
            pd = p.copy()
            for j in range(1, self._ny-1):
                for i in range(1, self._nx-1):
                    p[j, i] = (((pd[j, i+1] + pd[j, i-1]) * self._dy**2 +
                                (pd[j+1, i] + pd[j-1, i]) * self._dx**2) /
                                 (2 * (self._dx**2 + self._dy**2)) -
                                    self._dx**2 * self._dy**2 / (2 * (self._dx**2 + self._dy**2)) * b[j, i])
            self.apply_pressure_BC(p)
        return p

    def apply_temperature_BC(self, t):
        ''' Apply boundary conditions for temperature '''
        for i in range(self._ny):
            for j in range(self._nx):
                if self._obstacle[i, j] == 0:
                    t[i, j] = self._obstacle_temperture

    def apply_concentration_BC(self, c):
        ''' Apply boundary conditions for concentration '''
        for i in range(self._ny):
            for j in range(self._nx):
                if self._obstacle[i, j] == 0:
                    c[i, j] = 0

    def calculate_temperature_field(self, t, u, v, dt, fluid: Fluid):
        laplace_t = self.diff_laplace(t)
        dt_dx = self.diff_x(t)
        dt_dy = self.diff_y(t)

        du_dx = self.diff_x(u)
        du_dy = self.diff_y(u)
        dv_dx = self.diff_x(v)
        dv_dy = self.diff_y(v)

        # Dissipation
        dissipation = 2 * (du_dx**2+dv_dy**2) + (du_dy+dv_dx)**2

        # Advection
        advection = u * dt_dx + v * dt_dy

        # Diffusion
        diffusion = (fluid.thermal_conductivity * laplace_t + fluid.viscosity *
                     dissipation)/(fluid.density * fluid.specific_heat_capacity)

        # Perform time step
        t += dt * (-advection + diffusion)

        # Apply boundary conditions
        self.apply_temperature_BC(t)

    def calculate_concentration_field(self, c, u, v, dt, fluid: Fluid):
        diffusion = self.diff_laplace(c)

        advection = u * self.diff_x(c) + v * self.diff_y(c)

        # Perform time step
        c += dt * (-advection + fluid.diffusivity * diffusion)

        # Apply boundary conditions
        self.apply_concentration_BC(c)

    def calculate_velocity_field(self, u, v, p, dt, fluid: Fluid):
        laplace_u = self.diff_laplace(u)
        laplace_v = self.diff_laplace(v)

        u_adv = u * self.diff_x(u) + v * self.diff_y(u)
        v_adv = u * self.diff_x(v) + v * self.diff_y(v)

        u_diff = fluid.viscosity * laplace_u
        v_diff = fluid.viscosity * laplace_v

        # Perform tentative velocity step
        u += dt * (-u_adv + u_diff)
        v += dt * (-v_adv + v_diff)

        # Apply boundary conditions
        self.apply_velocity_BC(u, v)

        # Calculate pressure gradient
        rhs = fluid.density/dt * (self.diff_x(u) + self.diff_y(v))
        p[:,:] = self.solve_poisson(rhs)[:,:] # todo: check if this is correct

        # Project velocity field to make it divergence free
        u -= dt/fluid.density * self.diff_x(p)
        v -= dt/fluid.density * self.diff_y(p)

        # Apply boundary conditions
        self.apply_velocity_BC(u, v)


    def simulate(self, fluid: Fluid, dt: float, steps: int):
        ''' Simulate fluid flow '''
        # Initialize fields
        u = self._u0.copy()
        v = self._v0.copy()
        p = self._p0.copy()
        t = self._t0.copy()
        c = self._c0.copy()

        # Apply boundary conditions
        self.apply_velocity_BC(u, v)
        self.apply_pressure_BC(p)
        self.apply_temperature_BC(t)
        self.apply_concentration_BC(c)

        for i in range(steps):
            self.calculate_velocity_field(u, v, p, dt, fluid)
            self.calculate_temperature_field(t, u, v, dt, fluid)
            self.calculate_concentration_field(c, u, v, dt, fluid)
        
        return u, v, p, t, c

    @staticmethod
    def save_results(u, v, p, t, c, path):
        pass

def from_picture(path: str,
                width: float, height: float,
                obstacle_temperature: float, obstacle_pressure: float):
    ''' Create a fluid domain from a picture '''
    
    # Load image
    img = Image.open(path)
    width_px, height_px = img.size

    # Remove alpha channel
    png = img.convert('RGBA')
    background = Image.new('RGBA', png.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, png)

    # Convert to grayscale
    grayscale = alpha_composite.convert('L')

    # Creating simulation with ny = height_px and nx = width_px
    s = Simulation(width, height, width_px, height_px)

    # Set obstacle from grayscale image
    for y in range(height_px):
        for x in range(width_px):
            if grayscale.getpixel((x, y)) <= 128:
                s._obstacle[y, x] = 0
                s._t0[y, x] = obstacle_temperature
                s._p0[y, x] = obstacle_pressure
                s._u0[y, x] = 0
                s._v0[y, x] = 0
                s._c0[y, x] = 0
    return s
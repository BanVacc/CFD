from distutils.debug import DEBUG
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm
import pandas as pd
import json

ATM = 101325
DEBUG = True
# todo: dt from Courant condition
# ? obsticle properties
# fix stability (nm)
# fix heat transfer (check)
# fix pressure (check)
# pressure BC


class FluidSimulation:
    def __init__(self,
                 length_x=1, length_y=1,
                 denisty=997,
                 viscosity=8.9E-4,
                 specific_heat_capacity=4181,
                 thermal_conductivity=0.607,
                 initial_temperature_K=273.15+20,
                 n_points_x=41, n_points_y=41, dt=1E-5,):

        # physical properties of fluid
        self.rho = denisty  # density [kg/m^3]
        self.mu = viscosity  # viscosity [kg/(m*s)],[Pa*s]
        self.nu = viscosity / denisty  # kinematic viscosity [m^2/s]
        self.cp = specific_heat_capacity  # specific heat capacity, J/(kg*K)
        self.k = thermal_conductivity  # thermal conductivity , W/(m*K)
        self.initial_temperature = initial_temperature_K  # initial temperature, K

        # grid propeпrties
        self.domain_size_x = length_x  # length of domain in x direction [m]
        self.domain_size_y = length_y  # length of domain in y direction [m]
        self.n_points_x = n_points_x  # number of points in x direction
        self.n_points_y = n_points_y  # number of points in y direction
        self.grid_size = (self.n_points_y, self.n_points_x)

        # simulation properties
        self.n_pressure_iterations = 50  # iterations for pressure correction
        self.stability_factor = 0.5  # less than 1
        # grid spacing in x direction [m]
        self.dx = length_x / (self.n_points_x - 1)
        # grid spacing in y direction [m]
        self.dy = length_y / (self.n_points_y - 1)
        self.dt = dt  # time step [s]

        # initial conditions for velocity
        self.u_initial = np.zeros(self.grid_size)
        self.v_initial = np.zeros(self.grid_size)
        self.t_initial = np.zeros(self.grid_size)
        self.t_initial.fill(self.initial_temperature)
        self.p_initial = np.zeros(self.grid_size)

        # box obstacle mask
        self.obstacle_mask = np.ones(self.grid_size)  # 0 - obstacle, 1 - fluid
        self.obstacle_mask[0, :] = 0
        self.obstacle_mask[-1, :] = 0
        self.obstacle_mask[:, 0] = 0
        self.obstacle_mask[:, -1] = 0

        # obstacle temperature and pressure
        self.obstacle_temperature = self.initial_temperature
        self.obstacle_pressure = 0

        if DEBUG:
            #! debug only
            print("n_points_x", self.n_points_x)
            print("n_points_y", self.n_points_y)
            print('length_x', length_x)
            print('length_y', length_y)
            print('dx = ', self.dx)
            print('dy = ', self.dy)
            print('dt = ', self.dt)
            print('is stable = ', self.is_stable())

    @staticmethod
    def load_from_json(json_file):
        ''' Load simulation parameters from json file '''
        with open(json_file, 'r') as f:
            json = json.load(f)
            length_x = json['length_x']
            length_y = json['length_y']
            n_points_x = json['n_points_x']
            n_points_y = json['n_points_y']
            obstacle_mask = np.array(json['obstacle_mask'])
            u_initial = np.array(json['u'])
            v_initial = np.array(json['v'])
            p_initial = np.array(json['p'])
            t_initial = np.array(json['t'])
            obstacle_temperature = json['obstacle_temperature']
            obstacle_pressure = json['obstacle_pressure']
            density = json['density']
            viscosity = json['viscosity']
            specific_heat_capacity = json['specific_heat_capacity']
            thermal_conductivity = json['thermal_conductivity']
            initial_temperature_K = json['initial_temperature_K']
            dt = json['dt']

            simulation =  FluidSimulation(length_x=length_x, length_y=length_y, 
            denisty=density, viscosity=viscosity, specific_heat_capacity=specific_heat_capacity, thermal_conductivity=thermal_conductivity, 
            initial_temperature_K=initial_temperature_K, n_points_x=n_points_x, n_points_y=n_points_y, dt=dt)
            simulation.set_velocity_IC(u_initial, v_initial)
            simulation.set_pressure_IC(p_initial)
            simulation.set_temperature_IC(t_initial)
            simulation.init_obstacle(obstacle_mask, obstacle_temperature, obstacle_pressure)
            return simulation

    def init_obstacle(self, mask, temperature: float, pressure: float):
        ''' Initialize obstacle mask by picture '''
        self.obstacle_temperature = temperature
        self.obstacle_pressure = pressure

        self.obstacle_mask[-1, :] = 0
        self.obstacle_mask[:, -1] = 0
        self.obstacle_mask[:, 0] = 0
        self.obstacle_mask[0, :] = 0

        #self.obstacle_mask = mask.copy()

        # mid_y = self.n_points_y // 2
        # mid_x = self.n_points_x // 2

        # self.obstacle_mask[:, mid_x-5:mid_x+5] = 0
        # self.obstacle_mask[mid_y-3:mid_y+3, :] = 1
        # #! hardcode cavity
        #self.obstacle_mask[0:int(self.n_points_y*1/4), 1:-1] = 0.0
        #self.obstacle_mask[int(self.n_points_y*3/4):, 1:-1] = 0.0
        #self.obstacle_mask[:-20,5:-10] = 0
        #! hardcode circle initial condition
        x_center = (self.n_points_x / 2) * self.dx
        y_center = (self.n_points_y / 2) * self.dy
        radius = ((self.n_points_x / 8) * self.dx)**2 + \
            ((self.n_points_y/8)*self.dy)**2
        radius = radius**0.5

        for i in range(self.n_points_y):
            for j in range(self.n_points_x):
                x = i * self.dx
                y = j * self.dy
                if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
                    # np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
                    self.obstacle_mask[i, j] = 0
                    # -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
                    self.obstacle_mask[i, j] = 0

    def set_velocity_IC(self, u_initial, v_initial):
        ''' Initialize velocity '''
        self.u_initial = u_initial.copy()
        self.v_initial = v_initial.copy()

    def set_temperature_IC(self, t_initial):
        ''' Initialize temperature '''
        self.t_initial = t_initial.copy()

    def set_pressure_IC(self, p_initial):
        ''' Initialize pressure '''
        self.p_initial = p_initial.copy()

    def diff_x(self, field):
        ''' Calculate central difference in x direction '''
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 2:]-field[1:-1, 0:-2]) / (2 * self.dx)
        return diff

    def diff_y(self, field):
        ''' Calculate central difference in y direction '''
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[2:, 1:-1]-field[0:-2, 1:-1]) / (2 * self.dy)
        return diff

    def diff_laplace(self, field):
        ''' Calculate laplace operator '''
        laplace = np.zeros_like(field)
        laplace[1:-1, 1:-1] = (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, 0:-2]) / (self.dx**2) + \
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] +
             field[0:-2, 1:-1]) / (self.dy**2)
        return laplace

    def is_stable(self):
        ''' Check stability condition  CFL'''
        return self.dt <= (self.dx*self.dy) / (2 * self.nu)

    def draw_plot(self, u, v, p, t):
        x = np.linspace(0, self.domain_size_x, self.n_points_x)
        y = np.linspace(0, self.domain_size_y, self.n_points_y)
        X, Y = np.meshgrid(x, y)

        plt.style.use('dark_background')

        # Obstacle mask
        obstacle = self.obstacle_mask.copy()

        fig_obstacle, ax_obstacle = plt.subplots(dpi=100)
        ax_obstacle.set_title('Obstacle mask')
        ax_obstacle.set_xlabel('$x,[m]$')
        ax_obstacle.set_ylabel('$y,[m]$')
        ax_obstacle.set_xlim(0, self.domain_size_x)
        ax_obstacle.set_ylim(0, self.domain_size_y)
        ax_obstacle.set_aspect('equal')
        ax_obstacle.pcolor(X, Y, obstacle,
                           cmap='gray', edgecolors='c', linewidths=0.1)

        def streamplot_with_background(u_component, v_component, background, title, colorbar_label, color_map='viridis'):
            fig, ax = plt.subplots(dpi=100)
            ax.set_title(title)
            ax.set_xlabel('$x,[m]$')
            ax.set_ylabel('$y,[m]$')
            ax.set_xlim(0, self.domain_size_x)
            ax.set_ylim(0, self.domain_size_y)
            ax.streamplot(X, Y, u_component, v_component,
                          color="black", density=1, linewidth=1, arrowsize=1)
            contour = ax.contourf(
                X, Y, background, cmap=color_map, antialiased=True, levels=100)
            colorbar = fig.colorbar(contour, ax=ax, label=colorbar_label)

        # Velocity,pressure and temperature initial conditions
        u_initial = self.u_initial.copy()
        v_initial = self.v_initial.copy()
        self.apply_velocity_BC(u_initial, v_initial)

        p_initial = self.p_initial.copy()
        self.apply_pressure_BC(p_initial)

        t_initial = self.t_initial.copy()
        self.apply_temperature_boundary_conditions(t_initial)

        streamplot_with_background(u_initial, v_initial, p_initial,
                                   '$\mathbf{u}$ and $p$ initial conditions',
                                   colorbar_label='Pressure $[Pa]$',
                                   color_map='Wistia')
        streamplot_with_background(u_initial, v_initial, t_initial,
                                   '$\mathbf{u}$ and $t$ initial conditions',
                                   colorbar_label='Temperature $[K]$',
                                   color_map='jet')

        # Velocity, pressure and temperature final conditions
        streamplot_with_background(
            u, v, p, '$\mathbf{u}$ and $p$ final conditions',
            colorbar_label='Pressure $[Pa]$',
            color_map='Wistia')
        streamplot_with_background(
            u, v, t, '$\mathbf{u}$ and $t$ final conditions',
            colorbar_label='Temperature $[K]$',
            color_map='jet')

    def apply_velocity_BC(self, u, v):
        ''' Apply boundary conditions for velocity '''
        # set velocity on obstacle to zero
        u[:, :] *= self.obstacle_mask[:, :]
        v[:, :] *= self.obstacle_mask[:, :]

    def apply_pressure_BC(self, p):
        ''' Apply boundary conditions for pressure '''
        for i in range(self.n_points_y):
            for j in range(self.n_points_x):
                if self.obstacle_mask[i, j] == 0:
                    p[i, j] = self.obstacle_pressure
        
        # todo implement boundary conditions for pressure
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:,  0] = p[:,  1]
        p[-1, :] = 0

    def calculate_pressure_field(self, p, b):
        ''' Solve poission equation '''
        # copied from https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/13_Step_10.ipynb
        p = np.zeros((self.n_points_y, self.n_points_x))  # initial guess
        for _ in range(self.n_pressure_iterations):
            pd = p.copy()
            p[1:-1, 1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * self.dy**2 +
                              (pd[2:, 1:-1] + pd[:-2, 1:-1]) * self.dx**2 -
                              b[1:-1, 1:-1] * self.dx**2 * self.dy**2) /
                             (2 * (self.dx**2 + self.dy**2)))
            self.apply_pressure_BC(p)
        return p

    def apply_temperature_boundary_conditions(self, t):
        ''' Apply boundary conditions for temperature '''
        for i in range(self.n_points_y):
            for j in range(self.n_points_x):
                if self.obstacle_mask[i, j] == 0:
                    t[i, j] = self.obstacle_temperature

    def simulate(self, n_iterations:int, filename:str|None = None):
        ''' Simulate flow '''

        # check stability condition
        if not self.is_stable():
            raise Exception("Stability condition is not satisfied")
        else:
            print("Stability condition is satisfied")

        # initialisation of initial conditions
        u = self.u_initial.copy()
        v = self.v_initial.copy()
        p = self.p_initial.copy()
        t = self.t_initial.copy()

        # apply boundary conditions
        self.apply_velocity_BC(u, v)
        self.apply_pressure_BC(p)
        self.apply_temperature_boundary_conditions(t)

        # main simulation loop
        for iteration in tqdm(range(n_iterations)):
            du_dx = self.diff_x(u)
            du_dy = self.diff_y(u)
            dv_dx = self.diff_x(v)
            dv_dy = self.diff_y(v)
            laplace_u = self.diff_laplace(u)
            laplace_v = self.diff_laplace(v)

            # Advection term : (u ⋅ ∇) u
            u_advection = u*du_dx + v*du_dy  # advection term for u
            v_advection = u*dv_dx + v*dv_dy
            # Diffusion term : ν ∇²u
            u_diffusion = self.nu * laplace_u
            v_diffusion = self.nu * laplace_v
            # perform a tentative velocity step by solving momentum equation with no pressure term
            u = u + self.dt * (-u_advection + u_diffusion)
            v = v + self.dt * (-v_advection + v_diffusion)

            # apply boundary conditions for tentative velocity
            self.apply_velocity_BC(u, v)

            du_tent_dx = self.diff_x(u)
            dv_tent_dy = self.diff_y(v)

            # calculate pressure field
            rhs = self.rho/self.dt * (du_tent_dx+dv_tent_dy)
            p = self.calculate_pressure_field(p, rhs)

            u_pressure_gradient = self.diff_x(p)
            v_pressure_gradient = self.diff_y(p)

            # correct velocity field to make it divergence free by subtracting pressure gradient
            u = u - self.dt/self.rho * u_pressure_gradient
            v = v - self.dt/self.rho * v_pressure_gradient

            # apply boundary conditions for corrected velocity
            self.apply_velocity_BC(u, v)

            # solve temperature equation
            laplace_t = self.diff_laplace(t)
            dt_dx = self.diff_x(t)
            dt_dy = self.diff_y(t)

            du_dx = self.diff_x(u)
            du_dy = self.diff_y(u)
            dv_dx = self.diff_x(v)
            dv_dy = self.diff_y(v)

            # Dissipation term : 2∇u⋅∇u + (∇u)²
            phi = 2 * (du_dx**2+dv_dy**2) + (du_dy+dv_dx)**2

            # thermal advection term : u ⋅ ∇T
            t_advection = u*dt_dx + v*dt_dy

            # thermal diffusion term : 1/(ρCp) (k∇²T + μΦ)
            t_diffusion = 1/(self.rho*self.cp) * \
                (self.k * laplace_t + self.mu * phi)

            # perform temperature step by solving energy equation
            t += self.dt * (-t_advection + t_diffusion)

            # apply boundary conditions for temperature field
            self.apply_temperature_boundary_conditions(t)

        # save results
        if filename is not None:
            with open(f'results/{filename}.json', 'w') as f:
                data = {
                'length_x': self.domain_size_x,
                'length_y': self.domain_size_y,
                'n_points_x': self.n_points_x,
                'n_points_y': self.n_points_y,
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt,

                'density': self.rho,
                'viscosity': self.mu,
                'kinematic_viscosity': self.nu,
                'specific_heat_capacity': self.cp,
                'thermal_conductivity': self.k,
                'initial_temperature': self.initial_temperature,

                'obstacle_mask': self.obstacle_mask.tolist(),
                'obstacle_temperature': self.obstacle_temperature,
                'obstacle_pressure': self.obstacle_pressure,

                'n_iterations': n_iterations,

                'u_initial': self.u_initial.tolist(),
                'v_initial': self.v_initial.tolist(),
                'p_initial': self.p_initial.tolist(),
                't_initial': self.t_initial.tolist(),

                'u_final': u.tolist(),
                'v_final': v.tolist(),
                'p_final': p.tolist(),
                't_final': t.tolist()}
            
                json_string = json.dumps(data)
                f.write(json_string)

        return u, v, p, t

from distutils.debug import DEBUG
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm

DEBUG = True
# todo: dt from Courant condition
# ? obsticle properties

class FluidSimulation:
    def __init__(self,
                 length_x=1, length_y=1,
                 denisty=997, viscosity=8.9E-4,
                 specific_heat_capacity=4181,
                 thermal_conductivity=0.607,
                 initial_temperature_K=273+20,
                 n_points_x=100, n_points_y=100, dt=1E-5):

        # physical properties of fluid
        self.rho = denisty  # density [kg/m^3]
        self.mu = viscosity  # viscosity [kg/(m*s)],[Pa*s]
        self.nu = viscosity / denisty  # kinematic viscosity [m^2/s]
        self.cp = specific_heat_capacity  # specific heat capacity, J/(kg*K)
        self.k = thermal_conductivity  # thermal conductivity , W/(m*K)
        self.T0 = initial_temperature_K  # initial temperature, K

        # grid propeпrties
        self.domain_size_x = length_x  # length of domain in x direction [m]
        self.domain_size_y = length_y  # length of domain in y direction [m]
        self.n_points_x    = n_points_x  # number of points in x direction
        self.n_points_y    = n_points_y  # number of points in y direction

        # simulation properties
        self.n_pressure_iterations = 50  # iterations for pressure correction
        self.stability_factor = 0.5 # less than 1
        self.dx = length_x / (self.n_points_x - 1) # grid spacing in x direction [m]
        self.dy = length_y / (self.n_points_y - 1) # grid spacing in y direction [m]
        self.dt = dt  # time step [s]

        self.obstacle_mask = np.ones(
            (self.n_points_y, self.n_points_x))  # 1 - fluid, 0 - obstacle

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
        

    def init_obstacle_mask(self):
        ''' Initialize obstacle mask by picture '''
        # #! hardcode cavity
        self.obstacle_mask[0:int(self.n_points_y*1/4), 1:-1] = 0.0
        self.obstacle_mask[int(self.n_points_y*3/4):, 1:-1] = 0.0
        self.obstacle_mask[:-20,5:-10] = 0

        # #! hardcode circle initial condition
        # x_center = (self.n_points_x / 2) * self.dx
        # y_center = (self.n_points_y / 2) * self.dy
        # radius = ((self.n_points_x / 8) * self.dx)**2 + \
        #     ((self.n_points_y/8)*self.dy)**2
        # radius = radius**0.5

        # for i in range(self.n_points_y):
        #     for j in range(self.n_points_x):
        #         x = i * self.dx
        #         y = j * self.dy
        #         if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
        #             # np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        #             self.obstacle_mask[i, j] = 0
        #             # -np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        #             self.obstacle_mask[i, j] = 0

    def central_difference_x(self, field):
        ''' Calculate central difference in x direction '''
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 2:]-field[1:-1, 0:-2]) / (2 * self.dx)
        return diff

    def central_difference_y(self, field):
        ''' Calculate central difference in y direction '''
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[2:, 1:-1]-field[0:-2, 1:-1]) / (2 * self.dy)
        return diff

    def laplace(self, field):
        ''' Calculate laplace operator '''
        laplace = np.zeros_like(field)
        laplace[1:-1, 1:-1] = (field[1:-1, 2:] - 2 * field[1:-1, 1:-1]+field[1:-1, 0:-2]) / (self.dx**2) + \
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
        fig_obstacle, ax_obstacle = plt.subplots(dpi=100)
        ax_obstacle.set_title('Obstacle mask')
        ax_obstacle.set_xlabel('$x,[m]$')
        ax_obstacle.set_ylabel('$y,[m]$')
        ax_obstacle.set_xlim(0, self.domain_size_x)
        ax_obstacle.set_ylim(0, self.domain_size_y)
        ax_obstacle.set_aspect('equal')
        ax_obstacle.pcolor(X, Y, self.obstacle_mask,
                           cmap='gray', edgecolors='c', linewidths=0.1)
        # todo imshow or pcolor
        # ax_obstacle.imshow(self.obstacle_mask, cmap=cm.gray,
        #                    origin='lower',
        #                    extent=[0, self.domain_size_x,
        #                            0, self.domain_size_y],
        #                    interpolation='none')

        def streamplot_with_background(u_component, v_component, background, title, colorbar_label, color_map='viridis'):
            fig, ax = plt.subplots(dpi=150)
            ax.set_title(title)
            ax.set_xlabel('$x,[m]$')
            ax.set_ylabel('$y,[m]$')
            ax.set_xlim(0, self.domain_size_x)
            ax.set_ylim(0, self.domain_size_y)
            ax.streamplot(X, Y, u_component, v_component,
                          color="black", density=1, linewidth=1, arrowsize=1)
            contour = ax.contourf(
                X, Y, background, cmap=color_map, antialiased=True)
            colorbar = fig.colorbar(contour, ax=ax, label=colorbar_label)

        # Velocity,pressure and temperature initial conditions
        u_initial = np.zeros_like(u)
        v_initial = np.zeros_like(v)
        self.apply_velocity_initial_conditions(u_initial, v_initial)
        self.apply_velocity_boundary_conditions(u_initial, v_initial)

        p_initial = np.zeros_like(p)
        self.apply_pressure_initial_conditions(p_initial)
        self.apply_pressure_boundary_conditions(p_initial)

        t_initial = np.zeros_like(t)
        self.apply_temperature_initial_conditions(t_initial)
        self.apply_temperature_boundary_conditions(t_initial)

        streamplot_with_background(u_initial, v_initial, p_initial,
                                   'Velocity and pressure initial conditions',
                                   colorbar_label='Pressure $[Pa]$',
                                   color_map='Wistia')
        streamplot_with_background(u_initial, v_initial, t_initial,
                                   'Velocity and temperature initial conditions',
                                   colorbar_label='Temperature $[K]$',
                                   color_map='plasma')

        # Velocity, pressure and temperature final conditions
        streamplot_with_background(
            u, v, p, 'Velocity and pressure final conditions',
            colorbar_label='Pressure $[Pa]$',
            color_map='Wistia')
        streamplot_with_background(
            u, v, t, 'Velocity and temperature final conditions',
            colorbar_label='Temperature $[K]$',
            color_map='plasma')

    def apply_velocity_initial_conditions(self, u, v):
        ''' Apply initial conditions for velocity '''
        # set initial velocity to zero
        u[:, :] = 0.0
        v[:, :] = 0.0
        #! hardcode
        u[int(self.n_points_y*1/4):int(self.n_points_y*3/4), 1:10] = 1

    def apply_velocity_boundary_conditions(self, u, v):
        ''' Apply boundary conditions for velocity '''
        # set velocity on obstacle to zero
        u[:, :] *= self.obstacle_mask[:, :]
        v[:, :] *= self.obstacle_mask[:, :]

        # set velocity on boundary to zero
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = 0.0
        v[0, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[-1, :] = 0.0

    def apply_pressure_initial_conditions(self, p):
        ''' Apply initial conditions for pressure '''
        p[:, :] = 101325+1000  # ! hardcode

    def apply_pressure_boundary_conditions(self, p):
        ''' Apply boundary conditions for pressure '''
        # ? set pressure on obstacle to zero
        p[:, :] *= self.obstacle_mask[:, :]
        # todo
        p[:, -1] = p[:, -2]
        p[0, :] = p[1, :]
        p[:,  0] = p[:,  1]
        p[-1, :] = 101_325

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
            self.apply_pressure_boundary_conditions(p)
        return p

    def apply_temperature_initial_conditions(self, t):
        ''' Apply initial conditions for temperature '''
        t[:, :] = self.T0  # ! hardcode
        mid_point_y = self.n_points_y//2
        mid_point_x = self.n_points_x//2
        t[1, 1:-1] = self.T0 + 0.1

    def apply_temperature_boundary_conditions(self, t):
        ''' Apply boundary conditions for temperature '''
        # ? set temperature on obstacle to T0
        t[:, :] *= self.T0 * self.obstacle_mask[:, :]

        t[0, :] = self.T0
        t[:, 0] = self.T0
        t[:, -1] = self.T0
        t[-1, :] = self.T0

    def simulate(self, n_iterations):

        # check stability condition
        if not self.is_stable():
            raise Exception("Stability condition is not satisfied")
        else:
            print("Stability condition is satisfied")

        grid_shape = (self.n_points_y, self.n_points_x)

        # initialisation
        u = np.zeros(grid_shape)  # u component of velocity
        v = np.zeros(grid_shape)  # v component of velocity
        p = np.zeros(grid_shape)  # pressure field
        t = np.zeros(grid_shape)  # temperature field

        # apply initial conditions for velocity, pressure and temperature
        self.apply_velocity_initial_conditions(u, v)
        self.apply_pressure_initial_conditions(p)
        self.apply_temperature_initial_conditions(t)

        # apply boundary conditions
        self.apply_velocity_boundary_conditions(u, v)
        self.apply_pressure_boundary_conditions(p)

        # main simulation loop
        for iteration in tqdm(range(n_iterations)):
            du_dx = self.central_difference_x(u)
            du_dy = self.central_difference_y(u)
            dv_dx = self.central_difference_x(v)
            dv_dy = self.central_difference_y(v)
            laplace_u = self.laplace(u)
            laplace_v = self.laplace(v)

            # Advection term : (u ⋅ ∇) u
            u_advection = u*du_dx + v*du_dy  # advection term for u
            v_advection = u*dv_dx + v*dv_dy  # todo

            # Diffusion term : ν ∇²u
            u_diffusion = self.nu * laplace_u  # todo
            v_diffusion = self.nu * laplace_v  # todo

            # perform a tentative velocity step by solving momentum equation with no pressure term
            u = u + self.dt * (-u_advection + u_diffusion)
            v = v + self.dt * (-v_advection + v_diffusion)

            # apply boundary conditions for tentative velocity
            self.apply_velocity_boundary_conditions(u, v)

            du_tent_dx = self.central_difference_x(u)
            dv_tent_dy = self.central_difference_y(v)

            # calculate pressure field
            rhs = self.rho/self.dt * (du_tent_dx+dv_tent_dy)
            p = self.calculate_pressure_field(p, rhs)

            u_pressure_gradient = self.central_difference_x(p)
            v_pressure_gradient = self.central_difference_y(p)

            # correct velocity field to make it divergence free by subtracting pressure gradient
            u = u - self.dt/self.rho * u_pressure_gradient
            v = v - self.dt/self.rho * v_pressure_gradient

            # apply boundary conditions for corrected velocity
            self.apply_velocity_boundary_conditions(u, v)

            # solve temperature equation
            laplace_t = self.laplace(t)
            dt_dx = self.central_difference_x(t)
            dt_dy = self.central_difference_y(t)

            du_dx = self.central_difference_x(u)
            du_dy = self.central_difference_y(u)
            dv_dx = self.central_difference_x(v)
            dv_dy = self.central_difference_y(v)

            # Dissipation term : 2∇u⋅∇u + (∇u)²
            phi = 2 * (du_dx**2+dv_dy**2) + (du_dy+dv_dx)**2

            # thermal advection term : u ⋅ ∇T
            t_advection = u*dt_dx + v*dt_dy

            # thermal diffusion term : k/(ρCp) ∇²T ∇²T
            t_diffusion = 1/(self.rho*self.cp) * \
                (self.k * laplace_t + self.mu * phi)

            # perform temperature step by solving energy equation
            t = t + self.dt * (-t_advection + t_diffusion)

            # apply boundary conditions for temperature field
            self.apply_temperature_boundary_conditions(t)

        return u, v, p, t

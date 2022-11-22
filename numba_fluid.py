from numba import float64
from numba.experimental import jitclass

_fluid_spec = [
    ('density', float64),
    ('viscosity', float64),
    ('specific_heat_capacity', float64),
    ('thermal_conductivity', float64),
    ('diffusivity', float64),
]
@jitclass(_fluid_spec)
class Fluid (object):
    def __init__(self, density, viscosity, specific_heat_capacity, thermal_conductivity, diffusivity):
        self.density = density
        self.viscosity = viscosity
        self.specific_heat_capacity = specific_heat_capacity
        self.thermal_conductivity = thermal_conductivity
        self.diffusivity = diffusivity

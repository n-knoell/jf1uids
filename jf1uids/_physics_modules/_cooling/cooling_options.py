from types import NoneType
from typing import NamedTuple
from typing import Union
import jax.numpy as jnp
from jaxtyping import PyTree

from jf1uids._physics_modules._cooling._cooling_tables import schure_cooling
from jf1uids.units.unit_helpers import CodeUnits
from astropy import constants as c
from astropy import units as u

SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2
NEURAL_NET_COOLING = 3
NEURAL_NET_COOLING_WITH_DENSITY = 4

EXPLICIT_COOLING = 1
IMPLICIT_COOLING = 2

sep_in_au = 20  #upper-limit: 80, lower-limit: 5
length_temp = sep_in_au / 0.1  # before
mass_temp = 0.01 / 1  #1e-4
# velocity_temp = length_temp / 2   #40
code_length = length_temp * u.au 
code_mass = mass_temp * u.M_sun
code_velocity = ((c.G * code_mass / code_length)**0.5).to(u.km/u.s)     # code_velocity = velocity_temp * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value

class SimplePowerLawParams(NamedTuple):
    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8

class PiecewisePowerLawParams(NamedTuple):
    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = None #1e8

class CoolingNetConfig(NamedTuple):
    network_static: Union[PyTree, NoneType] = None

class CoolingNetParams(NamedTuple):
    network_params: Union[PyTree, NoneType] = None

COOLING_CURVE_TYPE = Union[SimplePowerLawParams, PiecewisePowerLawParams, CoolingNetParams]

class CoolingCurveConfig(NamedTuple):
    cooling_curve_type: int = SIMPLE_POWER_LAW
    #: In case of neural the cooling the network architecture
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()


class CoolingConfig(NamedTuple):
    cooling: bool = False
    cooling_method: int = EXPLICIT_COOLING
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()

class CoolingParams(NamedTuple):
    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02
    floor_temperature: float = 1e4
    # cooling_curve_params: COOLING_CURVE_TYPE = PiecewisePowerLawParams()
    cooling_curve_params: COOLING_CURVE_TYPE = schure_cooling(code_units)
    # cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from astropy import units as u
from jf1uids import CodeUnits
import os

save_path = os.path.join('two_winds_ext')

# loss_map = np.load(save_path+'/loss_map.npz')['arr']
# mass_list = np.load(save_path+'/mass_list.npz')['arr']
# vel_list = np.load(save_path+'/vel_list.npz')['arr']
loss_map = np.load(save_path+'/loss_map.npz')['arr']
mass_list1 = np.load(save_path+'/mass_list1.npz')['arr']
mass_list2 = np.load(save_path+'/mass_list2.npz')['arr']

print(loss_map)
print(min(loss_map))
print(max(loss_map))
print(mass_list1)
print(mass_list2)


wind_vel_inf = 50.0
mass_loss_rate1 = 1e-11 / 1.309 #2.965e-3 / 1e4 / T_orb * mass_source     #2.965e-3 / 1e6 / T_orb * mass_source
mass_loss_rate2 = 10 * mass_loss_rate1

R_forced = 0.1
mass_source = 1e-4
length_temp = 5 / R_forced
mass_temp = 40 / mass_source
velocity_temp = 2000 / wind_vel_inf
time_temp = length_temp / velocity_temp  #make time_temp as high as possible to have small code time units
code_length = length_temp * u.au 

code_mass = mass_temp * u.M_sun
code_velocity = velocity_temp * u.km / u.s
# code_time = code_length / code_velocity
code_units = CodeUnits(code_length, code_mass, code_velocity)

fig, axs = plt.subplots(1, 1, figsize=(10, 8))
# axs.axis('equal')

divider = make_axes_locatable(axs)
cax = divider.append_axes('right', size='5%', pad=0.05)
norm = LogNorm(vmin=loss_map.min(), vmax=loss_map.max())
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)
plt.colorbar(mapper, cax=cax, orientation='vertical', label='primitive state loss')

axs.scatter((mass_list1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (mass_list2 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, c=loss_map, cmap='inferno', norm=norm, s = 60, marker = "s")

axs.set_xlabel(r'${\dot M}_1 [M_\odot yr^{-1}]$')
axs.set_ylabel(r'${\dot M}_2 [M_\odot yr^{-1}]$')

log_option = True
if log_option:
    axs.set_xscale('log')
    axs.set_yscale('log')
axs.set_title('Loss landscape (density slice)')

fig.savefig(save_path+'/loss_landscape_log.png')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from astropy import units as u
from jf1uids import CodeUnits
import os
import jax.numpy as jnp

save_path = os.path.join('two_winds')

loss_list1 = np.load(save_path+'/results/loss_list1.npz')['arr']
loss_list2 = np.load(save_path+'/results/loss_list2.npz')['arr']
loss_list3 = np.load(save_path+'/results/loss_list3.npz')['arr']
xlist1 = np.load(save_path+'/results/xlist1.npz', allow_pickle=True)['arr']
xlist2 = np.load(save_path+'/results/xlist2.npz', allow_pickle=True)['arr']
xlist3 = np.load(save_path+'/results/xlist3.npz', allow_pickle=True)['arr']

loss_map = np.load(save_path+'/loss_map.npz')['arr']
mass_list1 = np.load(save_path+'/mass_list1.npz')['arr']
mass_list2 = np.load(save_path+'/mass_list2.npz')['arr']

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

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
norm = LogNorm(vmin=loss_map.min(), vmax=loss_map.max())
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.inferno)   #viridis
plt.colorbar(mapper, cax=cax, orientation='vertical', label='Density slice loss')

axs[0].scatter((mass_list1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (mass_list2 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, c=loss_map, cmap='inferno', norm=norm, s = 15, marker = "s")  #s=15

axs[0].set_xlabel(r'${\dot M}_1 [M_\odot yr^{-1}]$')
axs[0].set_ylabel(r'${\dot M}_2 [M_\odot yr^{-1}]$')
axs[0].set_title('Loss landscape')

# plot the optimization path
xlist1 = jnp.array(xlist1)
axs[0].plot((xlist1[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist1[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='blue')
axs[0].scatter(
    [(xlist1[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist1[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='blue', s = 30, label='Initial guess 1'
)

# plot the optimization path
xlist2 = jnp.array(xlist2)
axs[0].plot((xlist2[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist2[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='green')
axs[0].scatter(
    [(xlist2[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist2[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='green', s = 30, label='Initial guess 2'
)

# plot the loss function
axs[1].plot(loss_list2, label='Loss 2', color='purple')

# plot the optimization path
xlist3 = jnp.array(xlist3)
axs[0].plot((xlist3[:, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, (xlist3[:, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value, color='red')
axs[0].scatter(
    [(xlist3[0, 0] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value], [(xlist3[0, 1] * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='red', s = 30, label='Initial guess 3'
)

# plot the loss function
# axs[0].set_xlim(5e-8, 5e-5)
# axs[0].set_ylim(1e-7, 1e-4)
axs[1].plot(loss_list3, label='Loss 3', color='red')
# plot the loss function
axs[1].plot(loss_list1, label='Loss 1', color='blue')

axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Density slice loss')
axs[1].set_title('Loss convergence')
axs[1].set_yscale('log')
# axs[0].set_xscale('log')
# axs[0].set_yscale('log')
axs[1].set_xlim(0, 300)

axs[1].legend(loc='upper right')
plt.tight_layout()

# mark the true value as a red dot
axs[0].scatter(
    [(mass_loss_rate1 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    [(mass_loss_rate2 * code_units.code_mass / code_units.code_time).to(u.M_sun / u.yr).value],
    c='white', s = 80, label='True parameters', zorder = 10, edgecolors='black'
)            #marker = "."

axs[0].legend()
axs[1].legend()

fig.savefig('two_winds/params_opt.png')
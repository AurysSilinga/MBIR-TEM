# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:30 2013

@author: Jan
"""


import time
import os

import numpy as np
from numpy import pi

import shelve

import pyramid.magcreator as mc
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMFourier

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import RdBu


force_calculation = False
PHI_0 = -2067.83  # magnetic flux in T*nm²


print '\nACCESS SHELVE'
# Create / Open databank:
directory = '../../output/paper 1'
if not os.path.exists(directory):
    os.makedirs(directory)
data_shelve = shelve.open(directory + '/paper_1_shelve')

###############################################################################################
print 'CH5-1 PHASE SLICES FOURIER SPACE'

# Input parameters:
a = 0.5  # in nm
phi = pi/2
density = 100
dim = (32, 256, 256)  # in px (z, y, x)
# Create magnetic shape:
center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
radius = dim[1]/4  # in px
height = dim[0]/2  # in px

key = 'ch5-2-phase_slices_fourier'
if key in data_shelve and not force_calculation:
    print '--LOAD MAGNETIC DISTRIBUTION'
    (x_d, y_d0, y_d10, dy_d0, dy_d10, x_v, y_v0, y_v10, dy_v0, dy_v10) = data_shelve[key]
else:
    print '--CREATE MAGNETIC DISTRIBUTION'
    mag_shape = mc.Shapes.disc(dim, center, radius, height)

    print '--CREATE PHASE SLICES HOMOG. MAGN. DISC'
    # Arrays for plotting:
    x_d = []
    y_d0 = []
    y_d10 = []
    dy_d0 = []
    dy_d10 = []
    # Analytic solution:
    L = dim[1] * a  # in px/nm
    Lz = 0.5 * dim[0] * a  # in px/nm
    R = 0.25 * L  # in px/nm
    x0 = L / 2  # in px/nm

    def F_disc(x):
        coeff = - pi * Lz / (2*PHI_0) * 1E3  # in mrad -> *1000
        result = coeff * (- (x - x0) * np.sin(phi))
        result *= np.where(np.abs(x - x0) <= R, 1, (R / (x - x0)) ** 2)
        return result

    x_d.append(np.linspace(0, L, 5000))
    y_d0.append(F_disc(x_d[0]))
    y_d10.append(F_disc(x_d[0]))
    dy_d0.append(np.zeros_like(x_d[0]))
    dy_d10.append(np.zeros_like(x_d[0]))
    # Create MagData (Disc):
    mag_data_disc = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
    for i in range(5):
        print '----a =', mag_data_disc.a, 'nm', 'dim =', mag_data_disc.dim
        projector = SimpleProjector(mag_data_disc.dim)
        phase_map0 = PMFourier(mag_data_disc.a, projector, padding=0)(mag_data_disc)
        phase_map10 = PMFourier(mag_data_disc.a, projector, padding=10)(mag_data_disc)
        phase_map0.unit = 'mrad'
        phase_map10.unit = 'mrad'
        phase_map0.display_combined(density, 'Disc, a = {} nm'.format(mag_data_disc.a))
        phase_map10.display_combined(density, 'Disc, a = {} nm'.format(mag_data_disc.a))
        x_d.append(np.linspace(mag_data_disc.a * 0.5,
                               mag_data_disc.a * (mag_data_disc.dim[1]-0.5),
                               mag_data_disc.dim[1]))
        slice_pos = int(mag_data_disc.dim[1]/2)
        y_d0.append(phase_map0.phase[slice_pos, :]*1E3)  # *1E3: rad to mrad
        y_d10.append(phase_map10.phase[slice_pos, :]*1E3)  # *1E3: rad to mrad
        dy_d0.append(phase_map0.phase[slice_pos, :]*1E3 - F_disc(x_d[-1]))  # *1E3: in mrad
        dy_d10.append(phase_map10.phase[slice_pos, :]*1E3 - F_disc(x_d[-1]))  # *1E3: in mrad
        if i < 4: mag_data_disc.scale_down()

    print '--CREATE PHASE SLICES VORTEX STATE DISC'
    x_v = []
    y_v0 = []
    y_v10 = []
    dy_v0 = []
    dy_v10 = []
    # Analytic solution:
    L = dim[1] * a  # in px/nm
    Lz = 0.5 * dim[0] * a  # in px/nm
    R = 0.25 * L  # in px/nm
    x0 = L / 2  # in px/nm

    def F_vort(x):
        coeff = pi*Lz/PHI_0 * 1E3  # in mrad -> *1000
        result = coeff * np.where(np.abs(x - x0) <= R, (np.abs(x-x0)-R), 0)
        return result

    x_v.append(np.linspace(0, L, 5000))
    y_v0.append(F_vort(x_v[0]))
    y_v10.append(F_vort(x_v[0]))
    dy_v0.append(np.zeros_like(x_v[0]))
    dy_v10.append(np.zeros_like(x_v[0]))
    # Create MagData (Vortex):
    mag_data_vort = MagData(a, mc.create_mag_dist_vortex(mag_shape))
    for i in range(5):
        print '----a =', mag_data_vort.a, 'nm', 'dim =', mag_data_vort.dim
        projector = SimpleProjector(mag_data_vort.dim)
        phase_map0 = PMFourier(mag_data_vort.a, projector, padding=0)(mag_data_vort)
        phase_map10 = PMFourier(mag_data_vort.a, projector, padding=10)(mag_data_vort)
        phase_map0.unit = 'mrad'
        phase_map10.unit = 'mrad'
        print np.mean(phase_map0.phase)
        phase_map0 -= phase_map0.phase[0, 0]
        phase_map10 -= phase_map10.phase[0, 0]
        phase_map0.display_combined(density, 'Vortex, a = {} nm'.format(mag_data_vort.a))
        phase_map10.display_combined(density, 'Vortex, a = {} nm'.format(mag_data_vort.a))
        x_v.append(np.linspace(mag_data_vort.a * 0.5,
                               mag_data_vort.a * (mag_data_vort.dim[1]-0.5),
                               mag_data_vort.dim[1]))
        slice_pos = int(mag_data_vort.dim[1]/2)
        y_v0.append(phase_map0.phase[slice_pos, :]*1E3)  # *1E3: rad to mrad
        y_v10.append(phase_map10.phase[slice_pos, :]*1E3)  # *1E3: rad to mrad
        dy_v0.append(phase_map0.phase[slice_pos, :]*1E3 - F_vort(x_v[-1]))  # *1E3: in mrad
        dy_v10.append(phase_map10.phase[slice_pos, :]*1E3 - F_vort(x_v[-1]))  # *1E3: in mrad
        if i < 4: mag_data_vort.scale_down()

    # Shelve x, y and dy:
    print '--SAVE PHASE SLICES'
    data_shelve[key] = (x_d, y_d0, y_d10, dy_d0, dy_d10, x_v, y_v0, y_v10, dy_v0, dy_v10)

# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Central phase slices (padding = 0)', fontsize=20)

print '--PLOT/SAVE PHASE SLICES HOMOG. MAGN. DISC PADDING = 0'
# Plot phase slices:
axes[0].plot(x_d[0], y_d0[0], '-k', linewidth=1.5, label='analytic')
axes[0].plot(x_d[1], y_d0[1], '-r', linewidth=1.5, label='0.5 nm')
axes[0].plot(x_d[2], y_d0[2], '-m', linewidth=1.5, label='1 nm')
axes[0].plot(x_d[3], y_d0[3], '-y', linewidth=1.5, label='2 nm')
axes[0].plot(x_d[4], y_d0[4], '-g', linewidth=1.5, label='4 nm')
axes[0].plot(x_d[5], y_d0[5], '-c', linewidth=1.5, label='8 nm')
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].set_title('Homog. magn. disc', fontsize=18)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('phase [mrad]', fontsize=15)
axes[0].set_xlim(0, 128)
axes[0].set_ylim(-220, 220)
#    # Plot Zoombox and Arrow:
#    zoom = (23.5, 160, 15, 40)
#    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
#    axes[0].add_patch(rect)
#    axes[0].arrow(zoom[0]+zoom[2], zoom[1]+zoom[3]/2, 36, 0, length_includes_head=True,
#              head_width=10, head_length=4, fc='k', ec='k')
#    # Plot zoom inset:
#    ins_axis_d = plt.axes([0.33, 0.57, 0.14, 0.3])
#    ins_axis_d.plot(x_d[0], y_d0[0], '-k', linewidth=1.5, label='analytic')
#    ins_axis_d.plot(x_d[1], y_d0[1], '-r', linewidth=1.5, label='0.5 nm')
#    ins_axis_d.plot(x_d[2], y_d0[2], '-m', linewidth=1.5, label='1 nm')
#    ins_axis_d.plot(x_d[3], y_d0[3], '-y', linewidth=1.5, label='2 nm')
#    ins_axis_d.plot(x_d[4], y_d0[4], '-g', linewidth=1.5, label='4 nm')
#    ins_axis_d.plot(x_d[5], y_d0[5], '-c', linewidth=1.5, label='8 nm')
#    ins_axis_d.tick_params(axis='both', which='major', labelsize=14)
#    ins_axis_d.set_xlim(zoom[0], zoom[0]+zoom[2])
#    ins_axis_d.set_ylim(zoom[1], zoom[1]+zoom[3])
#    ins_axis_d.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
#    ins_axis_d.yaxis.set_major_locator(MaxNLocator(nbins=3))

print '--PLOT/SAVE PHASE SLICES VORTEX STATE DISC PADDING = 0'
# Plot phase slices:
axes[1].plot(x_v[0], y_v0[0], '-k', linewidth=1.5, label='analytic')
axes[1].plot(x_v[1], y_v0[1], '-r', linewidth=1.5, label='0.5 nm')
axes[1].plot(x_v[2], y_v0[2], '-m', linewidth=1.5, label='1 nm')
axes[1].plot(x_v[3], y_v0[3], '-y', linewidth=1.5, label='2 nm')
axes[1].plot(x_v[4], y_v0[4], '-g', linewidth=1.5, label='4 nm')
axes[1].plot(x_v[5], y_v0[5], '-c', linewidth=1.5, label='8 nm')
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].set_title('Vortex state disc', fontsize=18)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('phase [mrad]', fontsize=15)
axes[1].set_xlim(0, 128)
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
axes[1].legend()
#    # Plot Zoombox and Arrow:
#    zoom = (59, 340, 10, 55)
#    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
#    axes[1].add_patch(rect)
#    axes[1].arrow(zoom[0]+zoom[2]/2, zoom[1], 0, -193, length_includes_head=True,
#              head_width=2, head_length=20, fc='k', ec='k')
#    # Plot zoom inset:
#    ins_axis_v = plt.axes([0.695, 0.15, 0.075, 0.3])
#    ins_axis_v.plot(x_v[0], y_v0[0], '-k', linewidth=1.5, label='analytic')
#    ins_axis_v.plot(x_v[1], y_v0[1], '-r', linewidth=1.5, label='0.5 nm')
#    ins_axis_v.plot(x_v[2], y_v0[2], '-m', linewidth=1.5, label='1 nm')
#    ins_axis_v.plot(x_v[3], y_v0[3], '-y', linewidth=1.5, label='2 nm')
#    ins_axis_v.plot(x_v[4], y_v0[4], '-g', linewidth=1.5, label='4 nm')
#    ins_axis_v.plot(x_v[5], y_v0[5], '-c', linewidth=1.5, label='8 nm')
#    ins_axis_v.tick_params(axis='both', which='major', labelsize=14)
#    ins_axis_v.set_xlim(zoom[0], zoom[0]+zoom[2])
#    ins_axis_v.set_ylim(zoom[1], zoom[1]+zoom[3])
#    ins_axis_v.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
#    ins_axis_v.yaxis.set_major_locator(MaxNLocator(nbins=4))

plt.show()
plt.figtext(0.15, 0.13, 'a)', fontsize=30)
plt.figtext(0.57, 0.13, 'b)', fontsize=30)
plt.savefig(directory + '/ch5-1-phase_slice_padding_0.png', bbox_inches='tight')

# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Central phase slices (padding = 10)', fontsize=20)

print '--PLOT/SAVE PHASE SLICES HOMOG. MAGN. DISC PADDING = 10'
# Plot phase slices:
axes[0].plot(x_d[0], y_d10[0], '-k', linewidth=1.5, label='analytic')
axes[0].plot(x_d[1], y_d10[1], '-r', linewidth=1.5, label='0.5 nm')
axes[0].plot(x_d[2], y_d10[2], '-m', linewidth=1.5, label='1 nm')
axes[0].plot(x_d[3], y_d10[3], '-y', linewidth=1.5, label='2 nm')
axes[0].plot(x_d[4], y_d10[4], '-g', linewidth=1.5, label='4 nm')
axes[0].plot(x_d[5], y_d10[5], '-c', linewidth=1.5, label='8 nm')
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].set_title('Homog. magn. disc', fontsize=18)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('phase [mrad]', fontsize=15)
axes[0].set_xlim(0, 128)
axes[0].set_ylim(-220, 220)
#    # Plot Zoombox and Arrow:
#    zoom = (23.5, 160, 15, 40)
#    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
#    axes[0].add_patch(rect)
#    axes[0].arrow(zoom[0]+zoom[2], zoom[1]+zoom[3]/2, 36, 0, length_includes_head=True,
#              head_width=10, head_length=4, fc='k', ec='k')
#    # Plot zoom inset:
#    ins_axis_d = plt.axes([0.33, 0.57, 0.14, 0.3])
#    ins_axis_d.plot(x_d[0], y_d10[0], '-k', linewidth=1.5, label='analytic')
#    ins_axis_d.plot(x_d[1], y_d10[1], '-r', linewidth=1.5, label='0.5 nm')
#    ins_axis_d.plot(x_d[2], y_d10[2], '-m', linewidth=1.5, label='1 nm')
#    ins_axis_d.plot(x_d[3], y_d10[3], '-y', linewidth=1.5, label='2 nm')
#    ins_axis_d.plot(x_d[4], y_d10[4], '-g', linewidth=1.5, label='4 nm')
#    ins_axis_d.plot(x_d[5], y_d10[5], '-c', linewidth=1.5, label='8 nm')
#    ins_axis_d.tick_params(axis='both', which='major', labelsize=14)
#    ins_axis_d.set_xlim(zoom[0], zoom[0]+zoom[2])
#    ins_axis_d.set_ylim(zoom[1], zoom[1]+zoom[3])
#    ins_axis_d.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
#    ins_axis_d.yaxis.set_major_locator(MaxNLocator(nbins=3))

print '--PLOT/SAVE PHASE SLICES VORTEX STATE DISC PADDING = 10'
# Plot phase slices:
axes[1].plot(x_v[0], y_v10[0], '-k', linewidth=1.5, label='analytic')
axes[1].plot(x_v[1], y_v10[1], '-r', linewidth=1.5, label='0.5 nm')
axes[1].plot(x_v[2], y_v10[2], '-m', linewidth=1.5, label='1 nm')
axes[1].plot(x_v[3], y_v10[3], '-y', linewidth=1.5, label='2 nm')
axes[1].plot(x_v[4], y_v10[4], '-g', linewidth=1.5, label='4 nm')
axes[1].plot(x_v[5], y_v10[5], '-c', linewidth=1.5, label='8 nm')
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].set_title('Vortex state disc', fontsize=18)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('phase [mrad]', fontsize=15)
axes[1].set_xlim(0, 128)
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
axes[1].legend()
#    # Plot Zoombox and Arrow:
#    zoom = (59, 340, 10, 55)
#    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
#    axes[1].add_patch(rect)
#    axes[1].arrow(zoom[0]+zoom[2]/2, zoom[1], 0, -193, length_includes_head=True,
#              head_width=2, head_length=20, fc='k', ec='k')
#    # Plot zoom inset:
#    ins_axis_v = plt.axes([0.695, 0.15, 0.075, 0.3])
#    ins_axis_v.plot(x_v[0], y_v10[0], '-k', linewidth=1.5, label='analytic')
#    ins_axis_v.plot(x_v[1], y_v10[1], '-r', linewidth=1.5, label='0.5 nm')
#    ins_axis_v.plot(x_v[2], y_v10[2], '-m', linewidth=1.5, label='1 nm')
#    ins_axis_v.plot(x_v[3], y_v10[3], '-y', linewidth=1.5, label='2 nm')
#    ins_axis_v.plot(x_v[4], y_v10[4], '-g', linewidth=1.5, label='4 nm')
#    ins_axis_v.plot(x_v[5], y_v10[5], '-c', linewidth=1.5, label='8 nm')
#    ins_axis_v.tick_params(axis='both', which='major', labelsize=14)
#    ins_axis_v.set_xlim(zoom[0], zoom[0]+zoom[2])
#    ins_axis_v.set_ylim(zoom[1], zoom[1]+zoom[3])
#    ins_axis_v.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
#    ins_axis_v.yaxis.set_major_locator(MaxNLocator(nbins=4))

plt.show()
plt.figtext(0.15, 0.13, 'c)', fontsize=30)
plt.figtext(0.57, 0.13, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-1-phase_slice_padding_10.png', bbox_inches='tight')

# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Central phase slice errors (padding = 0)', fontsize=20)

print '--PLOT/SAVE PHASE SLICE ERRORS HOMOG. MAGN. DISC PADDING = 0'
# Plot phase slices:
axes[0].plot(x_d[0], dy_d0[0], '-k', linewidth=1.5, label='analytic')
axes[0].plot(x_d[1], dy_d0[1], '-r', linewidth=1.5, label='0.5 nm')
axes[0].plot(x_d[2], dy_d0[2], '-m', linewidth=1.5, label='1 nm')
axes[0].plot(x_d[3], dy_d0[3], '-y', linewidth=1.5, label='2 nm')
axes[0].plot(x_d[4], dy_d0[4], '-g', linewidth=1.5, label='4 nm')
axes[0].plot(x_d[5], dy_d0[5], '-c', linewidth=1.5, label='8 nm')
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].set_title('Homog. magn. disc', fontsize=18)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('phase [mrad]', fontsize=15)
axes[0].set_xlim(0, 128)

print '--PLOT/SAVE PHASE SLICE ERRORS VORTEX STATE DISC PADDING = 0'
# Plot phase slices:
axes[1].plot(x_v[0], dy_v0[0], '-k', linewidth=1.5, label='analytic')
axes[1].plot(x_v[1], dy_v0[1], '-r', linewidth=1.5, label='0.5 nm')
axes[1].plot(x_v[2], dy_v0[2], '-m', linewidth=1.5, label='1 nm')
axes[1].plot(x_v[3], dy_v0[3], '-y', linewidth=1.5, label='2 nm')
axes[1].plot(x_v[4], dy_v0[4], '-g', linewidth=1.5, label='4 nm')
axes[1].plot(x_v[5], dy_v0[5], '-c', linewidth=1.5, label='8 nm')
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].set_title('Vortex state disc', fontsize=18)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('phase [mrad]', fontsize=15)
axes[1].set_xlim(0, 128)
axes[1].legend(loc=4)

plt.show()
plt.figtext(0.15, 0.13, 'c)', fontsize=30)
plt.figtext(0.57, 0.13, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-1-phase_slice_errors_padding_0.png', bbox_inches='tight')

# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Central phase slice errors (padding = 10)', fontsize=20)

print '--PLOT/SAVE PHASE SLICE ERRORS HOMOG. MAGN. DISC PADDING = 10'
# Plot phase slices:
axes[0].plot(x_d[0], dy_d10[0], '-k', linewidth=1.5, label='analytic')
axes[0].plot(x_d[1], dy_d10[1], '-r', linewidth=1.5, label='0.5 nm')
axes[0].plot(x_d[2], dy_d10[2], '-m', linewidth=1.5, label='1 nm')
axes[0].plot(x_d[3], dy_d10[3], '-y', linewidth=1.5, label='2 nm')
axes[0].plot(x_d[4], dy_d10[4], '-g', linewidth=1.5, label='4 nm')
axes[0].plot(x_d[5], dy_d10[5], '-c', linewidth=1.5, label='8 nm')
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].set_title('Homog. magn. disc', fontsize=18)
axes[0].set_xlabel('x [nm]', fontsize=15)
axes[0].set_ylabel('phase [mrad]', fontsize=15)
axes[0].set_xlim(0, 128)

print '--PLOT/SAVE PHASE SLICE ERRORS VORTEX STATE DISC PADDING = 10'
# Plot phase slices:
axes[1].plot(x_v[0], dy_v10[0], '-k', linewidth=1.5, label='analytic')
axes[1].plot(x_v[1], dy_v10[1], '-r', linewidth=1.5, label='0.5 nm')
axes[1].plot(x_v[2], dy_v10[2], '-m', linewidth=1.5, label='1 nm')
axes[1].plot(x_v[3], dy_v10[3], '-y', linewidth=1.5, label='2 nm')
axes[1].plot(x_v[4], dy_v10[4], '-g', linewidth=1.5, label='4 nm')
axes[1].plot(x_v[5], dy_v10[5], '-c', linewidth=1.5, label='8 nm')
axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].set_title('Vortex state disc', fontsize=18)
axes[1].set_xlabel('x [nm]', fontsize=15)
axes[1].set_ylabel('phase [mrad]', fontsize=15)
axes[1].set_xlim(0, 128)
axes[1].legend(loc=4)

plt.show()
plt.figtext(0.15, 0.13, 'c)', fontsize=30)
plt.figtext(0.57, 0.13, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-1-phase_slice_errors_padding_10.png', bbox_inches='tight')

###############################################################################################
print 'CH5-2 PHASE DIFFERENCES FOURIER SPACE'

# Input parameters:
a = 1.0  # in nm
phi = pi/2
dim = (16, 128, 128)  # in px (z, y, x)
center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
radius = dim[1]/4  # in px
height = dim[0]/2  # in px

key = 'ch5-1-phase_diff_mag_dist'
if key in data_shelve and not force_calculation:
    print '--LOAD MAGNETIC DISTRIBUTIONS'
    (mag_data_disc, mag_data_vort) = data_shelve[key]
else:
    print '--CREATE MAGNETIC DISTRIBUTIONS'
    # Create magnetic shape (4 times the size):
    a_big = a / 2
    dim_big = (dim[0]*2, dim[1]*2, dim[2]*2)
    center_big = (dim_big[0]/2-0.5, dim_big[1]/2.-0.5, dim_big[2]/2.-0.5)
    radius_big = dim_big[1]/4  # in px
    height_big = dim_big[0]/2  # in px
    mag_shape = mc.Shapes.disc(dim_big, center_big, radius_big, height_big)
    # Create MagData (4 times the size):
    mag_data_disc = MagData(a_big, mc.create_mag_dist_homog(mag_shape, phi))
    mag_data_vort = MagData(a_big, mc.create_mag_dist_vortex(mag_shape, center_big))
    # Scale mag_data, grid spacing and dimensions:
    mag_data_disc.scale_down()
    mag_data_vort.scale_down()
    print '--SAVE MAGNETIC DISTRIBUTIONS'
    # Shelve magnetic distributions:
    data_shelve[key] = (mag_data_disc, mag_data_vort)

print '--PLOT/SAVE PHASE DIFFERENCES'
# Create projections along z-axis:
projector = SimpleProjector(dim)
# Get analytic solutions:
phase_map_ana_disc = an.phase_mag_disc(dim, a, phi, center, radius, height)
phase_map_ana_vort = an.phase_mag_vortex(dim, a, center, radius, height)

# Create phasemapper:
phasemapper = PMFourier(a, projector, padding=0)
# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Difference of the Fourier space approach from the analytical solution', fontsize=20)
# Create norm for the plots:
bounds = np.array([-100, -50, -25, -5, 0, 5, 25, 50, 100])
norm = BoundaryNorm(bounds, RdBu.N)
# Disc:
phase_map_num_disc = phasemapper(mag_data_disc)
phase_map_num_disc.unit = 'mrad'
phase_diff_disc = phase_map_num_disc - phase_map_ana_disc
RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
phase_diff_disc.display_phase('Homog. magn. disc, RMS = {:3.2f} mrad'.format(RMS_disc),
                              axis=axes[0], limit=np.max(bounds), norm=norm)
axes[0].set_aspect('equal')

# Create norm for the plots:
bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
norm = BoundaryNorm(bounds, RdBu.N)
# Vortex:
phase_map_num_vort = phasemapper(mag_data_vort)
phase_map_num_vort.unit = 'mrad'
phase_diff_vort = phase_map_num_vort - phase_map_ana_vort
phase_diff_vort -= np.mean(phase_diff_vort.phase)
RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))
phase_diff_vort.display_phase(u'Vortex state disc, RMS = {:3.2f} µrad'.format(RMS_vort*1000),
                              axis=axes[1], limit=np.max(bounds), norm=norm)
axes[1].set_aspect('equal')

plt.show()
plt.figtext(0.15, 0.2, 'a)', fontsize=30)
plt.figtext(0.52, 0.2, 'b)', fontsize=30)
plt.savefig(directory + '/ch5-2-fourier_phase_differe_no_padding.png', bbox_inches='tight')

# Create phasemapper:
phasemapper = PMFourier(a, projector, padding=10)
# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Difference of the Fourier space approach from the analytical solution', fontsize=20)
# Create norm for the plots:
bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
norm = BoundaryNorm(bounds, RdBu.N)
# Disc:
phase_map_num_disc = phasemapper(mag_data_disc)
phase_map_num_disc.unit = 'mrad'
phase_diff_disc = phase_map_num_disc - phase_map_ana_disc
RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
phase_diff_disc.display_phase(u'Homog. magn. disc, RMS = {:3.2f} µrad'.format(RMS_disc*1000),
                              axis=axes[0], limit=np.max(bounds), norm=norm)
axes[0].set_aspect('equal')

# Create norm for the plots:
bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
norm = BoundaryNorm(bounds, RdBu.N)
# Vortex:
phase_map_num_vort = phasemapper(mag_data_vort)
phase_map_num_vort.unit = 'mrad'
phase_diff_vort = phase_map_num_vort - phase_map_ana_vort
phase_diff_vort -= np.mean(phase_diff_vort.phase)
RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))
phase_diff_vort.display_phase(u'Vortex state disc, RMS = {:3.2f} µrad'.format(RMS_vort*1000),
                              axis=axes[1], limit=np.max(bounds), norm=norm)
axes[1].set_aspect('equal')

plt.show()
plt.figtext(0.15, 0.2, 'c)', fontsize=30)
plt.figtext(0.52, 0.2, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-2-fourier_phase_differe_padding_10.png', bbox_inches='tight')

###############################################################################################
print 'CH5-2 FOURIER PADDING VARIATION'

# Input parameters:
a = 1.0  # in nm
phi = pi/2
dim = (16, 128, 128)  # in px (z, y, x)
center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
radius = dim[1]/4  # in px
height = dim[0]/2  # in px

key = 'ch5-2-fourier_padding_mag_dist'
if key in data_shelve and not force_calculation:
    print '--LOAD MAGNETIC DISTRIBUTIONS'
    (mag_data_disc, mag_data_vort) = data_shelve[key]
else:
    print '--CREATE MAGNETIC DISTRIBUTIONS'
    # Create magnetic shape (4 times the size):
    a_big = a / 2
    dim_big = (dim[0]*2, dim[1]*2, dim[2]*2)
    center_big = (dim_big[0]/2-0.5, dim_big[1]/2.-0.5, dim_big[2]/2.-0.5)
    radius_big = dim_big[1]/4  # in px
    height_big = dim_big[0]/2  # in px
    mag_shape = mc.Shapes.disc(dim_big, center_big, radius_big, height_big)
    # Create MagData (4 times the size):
    mag_data_disc = MagData(a_big, mc.create_mag_dist_homog(mag_shape, phi))
    mag_data_vort = MagData(a_big, mc.create_mag_dist_vortex(mag_shape, center_big))
    # Scale mag_data, grid spacing and dimensions:
    mag_data_disc.scale_down()
    mag_data_vort.scale_down()
    print '--SAVE MAGNETIC DISTRIBUTIONS'
    # Shelve magnetic distributions:
    data_shelve[key] = (mag_data_disc, mag_data_vort)

# Create projections along z-axis:
projector = SimpleProjector(dim)
# Get analytic solutions:
phase_ana_disc = an.phase_mag_disc(dim, a, phi, center, radius, height)
phase_ana_vort = an.phase_mag_vortex(dim, a, center, radius, height)

# List of applied paddings:
padding_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

print '--LOAD/CREATE PADDING SERIES OF HOMOG. MAGN. DISC'
data_disc = np.zeros((3, len(padding_list)))
data_disc[0, :] = padding_list
for (i, padding) in enumerate(padding_list):
    key = ', '.join(['Padding->RMS|duration', 'Fourier', 'padding={}'.format(padding_list[i]),
                    'a={}'.format(a), 'dim={}'.format(dim), 'phi={}'.format(phi), 'disc'])
    if key in data_shelve and not force_calculation:
        data_disc[:, i] = data_shelve[key]
    else:
        print '----calculate and save padding =', padding_list[i]
        phasemapper = PMFourier(a, projector, padding=padding_list[i])
        start_time = time.time()
        phase_num_disc = phasemapper(mag_data_disc)
        data_disc[2, i] = time.time() - start_time
        phase_map_diff = phase_num_disc - phase_ana_disc
        phase_map_diff.unit = 'mrad'
        phase_map_diff.display_phase()
        data_disc[1, i] = np.sqrt(np.mean(phase_map_diff.phase**2))*1E3  # *1E3: rad to mraddd
        data_shelve[key] = data_disc[:, i]

print '--PLOT/SAVE PADDING SERIES OF HOMOG. MAGN. DISC'
# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Variation of the padding (homog. magn. disc)', fontsize=20)
# Plot RMS against padding:
axes[0].axhline(y=0.18, linestyle='--', color='g', label='RMS [mrad] (real space)')
axes[0].plot(data_disc[0], data_disc[1], 'go-', label='RMS [mrad] (Fourier space)')
axes[0].set_xlabel('padding', fontsize=15)
axes[0].set_ylabel('RMS [mrad]', fontsize=15)
axes[0].set_xlim(-0.5, 16.5)
axes[0].set_ylim(-5, 45)
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].legend()
# Plot zoom inset:
ins_axis_d = plt.axes([0.2, 0.35, 0.25, 0.4])
ins_axis_d.axhline(y=0.18, linestyle='--', color='g')
ins_axis_d.plot(data_disc[0], data_disc[1], 'go-')
ins_axis_d.set_yscale('log')
ins_axis_d.set_xlim(5.5, 16.5)
ins_axis_d.set_ylim(0.1, 1.1)
ins_axis_d.tick_params(axis='both', which='major', labelsize=14)
# Plot duration against padding:
axes[1].plot(data_disc[0], data_disc[2], 'bo-')
axes[1].set_xlabel('padding', fontsize=15)
axes[1].set_ylabel('duration [s]', fontsize=15)
axes[1].set_xlim(-0.5, 16.5)
axes[1].set_ylim(-0.05, 1.5)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=10))
axes[1].tick_params(axis='both', which='major', labelsize=14)

plt.show()
plt.figtext(0.15, 0.13, 'a)', fontsize=30)
plt.figtext(0.57, 0.17, 'b)', fontsize=30)
plt.savefig(directory + '/ch5-2-disc_padding_variation.png', bbox_inches='tight')

print '--LOAD/CREATE PADDING SERIES OF VORTEX STATE DISC'
data_vort = np.zeros((3, len(padding_list)))
data_vort[0, :] = padding_list
for (i, padding) in enumerate(padding_list):
    key = ', '.join(['Padding->RMS|duration', 'Fourier', 'padding={}'.format(padding_list[i]),
                    'a={}'.format(a), 'dim={}'.format(dim), 'phi={}'.format(phi), 'vort'])
    if key in data_shelve and not force_calculation:
        data_vort[:, i] = data_shelve[key]
    else:
        print '----calculate and save padding =', padding_list[i]
        phasemapper = PMFourier(a, projector, padding=padding_list[i])
        start_time = time.time()
        phase_num_vort = phasemapper(mag_data_vort)
        data_vort[2, i] = time.time() - start_time
        phase_map_diff = phase_num_vort - phase_ana_vort
        phase_map_diff -= np.mean(phase_map_diff.phase)
        phase_map_diff.unit = 'mrad'
        phase_map_diff.display_phase()
        data_vort[1, i] = np.sqrt(np.mean(phase_map_diff.phase**2))*1E3  # *1E3: rad to mrad
        data_shelve[key] = data_vort[:, i]

print '--PLOT/SAVE PADDING SERIES OF VORTEX STATE DISC'
# Create figure:
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Variation of the padding (Vortex state disc)', fontsize=20)
# Plot RMS against padding:
axes[0].axhline(y=0.22, linestyle='--', color='g', label='RMS [mrad] (real space)')
axes[0].plot(data_vort[0], data_vort[1], 'go-', label='RMS [mrad] (Fourier space)')
axes[0].set_xlabel('padding', fontsize=15)
axes[0].set_ylabel('RMS [mrad]', fontsize=15)
axes[0].set_xlim(-0.5, 16.5)
axes[0].set_ylim(-5, 45)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
axes[0].legend()
# Plot zoom inset:
ins_axis_v = plt.axes([0.2, 0.35, 0.25, 0.4])
ins_axis_v.axhline(y=0.22, linestyle='--', color='g')
ins_axis_v.plot(data_vort[0], data_vort[1], 'go-')
ins_axis_v.set_yscale('log')
ins_axis_v.set_xlim(5.5, 16.5)
ins_axis_v.set_ylim(0.1, 1.1)
ins_axis_v.tick_params(axis='both', which='major', labelsize=14)
# Plot duration against padding:
axes[1].plot(data_vort[0], data_vort[2], 'bo-')
axes[1].set_xlabel('padding', fontsize=15)
axes[1].set_ylabel('duration [s]', fontsize=15)
axes[1].set_xlim(-0.5, 16.5)
axes[1].set_ylim(-0.05, 1.5)
axes[1].xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
axes[1].yaxis.set_major_locator(MaxNLocator(nbins=10))
axes[1].tick_params(axis='both', which='major', labelsize=14)

plt.show()
plt.figtext(0.15, 0.13, 'c)', fontsize=30)
plt.figtext(0.57, 0.17, 'd)', fontsize=30)
plt.savefig(directory + '/ch5-2-vortex_padding_variation.png', bbox_inches='tight')

###############################################################################################
print 'CLOSING SHELVE\n'
# Close shelve:
data_shelve.close()
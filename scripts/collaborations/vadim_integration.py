# -*- coding: utf-8 -*-
"""Created on Thu Nov 07 16:47:52 2013 @author: Jan """


import os

import numpy as np
from numpy import pi

import pyramid
import pyramid.magcreator as mc
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve, PMElectric
from pyramid.magdata import MagData

import matplotlib.pyplot as plt

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)
# Set constants:
Q_E = 1  # 1.602176565e-19
EPSILON_0 = 1  # 8.8e-9
C_E = 1


def calculate_charge_batch(phase_map):
    def calculate_charge(grad_y, grad_x, l, r, t, b):
        left = np.sum(-grad_x[b:t, l])
        top = np.sum(grad_y[t, l:r])
        right = np.sum(grad_x[b:t, r])
        bottom = np.sum(-grad_y[b, l:r])
        integral = left + top + right + bottom
        Q = -EPSILON_0/C_E * integral
        return Q
    grad_y, grad_x = np.gradient(phase_map.phase, phase_map.a, phase_map.a)
    xi, yi = np.zeros(t-b), np.zeros(t-b)
    for i, ti in enumerate(np.arange(b, t)):
        xi[i] = ti
        yi[i] = calculate_charge(grad_y, grad_x, l, r, ti, b)
    return xi, yi


directory = '../../output/vadim/'
if not os.path.exists(directory):
    os.makedirs(directory)
# Set parameters:
a = 1.0  # in nm
phi = pi/4
density = 30
dim = (128, 128, 128)  # in px (z, y, x)
v_0 = 1
v_acc = 300000
l = dim[2]/4.
r = dim[2]/4. + dim[2]/2.
b = dim[1]/4.
t = dim[1]/4. + dim[1]/2.
# Create magnetic shape:
center = (dim[0]/2.-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
radius = dim[1]/8  # in px
height = dim[0]/4  # in px
width = (dim[0]/4, dim[1]/4, dim[2]/4)
mag_shape_sphere = mc.Shapes.sphere(dim, center, radius)
mag_shape_disc = mc.Shapes.disc(dim, center, radius, height)
mag_shape_slab = mc.Shapes.slab(dim, center, (height, radius*2, radius*2))
mag_shape_ellipsoid = mc.Shapes.ellipsoid(dim, (center[0], 0*center[1], center[2]),
                                          (radius*5, radius*10, radius*2))
# Create magnetization distributions:
mag_data_sphere = MagData(a, mc.create_mag_dist_homog(mag_shape_sphere, phi))
mag_data_disc = MagData(a, mc.create_mag_dist_homog(mag_shape_disc, phi))
mag_data_vortex = MagData(a, mc.create_mag_dist_vortex(mag_shape_disc))
mag_data_slab = MagData(a, mc.create_mag_dist_homog(mag_shape_slab, phi))
mag_data_ellipsoid = MagData(a, mc.create_mag_dist_homog(mag_shape_ellipsoid, phi))
# Create phasemapper:
projector = SimpleProjector(dim)
pm_mag = PMConvolve(a, projector)
pm_ele = PMElectric(a, projector, v_0, v_acc)
# Magnetic phase map of a homogeneously magnetized disc:
phase_map_mag_disc = pm_mag(mag_data_disc)
phase_map_mag_disc.save_to_txt(directory+'phase_map_mag_disc.txt')
axis, _ = phase_map_mag_disc.display_combined(density=density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mag_disc.png')
x, y = calculate_charge_batch(phase_map_mag_disc)
np.savetxt(directory+'charge_integral_mag_disc.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mag_disc.png')
# Magnetic phase map of a vortex state disc:
phase_map_mag_vortex = pm_mag(mag_data_vortex)
phase_map_mag_vortex.save_to_txt(directory+'phase_map_mag_vortex.txt')
axis, _ = phase_map_mag_vortex.display_combined(density=density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mag_vortex.png')
x, y = calculate_charge_batch(phase_map_mag_vortex)
np.savetxt(directory+'charge_integral_mag_vortex.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mag_vortex.png')
# MIP phase of a slab:
phase_map_mip_slab = pm_ele(mag_data_slab)
phase_map_mip_slab.save_to_txt(directory+'phase_map_mip_slab.txt')
axis, _ = phase_map_mip_slab.display_combined(density=density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mip_slab.png')
x, y = calculate_charge_batch(phase_map_mip_slab)
np.savetxt(directory+'charge_integral_mip_slab.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mip_slab.png')
# MIP phase of a disc:
phase_map_mip_disc = pm_ele(mag_data_disc)
phase_map_mip_disc.save_to_txt(directory+'phase_map_mip_disc.txt')
axis, _ = phase_map_mip_disc.display_combined(density=density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mip_disc.png')
x, y = calculate_charge_batch(phase_map_mip_disc)
np.savetxt(directory+'charge_integral_mip_disc.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mip_disc.png')
# MIP phase of a sphere:
phase_map_mip_sphere = pm_ele(mag_data_sphere)
phase_map_mip_sphere.save_to_txt(directory+'phase_map_mip_sphere.txt')
axis, _ = phase_map_mip_sphere.display_combined(density=density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mip_sphere.png')
x, y = calculate_charge_batch(phase_map_mip_sphere)
np.savetxt(directory+'charge_integral_mip_sphere.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mip_sphere.png')
# MIP phase of an ellipsoid:
phase_map_mip_ellipsoid = pm_ele(mag_data_ellipsoid)
phase_map_mip_ellipsoid.save_to_txt(directory+'phase_map_mip_ellipsoid.txt')
axis, _ = phase_map_mip_ellipsoid.display_combined(phase_map_mip_ellipsoid, density)
axis.axvline(l, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axvline(r, b/dim[1], t/dim[1], linewidth=2, color='g')
axis.axhline(b, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.axhline(t, l/dim[2], r/dim[2], linewidth=2, color='g')
axis.arrow(l+(r-l)/2, b, 0, t-b, length_includes_head=True,
           head_width=(r-l)/10, head_length=(r-l)/10, fc='g', ec='g')
plt.savefig(directory+'phase_map_mip_ellipsoid.png')
x, y = calculate_charge_batch(phase_map_mip_ellipsoid)
np.savetxt(directory+'charge_integral_mip_ellipsoid.txt', np.array([x, y]).T)
plt.figure()
plt.plot(x, y)
plt.savefig(directory+'charge_integral_mip_ellipsoid.png')

# -*- coding: utf-8 -*-
"""Created on Tue Sep 03 12:55:40 2013 @author: Jan"""


import os

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

import pyramid

import itertools

import logging
import logging.config


LOGGING_CONF = os.path.join(os.path.dirname(os.path.realpath(pyramid.__file__)), 'logging.ini')


logging.config.fileConfig(LOGGING_CONF, disable_existing_loggers=False)

dim = (8, 8)
offset = (dim[0]/2., dim[1]/2.)
phi = 1.000000001*pi/4
field = np.zeros(dim)


def get_position(p, m, b, size):
    x, y = np.array(p)[:, 0]+0.5, np.array(p)[:, 1]+0.5
    return (y-m*x-b)/np.sqrt(m**2+1) + size/2.


def get_impact(pos, r, size):
    return [x for x in np.arange(np.floor(pos-r), np.floor(pos+r)+1, dtype=int) if 0 <= x < size]


def get_weight(d, r):
    return (1 - d/r) * (d < r)


direction = (-np.cos(phi), np.sin(phi))
xi = range(dim[0])
yj = range(dim[1])
ii, jj = np.meshgrid(xi, yj)

r = 1/np.sqrt(np.pi)

m = direction[0]/direction[1]
b = offset[0] - m * offset[1]

voxels = list(itertools.product(yj, xi))

positions = get_position(voxels, m, b, 2*dim[0])

weights = []
for i, voxel in enumerate(voxels):
    voxel_weights = []
    impacts = get_impact(positions[i], r, dim[0])
    for impact in impacts:
        distance = np.abs(impact+0.5 - positions[i])
        voxel_weights.append((impact, get_weight(distance, r)))
    weights.append((voxel, voxel_weights))

pixels = np.floor(positions).astype(int)

pixel_hits = zip(set(pixels), [list(pixels).count(i) for i in set(pixels)])


def Y(x):
    return direction[0]/direction[1] * (x - offset[1]) + offset[0]


def Y_perp(x):
    return - direction[1]/direction[0] * (x - offset[1]) + offset[0]


def X(y):
    return direction[1]/direction[0] * (y - offset[0]) + offset[1]


x = np.linspace(-2, 2, 10)

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, aspect='equal')
axis.pcolormesh(field, cmap='PuRd')
axis.grid(which='both', color='k', linestyle='-')
x = np.linspace(0, dim[1])
y = Y(x)
y_perp = Y_perp(x)
axis.plot(x, y, '-r', linewidth=2)
axis.plot(x, y_perp, '-g', linewidth=2)
axis.set_xlim(0, dim[1])
axis.set_ylim(0, dim[0])
axis.xaxis.set_major_locator(MaxNLocator(nbins=dim[1], integer=True))
axis.yaxis.set_major_locator(MaxNLocator(nbins=dim[0], integer=True))

for i, p in enumerate(voxels):
    if 0 <= positions[i] < dim[0]:
        color = 'k'
    else:
        color = 'r'
    plt.annotate('{:1.1f}'.format(positions[i]), p, size=8, color=color)

plt.show()

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, aspect='equal')
axis.scatter(positions, 0.5*np.ones_like(positions))
axis.grid(which='both', color='k', linestyle='-')
axis.vlines((0.5*dim[0], 1.5*dim[0]), 0, 1, colors='r', linewidth=3)
axis.set_xlim(0, 2*dim[0])
axis.set_ylim(0, 1)
axis.xaxis.set_major_locator(MaxNLocator(nbins=2*dim[0], integer=True))
axis.yaxis.set_major_locator(NullLocator())

for i, px in enumerate(pixel_hits):
    plt.annotate('{:d}'.format(px[1]), (px[0], 0.6), size=8)

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:26:42 2015

@author: Jan
"""


import numpy as np
import itertools


zoom = 11

R = (3/(4*np.pi))**(1/3.)

Rz = R * zoom

dim_zoom = (3*zoom, 3*zoom)
cent_zoom = (np.asarray(dim_zoom)/2.).astype(dtype=np.int)

y, x = np.indices(dim_zoom)
y -= cent_zoom[0]
x -= cent_zoom[1]

d = np.where(np.hypot(x, y) <= Rz, np.sqrt(Rz**2-x**2-y**2), 0)
d /= d.sum()

lookup = np.zeros((3, 3, zoom, zoom))

for impact in list(itertools.product(range(3), range(3))):
    imp_zoom = np.array(impact) * zoom
    for position in list(itertools.product(range(zoom), range(zoom))):
        shift = np.array(position) - np.array((zoom//2, zoom//2))
        imp_shift = imp_zoom - shift
        lb = np.where(imp_shift >= 0, imp_shift, [0, 0])
        tr = np.where(imp_shift < 3*zoom, imp_shift+np.array((zoom, zoom)), [3*zoom, 3*zoom])
        weight = d[lb[0]:tr[0], lb[1]:tr[1]].sum()
        lookup[impact[0], impact[1], position[0], position[1]] = weight

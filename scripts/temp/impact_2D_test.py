# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:18:22 2015

@author: Jan
"""


import numpy as np
import matplotlib.pyplot as plt
import pyramid as py


# INPUT: ##########################################################################################
position = (1.4, 1.3)
dim_uv = (3, 3)
###################################################################################################

pos_rel, pos_ind = np.modf(position)

R = (3/(4*np.pi))**(1/3.)

impacts = [pos_ind]

0 <= pos_ind[0] < dim_uv[0]

# pixel top:
if 1 - pos_rel[0] < R and pos_ind[0] < dim_uv[0]-1:
    impacts.append(pos_ind + np.array([1, 0]))
# pixel bottom:
if pos_rel[0] < R and pos_ind[0] > 0:
    impacts.append(pos_ind + np.array([-1, 0]))
# pixel right:
if 1 - pos_rel[1] < R and pos_ind[1] < dim_uv[1]-1:
    impacts.append(pos_ind + np.array([0, 1]))
# pixel left:
if pos_rel[1] < R and pos_ind[1] > 0:
    impacts.append(pos_ind + np.array([0, -1]))
# pixel top-right:
if np.hypot(*(pos_rel-np.array([1, 1]))) < R and pos_ind[0] < dim_uv[0]-1 and pos_ind[1] < dim_uv[1]-1:
    impacts.append(pos_ind + np.array([1, 1]))
# pixel top-left:
if np.hypot(*(pos_rel-np.array([1, 0]))) < R and pos_ind[0] < dim_uv[0]-1 and pos_ind[1] > 0:
    impacts.append(pos_ind + np.array([1, -1]))
# pixel bottom-right:
if np.hypot(*(pos_rel-np.array([0, 1]))) < R and pos_ind[0] > 0 and pos_ind[1] < dim_uv[1]-1:
    impacts.append(pos_ind + np.array([-1, 1]))
# pixel bottom-left:
if np.hypot(*(pos_rel-np.array([0, 0]))) < R and pos_ind[0] > 0 and pos_ind[1] > 0:
    impacts.append(pos_ind + np.array([-1, -1]))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.set_aspect('equal')
axis.scatter(position[1], position[0], color='g')
axis.scatter(np.array(impacts)[:, 1]+0.5, np.array(impacts)[:, 0]+0.5)
circle = plt.Circle((position[1], position[0]), R, color='g', fill=False)
axis = plt.gcf().gca()
axis.add_artist(circle)
axis.set_xlim(-0.6, dim_uv[1]+0.6)
axis.set_ylim(-0.6, dim_uv[0]+0.6)
axis.axhline(0, color='r')
axis.axhline(1)
axis.axhline(2)
axis.axhline(3, color='r')
axis.axvline(0, color='r')
axis.axvline(1)
axis.axvline(2)
axis.axvline(3, color='r')

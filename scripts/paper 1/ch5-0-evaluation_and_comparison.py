# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:20 2013

@author: Jan
"""


import sys
import traceback
import pdb
import os

import numpy as np
from numpy import pi

import shelve

import pyramid.magcreator as mc
from pyramid.magdata import MagData

import matplotlib.pyplot as plt

from matplotlib.ticker import FixedFormatter, IndexLocator


def run():

    print '\nACCESS SHELVE'
    # Create / Open databank:
    directory = '../../output/paper 1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_shelve = shelve.open(directory + '/paper_1_shelve')

    ###############################################################################################
    print 'CH5-0 KERNELS'
    
    x = np.linspace(-5, 5, 1000)
    
    y_r = x/np.abs(x)**3
    y_k = x/np.abs(x)**2
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, aspect='equal')
    axis.plot(x,y_r, 'r', label=r'$r/|r|^3$')
    axis.plot(x,y_k, 'b', label=r'$k/|k|^2$')
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.axvline(0, linewidth=2, color='k')
    axis.axhline(0, linewidth=2, color='k')
    axis.legend()

    ###############################################################################################
    print 'CH5-0 MAGNETIC DISTRIBUTIONS'

    key = 'ch5-0-magnetic_distributions'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTIONS'
        (mag_data_disc, mag_data_vort) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTIONS'
        # Input parameters:
        a = 1.0  # in nm
        phi = pi/2
        dim = (16, 128, 128)  # in px (z, y, x)
        # Create magnetic shape:
        center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
        radius = dim[1]/4  # in px
        height = dim[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        print '--CREATE MAGN. DISTR. OF HOMOG. MAG. DISC'
        mag_data_disc = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
        mag_data_disc.scale_down(2)
        print '--CREATE MAGN. DISTR. OF VORTEX STATE DISC'
        mag_data_vort = MagData(a, mc.create_mag_dist_vortex(mag_shape, center))
        mag_data_vort.scale_down(2)
        # Mayavi-Plots:
        mag_data_disc.quiver_plot3d()
        mag_data_vort.quiver_plot3d()
        print '--SHELVE MAGNETIC DISTRIBUTIONS'
        data_shelve[key] = (mag_data_disc, mag_data_vort)

    print '--PLOT/SAVE MAGNETIC DISTRIBUTIONS'
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Magnetic Distributions', fontsize=20)
    # Plot MagData (Disc):
    mag_data_disc.quiver_plot('Homog. magn. disc', axis=axes[0])
    axes[0].set_aspect('equal')
    axes[0].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))
    axes[0].yaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))
    axes[0].xaxis.set_major_formatter(FixedFormatter([16*i for i in range(10)]))
    axes[0].yaxis.set_major_formatter(FixedFormatter([16*i for i in range(10)]))
    axes[0].set_xlabel('x [nm]', fontsize=15)
    axes[0].set_ylabel('x [ym]', fontsize=15)
    # Plot MagData (Disc):
    mag_data_vort.quiver_plot('Vortex state disc', axis=axes[1])
    axes[1].set_aspect('equal')
    axes[1].xaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))
    axes[1].yaxis.set_major_locator(IndexLocator(base=4, offset=-0.5))
    axes[1].xaxis.set_major_formatter(FixedFormatter([16*i for i in range(10)]))
    axes[1].yaxis.set_major_formatter(FixedFormatter([16*i for i in range(10)]))
    axes[1].set_xlabel('x [nm]', fontsize=15)
    axes[1].set_ylabel('x [ym]', fontsize=15)
    # Save Plots:
    plt.figtext(0.15, 0.15, 'a)', fontsize=30)
    plt.figtext(0.57, 0.15, 'b)', fontsize=30)
    plt.savefig(directory + '/ch5-0-magnetic_distributions.png', bbox_inches='tight')

    ###############################################################################################
    print 'CLOSING SHELVE\n'
    # Close shelve:
    data_shelve.close()

    ###############################################################################################

if __name__ == "__main__":
    try:
        run()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

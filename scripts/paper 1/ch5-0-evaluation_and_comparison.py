# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:20 2013

@author: Jan
"""


import sys
import traceback
import pdb
import os

from numpy import pi

import shelve

import pyramid.magcreator as mc
from pyramid.magdata import MagData

import matplotlib.pyplot as plt


def run():

    print '\nACCESS SHELVE'
    # Create / Open databank:
    directory = '../../output/paper 1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_shelve = shelve.open(directory + '/paper_1_shelve')

    ###############################################################################################
    print 'CH5-0 MAGNETIC DISTRIBUTIONS'

    key = 'ch5-0-magnetic_distributions'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTIONS'
        (mag_data_disc, mag_data_vort) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTIONS'
        # Input parameters:
        res = 0.5  # in nm
        phi = pi/2
        dim = (32, 256, 256)  # in px (z, y, x)
        # Create magnetic shape:
        center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts at 0!
        radius = dim[1]/4  # in px
        height = dim[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        print '--CREATE MAGN. DISTR. OF HOMOG. MAG. DISC'
        mag_data_disc = MagData(res, mc.create_mag_dist_homog(mag_shape, phi))
        mag_data_disc.scale_down(3)
        print '--CREATE MAGN. DISTR. OF VORTEX STATE DISC'
        mag_data_vort = MagData(res, mc.create_mag_dist_vortex(mag_shape, center))
        mag_data_vort.scale_down(3)
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
    # Plot MagData (Disc):
    mag_data_vort.quiver_plot('Vortex state disc', axis=axes[1])
    axes[1].set_aspect('equal')
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

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:37:30 2013

@author: Jan
"""
import pdb
import traceback
import sys
import os

import numpy as np
from numpy import pi

import shelve

import pyramid.magcreator as mc
import pyramid.projector as pj
import pyramid.phasemapper as pm
import pyramid.holoimage as hi
import pyramid.analytic as an
from pyramid.magdata import MagData
from pyramid.phasemap import PhaseMap

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import RdBu
from matplotlib.patches import Rectangle


PHI_0 = -2067.83  # magnetic flux in T*nm²


def run():

    print '\nACCESS SHELVE'
    # Create / Open databank:
    directory = '../../output/paper 1'
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_shelve = shelve.open(directory + '/paper_1_shelve')

    ###############################################################################################
    print 'CH5-1 ANALYTIC SOLUTIONS'

    # Input parameters:
    res = 0.125  # in nm
    phi = pi/2
    dim = (128, 1024, 1024)  # in px (z, y, x)
    # Create magnetic shape:
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px
    print '--CALCULATE ANALYTIC SOLUTIONS'
    # Get analytic solution:
    phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
    phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)
    phase_map_ana_disc = PhaseMap(res, phase_ana_disc)
    phase_map_ana_vort = PhaseMap(res, phase_ana_vort)
    print '--PLOT/SAVE ANALYTIC SOLUTIONS'
    hi.display_combined(phase_map_ana_disc, 100, 'Analytic solution: hom. magn. disc', 'bilinear')
    axis = plt.gcf().add_subplot(1, 2, 2, aspect='equal')
    axis.axhline(y=512, linewidth=3, linestyle='--', color='r')
    plt.savefig(directory + '/ch5-1-analytic_solution_disc.png', bbox_inches='tight')
    hi.display_combined(phase_map_ana_vort, 100, 'Analytic solution: Vortex state', 'bilinear')
    axis = plt.gcf().add_subplot(1, 2, 2, aspect='equal')
    axis.axhline(y=512, linewidth=3, linestyle='--', color='r')
    plt.savefig(directory + '/ch5-1-analytic_solution_vort.png', bbox_inches='tight')
    # Get colorwheel:
    hi.make_color_wheel()
    plt.savefig(directory + '/ch5-1-colorwheel.png', bbox_inches='tight')

    ###############################################################################################
    print 'CH5-1 PHASE SLICES REAL SPACE'  # TODO: Verschieben
    
    # Input parameters:
    res = 0.25  # in nm
    phi = pi/2
    density = 20
    dim = (64, 512, 512)  # in px (z, y, x)
    # Create magnetic shape:
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px
    
    key = 'ch5-1-phase_slice_mag_dist'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTION'
        mag_shape = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTION'
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        print '--SAVE MAGNETIC DISTRIBUTION'
        data_shelve[key] = mag_shape

    key = 'ch5-1-phase_slice_disc'
    if key in data_shelve:
        print '--LOAD PHASE SLICES HOMOG. MAGN. DISC'
        (x, y) = data_shelve[key]
    else:
        print '--CREATE PHASE SLICES HOMOG. MAGN. DISC'
        # Arrays for plotting:
        x = []
        y = []
        # Analytic solution:
        L = dim[1] * res  # in px/nm
        Lz = 0.5 * dim[0] * res  # in px/nm
        R = 0.25 * L  # in px/nm
        x0 = L / 2  # in px/nm
    
        def F_disc(x):
            coeff = - pi * Lz / (2*PHI_0)
            result = coeff * (- (x - x0) * np.sin(phi))
            result *= np.where(np.abs(x - x0) <= R, 1, (R / (x - x0)) ** 2)
            return result
        
        x.append(np.linspace(0, L, 5000))
        y.append(F_disc(x[0]))
        # Create and plot MagData (Disc):
        mag_data_disc = MagData(res, mc.create_mag_dist(mag_shape, phi))
        for i in range(5):
            mag_data_disc.scale_down()
            print '----res =', mag_data_disc.res, 'nm', 'dim =', mag_data_disc.dim
            projection = pj.simple_axis_projection(mag_data_disc)
            phase_map = PhaseMap(mag_data_disc.res, 
                                 pm.phase_mag_real(mag_data_disc.res, projection, 'slab'))
            hi.display_combined(phase_map, density, 'Disc, res = {}'.format(res))
            x.append(np.linspace(0, mag_data_disc.dim[1]*mag_data_disc.res, mag_data_disc.dim[1]))
            y.append(phase_map.phase[int(mag_data_disc.dim[1]/2), :])
        # Shelve x and y:
        print '--SAVE PHASE SLICES HOMOG. MAGN. DISC'
        data_shelve[key] = (x, y)
    
    print '--PLOT/SAVE PHASE SLICES HOMOG. MAGN. DISC'
    # Plot phase slices:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x[0], y[0], 'k', label='analytic')
    axis.plot(x[1], y[1], 'r', label='0.5 nm')
    axis.plot(x[2], y[2], 'm', label='1 nm')
    axis.plot(x[3], y[3], 'y', label='2 nm')
    axis.plot(x[4], y[4], 'g', label='4 nm')
    axis.plot(x[5], y[5], 'c', label='8 nm')
    plt.tick_params(axis='both', which='major', labelsize=14)
    axis.set_title('DISC', fontsize=18)
    axis.set_xlabel('x [nm]', fontsize=15)
    axis.set_ylabel('phase [rad]', fontsize=15)
    axis.set_xlim(0, 128)
    axis.set_ylim(-0.22, 0.22)
    axis.legend()
    # Plot Zoombox and Arrow:
    rect1 = Rectangle((23.5, 0.16), 15, 0.04, fc='w', ec='k')
    axis.add_patch(rect1)
    plt.arrow(32.5, 0.16, 0.0, -0.16, length_includes_head=True, 
              head_width=2, head_length=0.02, fc='k', ec='k')
    # Plot zoom inset:
    ins_axis = plt.axes([0.2, 0.2, 0.3, 0.3])
    ins_axis.plot(x[0], y[0], 'k', label='analytic')
    ins_axis.plot(x[1], y[1], 'r', label='0.5 nm')
    ins_axis.plot(x[2], y[2], 'm', label='1 nm')
    ins_axis.plot(x[3], y[3], 'y', label='2 nm')
    ins_axis.plot(x[4], y[4], 'g', label='4 nm')
    ins_axis.plot(x[5], y[5], 'c', label='8 nm')
    plt.tick_params(axis='both', which='major', labelsize=14)
    ins_axis.set_xlim(23.5, 38.5)
    ins_axis.set_ylim(0.16, 0.2)
    ins_axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
    ins_axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.show()
    plt.savefig(directory + '/ch5-1-disc_slice_comparison.png', bbox_inches='tight')
    
    key = 'ch5-1-phase_slice_vort'
    if key in data_shelve:
        print '--LOAD PHASE SLICES VORTEX STATE DISC'
        (x, y) = data_shelve[key]
    else:
        print '--CREATE PHASE SLICES VORTEX STATE DISC'
        x = []
        y = []
        # Analytic solution:
        L = dim[1] * res  # in px/nm
        Lz = 0.5 * dim[0] * res  # in px/nm
        R = 0.25 * L  # in px/nm
        x0 = L / 2  # in px/nm
    
        def F_vort(x):
            coeff = pi*Lz/PHI_0
            result = coeff * np.where(np.abs(x - x0) <= R, (np.abs(x-x0)-R), 0)
            return result
        
        x.append(np.linspace(0, L, 5001))
        y.append(F_vort(x[0]))
        # Create and plot MagData (Vortex):
        mag_data_vort = MagData(res, mc.create_mag_dist_vortex(mag_shape))
        for i in range(5):
            mag_data_vort.scale_down()
            print '----i =', i, 'dim =', mag_data_vort.dim, 'res =', mag_data_vort.res, 'nm'
            projection = pj.simple_axis_projection(mag_data_vort)
            phase_map = PhaseMap(mag_data_vort.res, 
                                 pm.phase_mag_real(mag_data_vort.res, projection, 'slab'))
            hi.display_combined(phase_map, density, 'Disc, res = {}'.format(res))
            x.append(np.linspace(0, mag_data_vort.dim[1]*mag_data_vort.res, mag_data_vort.dim[1]))
            y.append(phase_map.phase[int(mag_data_vort.dim[1]/2), :])
        # Shelve x and y:
        print '--SAVE PHASE SLICES VORTEX STATE DISC'
        data_shelve[key] = (x, y)

    print '--PLOT/SAVE PHASE SLICES VORTEX STATE DISC'
    # Plot:
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(x[0], y[0], 'k', label='analytic')
    axis.plot(x[1], y[1], 'r', label='0.5 nm')
    axis.plot(x[2], y[2], 'm', label='1 nm')
    axis.plot(x[3], y[3], 'y', label='2 nm')
    axis.plot(x[4], y[4], 'g', label='4 nm')
    axis.plot(x[5], y[5], 'c', label='8 nm')
    plt.tick_params(axis='both', which='major', labelsize=14)
    axis.set_title('VORTEX', fontsize=18)
    axis.set_xlabel('x [nm]', fontsize=15)
    axis.set_ylabel('phase [rad]', fontsize=15)
    axis.set_xlim(0, 128)
    axis.legend()
    # Plot Zoombox and Arrow:
    zoom = (59, 0.34, 10, 0.055)
    rect1 = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
    axis.add_patch(rect1)
    plt.arrow(zoom[0]+zoom[2]/2, zoom[1], 0, -0.19, length_includes_head=True, 
              head_width=2, head_length=0.02, fc='k', ec='k')
    # Plot zoom inset:
    ins_axis = plt.axes([0.47, 0.15, 0.15, 0.3])
    ins_axis.plot(x[0], y[0], 'k', label='analytic')
    ins_axis.plot(x[1], y[1], 'r', label='0.5 nm')
    ins_axis.plot(x[2], y[2], 'm', label='1 nm')
    ins_axis.plot(x[3], y[3], 'y', label='2 nm')
    ins_axis.plot(x[4], y[4], 'g', label='4 nm')
    ins_axis.plot(x[5], y[5], 'c', label='8 nm')
    plt.tick_params(axis='both', which='major', labelsize=14)
    ins_axis.set_xlim(zoom[0], zoom[0]+zoom[2])
    ins_axis.set_ylim(zoom[1], zoom[1]+zoom[3])
    ins_axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
    ins_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.savefig(directory + '/ch5-1-vortex_slice_comparison.png', bbox_inches='tight')
    
    ###############################################################################################
    print 'CH5-1 PHASE DIFFERENCES REAL SPACE'

    # Input parameters:
    res = 1.0  # in nm
    phi = pi/2
    dim = (16, 128, 128)  # in px (z, y, x)
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px

    key = 'ch5-1-phase_diff_mag_dist'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTIONS'
        (mag_data_disc, mag_data_vort) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTIONS'
        # Create magnetic shape (4 times the size):
        res_big = res / 2
        dim_big = (dim[0]*2, dim[1]*2, dim[2]*2)
        center_big = (dim_big[0]/2-0.5, dim_big[1]/2.-0.5, dim_big[2]/2.-0.5)
        radius_big = dim_big[1]/4  # in px
        height_big = dim_big[0]/2  # in px
        mag_shape = mc.Shapes.disc(dim_big, center_big, radius_big, height_big)
        # Create MagData (4 times the size):
        mag_data_disc = MagData(res_big, mc.create_mag_dist(mag_shape, phi))
        mag_data_vort = MagData(res_big, mc.create_mag_dist_vortex(mag_shape, center_big))
        # Scale mag_data, resolution and dimensions:
        mag_data_disc.scale_down()
        mag_data_vort.scale_down()
        print '--SAVE MAGNETIC DISTRIBUTIONS'
        # Shelve magnetic distributions:
        data_shelve[key] = (mag_data_disc, mag_data_vort)

    print '--PLOT/SAVE PHASE DIFFERENCES'
    # Create projections along z-axis:
    projection_disc = pj.simple_axis_projection(mag_data_disc)
    projection_vort = pj.simple_axis_projection(mag_data_vort)
    # Get analytic solutions:
    phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
    phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)
    # Create norm for the plots:
    bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
    norm = BoundaryNorm(bounds, RdBu.N)
    # Disc:
    phase_num_disc = pm.phase_mag_real(res, projection_disc, 'slab')
    phase_diff_disc = PhaseMap(res, (phase_num_disc-phase_ana_disc)*1E3)  # in mrad -> *1000)
    RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
    phase_diff_disc.display('Deviation (homog. magn. disc), RMS = {:3.2f} mrad'.format(RMS_disc),
                            labels=('x-axis [nm]', 'y-axis [nm]', '$\Delta$phase [mrad]'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-1-disc_phase_diff.png', bbox_inches='tight')
    # Vortex:
    phase_num_vort = pm.phase_mag_real(res, projection_vort, 'slab')
    phase_diff_vort = PhaseMap(res, (phase_num_vort-phase_ana_vort)*1E3)  # in mrad -> *1000
    RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))
    phase_diff_vort.display('Deviation (vortex state disc), RMS = {:3.2f} mrad'.format(RMS_vort),
                            labels=('x-axis [nm]', 'y-axis [nm]', '$\Delta$phase [mrad]'),
                            limit=np.max(bounds), norm=norm)
    plt.savefig(directory + '/ch5-1-vortex_phase_diff.png', bbox_inches='tight')

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


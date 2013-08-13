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
    phase_map_ana_disc = PhaseMap(res, phase_ana_disc*1E3)  # in mrad -> *1000
    phase_map_ana_vort = PhaseMap(res, phase_ana_vort*1E3)  # in mrad -> *1000
    print '--PLOT/SAVE ANALYTIC SOLUTIONS'
    hi.display_combined(phase_map_ana_disc, 0.1, 'Analytic solution: hom. magn. disc', 'bilinear',
                        labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
    axis = plt.gcf().add_subplot(1, 2, 2, aspect='equal')
    axis.axhline(y=512, linewidth=3, linestyle='--', color='r')
    plt.figtext(0.15, 0.2, 'a)', fontsize=30, color='w')
    plt.figtext(0.52, 0.2, 'b)', fontsize=30)
    plt.savefig(directory + '/ch5-1-analytic_solution_disc.png', bbox_inches='tight')
    hi.display_combined(phase_map_ana_vort, 0.1, 'Analytic solution: vortex state', 'bilinear',
                        labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
    axis = plt.gcf().add_subplot(1, 2, 2, aspect='equal')
    axis.axhline(y=512, linewidth=3, linestyle='--', color='r')
    plt.figtext(0.15, 0.2, 'c)', fontsize=30, color='w')
    plt.figtext(0.52, 0.2, 'd)', fontsize=30)
    plt.savefig(directory + '/ch5-1-analytic_solution_vort.png', bbox_inches='tight')
    # Get colorwheel:
    hi.make_color_wheel()
    plt.figtext(0.15, 0.14, 'e)', fontsize=30, color='w')
    plt.savefig(directory + '/ch5-1-colorwheel.png', bbox_inches='tight')

    ###############################################################################################
    print 'CH5-1 PHASE SLICES REAL SPACE'
    
    # Input parameters:
    res = 0.25  # in nm
    phi = pi/2
    density = 0.1  # Because phase is in mrad -> amplification by 100 (0.001 * 100 = 0.1)
    dim = (64, 512, 512)  # in px (z, y, x)
    # Create magnetic shape:
    center = (dim[0]/2-0.5, dim[1]/2.-0.5, dim[2]/2.-0.5)  # in px (z, y, x) index starts with 0!
    radius = dim[1]/4  # in px
    height = dim[0]/2  # in px
    
    key = 'ch5-1-phase_slices_real'
    if key in data_shelve:
        print '--LOAD MAGNETIC DISTRIBUTION'
        (x_d, y_d, dy_d, x_v, y_v, dy_v) = data_shelve[key]
    else:
        print '--CREATE MAGNETIC DISTRIBUTION'
        mag_shape = mc.Shapes.disc(dim, center, radius, height)
        
        print '--CREATE PHASE SLICES HOMOG. MAGN. DISC'
        # Arrays for plotting:
        x_d = []
        y_d = []
        dy_d = []
        # Analytic solution:
        L = dim[1] * res  # in px/nm
        Lz = 0.5 * dim[0] * res  # in px/nm
        R = 0.25 * L  # in px/nm
        x0 = L / 2  # in px/nm
    
        def F_disc(x):
            coeff = - pi * Lz / (2*PHI_0) * 1E3  # in mrad -> *1000
            result = coeff * (- (x - x0) * np.sin(phi))
            result *= np.where(np.abs(x - x0) <= R, 1, (R / (x - x0)) ** 2)
            return result
        
        x_d.append(np.linspace(0, L, 5000))
        y_d.append(F_disc(x_d[0]))
        dy_d.append(np.zeros_like(x_d[0]))
        # Create MagData (Disc):
        mag_data_disc = MagData(res, mc.create_mag_dist(mag_shape, phi))
        for i in range(5):
            mag_data_disc.scale_down()
            print '----res =', mag_data_disc.res, 'nm', 'dim =', mag_data_disc.dim
            projection = pj.simple_axis_projection(mag_data_disc)
            phase_map = PhaseMap(mag_data_disc.res, 
                                 pm.phase_mag_real(mag_data_disc.res, projection, 'slab') * 1E3)
            hi.display_combined(phase_map, density, 'Disc, res = {} nm'.format(mag_data_disc.res),
                                labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
            x_d.append(np.linspace(mag_data_disc.res * 0.5, 
                                   mag_data_disc.res * (mag_data_disc.dim[1]-0.5), 
                                   mag_data_disc.dim[1]))
            slice_pos = int(mag_data_disc.dim[1]/2)
            y_d.append(phase_map.phase[slice_pos, :])
            dy_d.append(phase_map.phase[slice_pos, :] - F_disc(x_d[-1]))

        print '--CREATE PHASE SLICES VORTEX STATE DISC'
        x_v = []
        y_v = []
        dy_v = []
        # Analytic solution:
        L = dim[1] * res  # in px/nm
        Lz = 0.5 * dim[0] * res  # in px/nm
        R = 0.25 * L  # in px/nm
        x0 = L / 2  # in px/nm
    
        def F_vort(x):
            coeff = pi*Lz/PHI_0 * 1E3  # in mrad -> *1000
            result = coeff * np.where(np.abs(x - x0) <= R, (np.abs(x-x0)-R), 0)
            return result
        
        x_v.append(np.linspace(0, L, 5001))
        y_v.append(F_vort(x_v[0]))
        dy_v.append(np.zeros_like(x_v[0]))
        # Create MagData (Vortex):
        mag_data_vort = MagData(res, mc.create_mag_dist_vortex(mag_shape))
        for i in range(5):
            mag_data_vort.scale_down()
            print '----res =', mag_data_vort.res, 'nm', 'dim =', mag_data_vort.dim
            projection = pj.simple_axis_projection(mag_data_vort)
            phase_map = PhaseMap(mag_data_vort.res, 
                                 pm.phase_mag_real(mag_data_vort.res, projection, 'slab') * 1E3)
            hi.display_combined(phase_map, density, 'Disc, res = {} nm'.format(mag_data_vort.res),
                                labels=('x-axis [nm]', 'y-axis [nm]', 'phase [mrad]'))
            x_v.append(np.linspace(mag_data_vort.res * 0.5, 
                                   mag_data_vort.res * (mag_data_vort.dim[1]-0.5), 
                                   mag_data_vort.dim[1]))
            slice_pos = int(mag_data_vort.dim[1]/2)
            y_v.append(phase_map.phase[int(mag_data_vort.dim[1]/2), :])
            dy_v.append(phase_map.phase[slice_pos, :] - F_vort(x_v[-1]))

        # Shelve x, y and dy:
        print '--SAVE PHASE SLICES'
        data_shelve[key] = (x_d, y_d, dy_d, x_v, y_v, dy_v)

    # Create figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Central phase slices', fontsize=20)

    print '--PLOT/SAVE PHASE SLICES HOMOG. MAGN. DISC'
    # Plot phase slices:
    axes[0].plot(x_d[0], y_d[0], '-k', linewidth=1.5, label='analytic')
    axes[0].plot(x_d[1], y_d[1], '-r', linewidth=1.5, label='0.5 nm')
    axes[0].plot(x_d[2], y_d[2], '-m', linewidth=1.5, label='1 nm')
    axes[0].plot(x_d[3], y_d[3], '-y', linewidth=1.5, label='2 nm')
    axes[0].plot(x_d[4], y_d[4], '-g', linewidth=1.5, label='4 nm')
    axes[0].plot(x_d[5], y_d[5], '-c', linewidth=1.5, label='8 nm')
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].set_title('Homog. magn. disc', fontsize=18)
    axes[0].set_xlabel('x [nm]', fontsize=15)
    axes[0].set_ylabel('phase [mrad]', fontsize=15)
    axes[0].set_xlim(0, 128)
    axes[0].set_ylim(-220, 220)
    # Plot Zoombox and Arrow:
    zoom = (23.5, 160, 15, 40)
    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
    axes[0].add_patch(rect)
    axes[0].arrow(zoom[0]+zoom[2], zoom[1]+zoom[3]/2, 36, 0, length_includes_head=True, 
              head_width=10, head_length=4, fc='k', ec='k')
    # Plot zoom inset:
    ins_axis_d = plt.axes([0.33, 0.57, 0.14, 0.3])
    ins_axis_d.plot(x_d[0], y_d[0], '-k', linewidth=1.5, label='analytic')
    ins_axis_d.plot(x_d[1], y_d[1], '-r', linewidth=1.5, label='0.5 nm')
    ins_axis_d.plot(x_d[2], y_d[2], '-m', linewidth=1.5, label='1 nm')
    ins_axis_d.plot(x_d[3], y_d[3], '-y', linewidth=1.5, label='2 nm')
    ins_axis_d.plot(x_d[4], y_d[4], '-g', linewidth=1.5, label='4 nm')
    ins_axis_d.plot(x_d[5], y_d[5], '-c', linewidth=1.5, label='8 nm')
    ins_axis_d.tick_params(axis='both', which='major', labelsize=14)
    ins_axis_d.set_xlim(zoom[0], zoom[0]+zoom[2])
    ins_axis_d.set_ylim(zoom[1], zoom[1]+zoom[3])
    ins_axis_d.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
    ins_axis_d.yaxis.set_major_locator(MaxNLocator(nbins=3))

    print '--PLOT/SAVE PHASE SLICES VORTEX STATE DISC'
    # Plot phase slices:
    axes[1].plot(x_v[0], y_v[0], '-k', linewidth=1.5, label='analytic')
    axes[1].plot(x_v[1], y_v[1], '-r', linewidth=1.5, label='0.5 nm')
    axes[1].plot(x_v[2], y_v[2], '-m', linewidth=1.5, label='1 nm')
    axes[1].plot(x_v[3], y_v[3], '-y', linewidth=1.5, label='2 nm')
    axes[1].plot(x_v[4], y_v[4], '-g', linewidth=1.5, label='4 nm')
    axes[1].plot(x_v[5], y_v[5], '-c', linewidth=1.5, label='8 nm')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].set_title('Vortex state disc', fontsize=18)
    axes[1].set_xlabel('x [nm]', fontsize=15)
    axes[1].set_ylabel('phase [mrad]', fontsize=15)
    axes[1].set_xlim(0, 128)
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[1].legend()
    # Plot Zoombox and Arrow:
    zoom = (59, 340, 10, 55)
    rect = Rectangle((zoom[0], zoom[1]), zoom[2], zoom[3], fc='w', ec='k')
    axes[1].add_patch(rect)
    axes[1].arrow(zoom[0]+zoom[2]/2, zoom[1], 0, -193, length_includes_head=True, 
              head_width=2, head_length=20, fc='k', ec='k')
    # Plot zoom inset:
    ins_axis_v = plt.axes([0.695, 0.15, 0.075, 0.3])
    ins_axis_v.plot(x_v[0], y_v[0], '-k', linewidth=1.5, label='analytic')
    ins_axis_v.plot(x_v[1], y_v[1], '-r', linewidth=1.5, label='0.5 nm')
    ins_axis_v.plot(x_v[2], y_v[2], '-m', linewidth=1.5, label='1 nm')
    ins_axis_v.plot(x_v[3], y_v[3], '-y', linewidth=1.5, label='2 nm')
    ins_axis_v.plot(x_v[4], y_v[4], '-g', linewidth=1.5, label='4 nm')
    ins_axis_v.plot(x_v[5], y_v[5], '-c', linewidth=1.5, label='8 nm')
    ins_axis_v.tick_params(axis='both', which='major', labelsize=14)
    ins_axis_v.set_xlim(zoom[0], zoom[0]+zoom[2])
    ins_axis_v.set_ylim(zoom[1], zoom[1]+zoom[3])
    ins_axis_v.xaxis.set_major_locator(MaxNLocator(nbins=4, integer= True))
    ins_axis_v.yaxis.set_major_locator(MaxNLocator(nbins=4))
    
    plt.show()
    plt.figtext(0.15, 0.13, 'a)', fontsize=30)
    plt.figtext(0.57, 0.13, 'b)', fontsize=30)
    plt.savefig(directory + '/ch5-1-phase_slice_comparison.png', bbox_inches='tight')
    
    # Create figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Central phase slice errors', fontsize=20)

    print '--PLOT/SAVE PHASE SLICE ERRORS HOMOG. MAGN. DISC'
    # Plot phase slices:
    axes[0].plot(x_d[0], dy_d[0], '-k', linewidth=1.5, label='analytic')
    axes[0].plot(x_d[1], dy_d[1], '-r', linewidth=1.5, label='0.5 nm')
    axes[0].plot(x_d[2], dy_d[2], '-m', linewidth=1.5, label='1 nm')
    axes[0].plot(x_d[3], dy_d[3], '-y', linewidth=1.5, label='2 nm')
    axes[0].plot(x_d[4], dy_d[4], '-g', linewidth=1.5, label='4 nm')
    axes[0].plot(x_d[5], dy_d[5], '-c', linewidth=1.5, label='8 nm')
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].set_title('Homog. magn. disc', fontsize=18)
    axes[0].set_xlabel('x [nm]', fontsize=15)
    axes[0].set_ylabel('phase [mrad]', fontsize=15)
    axes[0].set_xlim(0, 128)

    print '--PLOT/SAVE PHASE SLICE ERRORS VORTEX STATE DISC'
    # Plot phase slices:
    axes[1].plot(x_v[0], dy_v[0], '-k', linewidth=1.5, label='analytic')
    axes[1].plot(x_v[1], dy_v[1], '-r', linewidth=1.5, label='0.5 nm')
    axes[1].plot(x_v[2], dy_v[2], '-m', linewidth=1.5, label='1 nm')
    axes[1].plot(x_v[3], dy_v[3], '-y', linewidth=1.5, label='2 nm')
    axes[1].plot(x_v[4], dy_v[4], '-g', linewidth=1.5, label='4 nm')
    axes[1].plot(x_v[5], dy_v[5], '-c', linewidth=1.5, label='8 nm')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].set_title('Vortex state disc', fontsize=18)
    axes[1].set_xlabel('x [nm]', fontsize=15)
    axes[1].set_ylabel('phase [mrad]', fontsize=15)
    axes[1].set_xlim(0, 128)
    axes[1].legend(loc=4)

    plt.show()
    plt.figtext(0.15, 0.13, 'a)', fontsize=30)
    plt.figtext(0.57, 0.13, 'b)', fontsize=30)
    plt.savefig(directory + '/ch5-1-phase_slice_errors.png', bbox_inches='tight')
    
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

    print '--CALCULATE PHASE DIFFERENCES'
    # Create projections along z-axis:
    projection_disc = pj.simple_axis_projection(mag_data_disc)
    projection_vort = pj.simple_axis_projection(mag_data_vort)
    # Get analytic solutions:
    phase_ana_disc = an.phase_mag_disc(dim, res, phi, center, radius, height)
    phase_ana_vort = an.phase_mag_vortex(dim, res, center, radius, height)
    # Create norm for the plots:
    bounds = np.array([-3, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 3])
    norm = BoundaryNorm(bounds, RdBu.N)
    # Calculations (Disc):
    phase_num_disc = pm.phase_mag_real(res, projection_disc, 'slab')
    phase_diff_disc = PhaseMap(res, (phase_num_disc-phase_ana_disc)*1E3)  # in mrad -> *1000)
    RMS_disc = np.sqrt(np.mean(phase_diff_disc.phase**2))
    # Calculations (Vortex):
    phase_num_vort = pm.phase_mag_real(res, projection_vort, 'slab')
    phase_diff_vort = PhaseMap(res, (phase_num_vort-phase_ana_vort)*1E3)  # in mrad -> *1000
    RMS_vort = np.sqrt(np.mean(phase_diff_vort.phase**2))

    print '--PLOT/SAVE PHASE DIFFERENCES'
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Difference of the real space approach from the analytical solution', fontsize=20)
    # Plot MagData (Disc):
    phase_diff_disc.display('Homog. magn. disc, RMS = {:3.2f} mrad'.format(RMS_disc),
                            labels=('x-axis [nm]', 'y-axis [nm]', '$\Delta$phase [mrad]'),
                            limit=np.max(bounds), norm=norm, axis=axes[0])
    axes[0].set_aspect('equal')
    # Plot MagData (Disc):
    phase_diff_vort.display('Vortex state disc, RMS = {:3.2f} mrad'.format(RMS_vort),
                            labels=('x-axis [nm]', 'y-axis [nm]', '$\Delta$phase [mrad]'),
                            limit=np.max(bounds), norm=norm, axis=axes[1])
    axes[1].set_aspect('equal')
    # Save Plots:
    plt.figtext(0.15, 0.2, 'a)', fontsize=30)
    plt.figtext(0.52, 0.2, 'b)', fontsize=30)
    plt.savefig(directory + '/ch5-1-phase_differences.png', bbox_inches='tight')

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


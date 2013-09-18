# -*- coding: utf-8 -*-
"""Class for the storage of phase data.

This module provides the :class:`~.PhaseMap` class whose instances can be used to store
phase data for a 2-dimensional grid. It is possible to load data from NetCDF4 or LLG (.txt) files
or to save the data in these formats. Also plotting methods are provided. See :class:`~.PhaseMap`
for further information.

"""


import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import netCDF4


class PhaseMap:

    '''Class for storing phase map data.

    Represents 2-dimensional phase maps. the phase information itself is stored in `phase`.
    The dimensions `dim` of the grid with resolution `res` will be calculated at construction
    time, but `res` has to be specified.

    Attributes
    ----------
    res : float
        The resolution of the grid (grid spacing in nm)
    dim : tuple (N=2)
        Dimensions of the grid.
    phase : :class:`~numpy.ndarray` (N=2)
        Matrix containing the phase shift.
    unit : {'rad', 'mrad', 'µrad'}, optional
        Set the unit of the phase map. This is important for the :func:`~.display` function,
        because the phase is scaled accordingly. Does not change the phase itself, which is
        always in `rad`.

    '''

    UNITDICT = {'rad': 1E0,
                'mrad': 1E3,
                'µrad': 1E6}

    def __init__(self, res, phase, unit='rad'):
        '''Constructor for a :class:`~.PhaseMap` object for storing phase data.

        Parameters
        ----------
        res : float
            The resolution of the grid (grid spacing) in nm.
        phase : :class:`~numpy.ndarray` (N=2)
            Matrix containing the phase shift.
        unit : {'rad', 'mrad', 'µrad'}, optional
            Set the unit of the phase map. This is important for the :func:`~.display` function,
            because the phase is scaled accordingly. Does not change the phase itself, which is
            always in `rad`.

        Returns
        -------
        phase_map : :class:`~.PhaseMap`
            The 2D phase shift as a :class:`~.PhaseMap` object.

        '''
        dim = np.shape(phase)
        assert len(dim) == 2, 'Phasemap has to be 2-dimensional!'
        self.res = res
        self.dim = dim
        self.unit = unit
        self.phase = phase

    def set_unit(self, unit):
        '''Set the unit for the phase map.

        Parameters
        ----------
        unit : {'rad', 'mrad'}, optional
            Set the unit of the phase map. This is important for the :func:`~.display` function,
            because the phase is scaled accordingly. Does not change the phase itself, which is
            always in `rad`.

        Returns
        -------
        None

        '''
        assert unit in ['rad', 'mrad']
        self.unit = unit

    @classmethod
    def load_from_txt(cls, filename):
        '''Construct :class:`~.PhaseMap` object from a human readable txt-file.

        Parameters
        ----------
        filename : string
            The name of the file from which to load the data.

        Returns
        -------
        phase_map : :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object containing the loaded data.

        '''
        with open(filename, 'r') as phase_file:
            phase_file.readline()  # Headerline is not used
            res = float(phase_file.readline()[13:-4])
            phase = np.loadtxt(filename, delimiter='\t', skiprows=2)
        return PhaseMap(res, phase)

    def save_to_txt(self, filename='..\output\phasemap_output.txt'):
        '''Save :class:`~.PhaseMap` data in a file with txt-format.

        Parameters
        ----------
        filename : string
            The name of the file in which to store the phase map data.
            The default is 'phasemap_output.txt'.

        Returns
        -------
        None

        '''
        with open(filename, 'w') as phase_file:
            phase_file.write('{}\n'.format(filename.replace('.txt', '')))
            phase_file.write('resolution = {} nm\n'.format(self.res))
            np.savetxt(phase_file, self.phase, fmt='%7.6e', delimiter='\t')

    @classmethod
    def load_from_netcdf4(cls, filename):
        '''Construct :class:`~.PhaseMap` object from NetCDF4-file.

        Parameters
        ----------
        filename : string
            The name of the NetCDF4-file from which to load the data. Standard format is '\*.nc'.

        Returns
        -------
        phase_map: :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object containing the loaded data.

        '''
        phase_file = netCDF4.Dataset(filename, 'r', format='NETCDF4')
        res = phase_file.res
        phase = phase_file.variables['phase'][:]
        phase_file.close()
        return PhaseMap(res, phase)

    def save_to_netcdf4(self, filename='..\output\phasemap_output.nc'):
        '''Save :class:`~.PhaseMap` data in a file with NetCDF4-format.

        Parameters
        ----------
        filename : string, optional
            The name of the NetCDF4-file in which to store the phase data.
            The default is 'phasemap_output.nc'.

        Returns
        -------
        None

        '''
        phase_file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        phase_file.res = self.res
        phase_file.createDimension('v_dim', self.dim[0])
        phase_file.createDimension('u_dim', self.dim[1])
        phase = phase_file.createVariable('phase', 'f', ('v_dim', 'u_dim'))
        phase[:] = self.phase
        print phase_file
        phase_file.close()

    def display(self, title='Phase Map', cmap='RdBu', limit=None, norm=None, axis=None):
        '''Display the phasemap as a colormesh.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Phase Map'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.
        limit : float, optional
            Plotlimit for the phase in both negative and positive direction (symmetric around 0).
            If not specified, the maximum amplitude the phase is used.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
            If not specified, :class:`~matplotlib.colors.Normalize` is automatically used.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.

        Returns
        -------
        None

        '''
        # Take units into consideration:
        phase = self.phase * self.UNITDICT[self.unit]
        if limit is None:
            limit = np.max(np.abs(phase))
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=(8.5, 7))
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        # Plot the phasemap:
        im = axis.pcolormesh(phase, cmap=cmap, vmin=-limit, vmax=limit, norm=norm)
        # Set the axes ticks and labels:
        axis.set_xlim(0, np.shape(phase)[1])
        axis.set_ylim(0, np.shape(phase)[0])
        ticks = (axis.get_xticks()*self.res).astype(int)
        axis.set_xticklabels(ticks)
        ticks = (axis.get_yticks()*self.res).astype(int)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.set_yticklabels(ticks)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel('x [nm]', fontsize=15)
        axis.set_ylabel('y [nm]', fontsize=15)
        # Add colorbar:
        fig = plt.gcf()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('phase shift [{}]'.format(self.unit), fontsize=15)
        # Show plot:
        plt.show()

    def display3d(self, title='Phase Map', cmap='RdBu'):
        '''Display the phasemap as a 3-D surface with contourplots.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Phase Map'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.

        Returns
        -------
        None

        '''
        # Take units into consideration:
        phase = self.phase * self.UNITDICT[self.unit]
        # Create figure and axis:
        fig = plt.figure()
        axis = Axes3D(fig)
        # Plot surface and contours:
        u = range(self.dim[1])
        v = range(self.dim[0])
        uu, vv = np.meshgrid(u, v)
        axis.plot_surface(uu, vv, phase, rstride=4, cstride=4, alpha=0.7, cmap=cmap,
                        linewidth=0, antialiased=False)
        axis.contourf(uu, vv, phase, 15, zdir='z', offset=np.min(phase), cmap=cmap)
        axis.view_init(45, -135)
        axis.set_xlabel('x-axis [px]')
        axis.set_ylabel('y-axis [px]')
        axis.set_zlabel('phase shift [{}]'.format(self.unit))
        # Show Plot:
        plt.show()

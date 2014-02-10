# -*- coding: utf-8 -*-
"""This module provides the :class:`~.PhaseMap` class for storing phase map data."""


import logging

import numpy as np
from numpy import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullLocator, MaxNLocator, FuncFormatter
from PIL import Image

from numbers import Number

import netCDF4


class PhaseMap(object):

    '''Class for storing phase map data.

    Represents 2-dimensional phase maps. The phase information itself is stored as a 2-dimensional
    matrix in `phase`, but can also be accessed as a vector via `phase_vec`. :class:`~.PhaseMap`
    objects support negation, arithmetic operators (``+``, ``-``, ``*``) and their augmented 
    counterparts (``+=``, ``-=``, ``*=``), with numbers and other :class:`~.PhaseMap`
    objects, if their dimensions and grid spacings match. It is possible to load data from NetCDF4
    or textfiles or to save the data in these formats. Methods for plotting the phase or a
    corresponding holographic contour map are provided. Holographic contour maps are created by
    taking the cosine of the (optionally amplified) phase and encoding the direction of the
    2-dimensional gradient via color. The directional encoding can be seen by using the
    :func:`~.make_color_wheel` function. Use the :func:`~.display_combined` function to plot the
    phase map and the holographic contour map next to each other.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim: tuple (N=2)
        Dimensions of the grid.
    phase: :class:`~numpy.ndarray` (N=2)
        Matrix containing the phase shift.
    phase_vec: :class:`~numpy.ndarray` (N=2)
        Vector containing the phase shift.
    unit: {'rad', 'mrad'}, optional
        Set the unit of the phase map. This is important for the :func:`display` function,
        because the phase is scaled accordingly. Does not change the phase itself, which is
        always in `rad`.

    '''

    log = logging.getLogger(__name__)

    UNITDICT = {u'rad': 1E0,
                u'mrad': 1E3,
                u'µrad': 1E6}

    CDICT =     {'red':   [(0.00, 1.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 1.0, 1.0),
                           (0.75, 0.0, 0.0),
                           (1.00, 0.0, 1.0)],

                 'green': [(0.00, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.50, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 1.0)],

                 'blue':  [(0.00, 1.0, 1.0),
                           (0.25, 0.0, 0.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.00, 1.0, 1.0)]}

    CDICT_INV = {'red':   [(0.00, 0.0, 1.0),
                           (0.25, 0.0, 0.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 1.0, 0.0)],

                 'green': [(0.00, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.00, 1.0, 0.0)],

                 'blue':  [(0.00, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 0.0)]}
    
    HOLO_CMAP = mpl.colors.LinearSegmentedColormap('my_colormap', CDICT, 256)
    HOLO_CMAP_INV = mpl.colors.LinearSegmentedColormap('my_colormap', CDICT_INV, 256)
    
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = a
    
    @property
    def dim(self):
        return self._dim

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        assert isinstance(phase, np.ndarray), 'Phase has to be a numpy array!'
        assert len(phase.shape) == 2, 'Phase has to be 2-dimensional!'
        self._phase = phase
        self._dim = phase.shape

    @property
    def phase_vec(self):
        return np.reshape(self.phase, -1)

    @phase_vec.setter
    def phase_vec(self, phase_vec):
        assert isinstance(phase_vec, np.ndarray), 'Vector has to be a numpy array!'
        assert np.size(phase_vec) == np.prod(self.dim), 'Vector size has to match phase!'
        self.phase = phase_vec.reshape(self.dim)

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        assert unit in self.UNITDICT, 'Unit not supported!'
        self._unit = unit

    def __init__(self, a, phase, unit='rad'):
        self.a = a
        self.phase = phase
        self.unit = unit
        self.log = logging.getLogger(__name__)
        self.log.info('Created '+str(self))

    def __repr__(self):
        self.log.info('Calling __repr__')
        return '%s(a=%r, phase=%r, unit=&r)' % \
            (self.__class__, self.a, self.phase, self.unit)

    def __str__(self):
        self.log.info('Calling __str__')
        return 'PhaseMap(a=%s, dim=%s)' % (self.a, self.dim)

    def __neg__(self):  # -self
        self.log.info('Calling __neg__')
        return PhaseMap(self.a, -self.phase, self.unit)
        
    def __add__(self, other):  # self + other
        self.log.info('Calling __add__')
        assert isinstance(other, (PhaseMap, Number)), \
            'Only PhaseMap objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, PhaseMap):
            self.log.info('Adding two PhaseMap objects')
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.phase.shape == self.dim, \
                'Added magnitude has to have the same dimensions!'
            return PhaseMap(self.a, self.phase+other.phase, self.unit)
        else:  # other is a Number
            self.log.info('Adding an offset')
            return PhaseMap(self.a, self.phase+other, self.unit)

    def __sub__(self, other):  # self - other
        self.log.info('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self.log.info('Calling __mul__')
        assert isinstance(other, Number), 'PhaseMap objects can only be multiplied by numbers!'
        return PhaseMap(self.a, other*self.phase, self.unit)

    def __radd__(self, other):  # other + self
        self.log.info('Calling __radd__')
        return self.__add__(other)

    def __rsub__(self, other):  # other - self
        self.log.info('Calling __rsub__')
        return -self.__sub__(other)

    def __rmul__(self, other):  # other * self
        self.log.info('Calling __rmul__')
        return self.__mul__(other)
    
    def __iadd__(self, other):  # self += other
        self.log.info('Calling __iadd__')
        return self.__add__(other)

    def __isub__(self, other):  # self -= other
        self.log.info('Calling __isub__')
        return self.__sub__(other)

    def __imul__(self, other):  # self *= other
        self.log.info('Calling __imul__')
        return self.__mul__(other)

    def save_to_txt(self, filename='..\output\phasemap_output.txt'):
        '''Save :class:`~.PhaseMap` data in a file with txt-format.

        Parameters
        ----------
        filename : string
            The name of the file in which to store the phase map data.
            The default is '..\output\phasemap_output.txt'.

        Returns
        -------
        None

        '''
        self.log.info('Calling save_to_txt')
        with open(filename, 'w') as phase_file:
            phase_file.write('{}\n'.format(filename.replace('.txt', '')))
            phase_file.write('grid spacing = {} nm\n'.format(self.a))
            np.savetxt(phase_file, self.phase, fmt='%7.6e', delimiter='\t')

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
        cls.log.info('Calling load_from_txt')
        with open(filename, 'r') as phase_file:
            phase_file.readline()  # Headerline is not used
            a = float(phase_file.readline()[15:-4])
            phase = np.loadtxt(filename, delimiter='\t', skiprows=2)
        return PhaseMap(a, phase)

    def save_to_netcdf4(self, filename='..\output\phasemap_output.nc'):
        '''Save :class:`~.PhaseMap` data in a file with NetCDF4-format.

        Parameters
        ----------
        filename : string, optional
            The name of the NetCDF4-file in which to store the phase data.
            The default is '..\output\phasemap_output.nc'.

        Returns
        -------
        None

        '''
        self.log.info('Calling save_to_netcdf4')
        phase_file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        phase_file.a = self.a
        phase_file.createDimension('v_dim', self.dim[0])
        phase_file.createDimension('u_dim', self.dim[1])
        phase = phase_file.createVariable('phase', 'f', ('v_dim', 'u_dim'))
        phase[:] = self.phase
        phase_file.close()

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
        cls.log.info('Calling load_from_netcdf4')
        phase_file = netCDF4.Dataset(filename, 'r', format='NETCDF4')
        a = phase_file.a
        phase = phase_file.variables['phase'][:]
        phase_file.close()
        return PhaseMap(a, phase)

    def display_phase(self, title='Phase Map', cmap='RdBu',
                      limit=None, norm=None, axis=None, show=True):
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
        show : bool, optional
            A switch which determines if the plot is shown at the end of plotting.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        '''
        self.log.info('Calling display_phase')
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
        axis.set_xlim(0, self.dim[1])
        axis.set_ylim(0, self.dim[0])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*self.a)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x*self.a)))
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel('x [nm]', fontsize=15)
        axis.set_ylabel('y [nm]', fontsize=15)
        # Add colorbar:
        fig = plt.gcf()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(u'phase shift [{}]'.format(self.unit), fontsize=15)
        # Show plot:
        if show:
            plt.show()
        # Return plotting axis:
        return axis

    def display_phase3d(self, title='Phase Map', cmap='RdBu'):
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
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        '''
        self.log.info('Calling display_phase3d')
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
        # Return plotting axis:
        return axis
    
    def display_holo(self, density=1, title='Holographic Contour Map',
                     axis=None, grad_encode='dark', interpolation='none', show=True):
        '''Display the color coded holography image.
    
        Parameters
        ----------
        density : float, optional
            The gain factor for determining the number of contour lines. The default is 1.
        title : string, optional
            The title of the plot. The default is 'Holographic Contour Map'.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method. No interpolation is used in the default case.
        show: bool, optional
            A switch which determines if the plot is shown at the end of plotting.
    
        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.
    
        '''# TODO: Docstring saturation!
        self.log.info('Calling display_holo')
        # Calculate the holography image intensity:
        img_holo = (1 + np.cos(density * self.phase)) / 2
        # Calculate the phase gradients, expressed by magnitude and angle:
        phase_grad_y, phase_grad_x = np.gradient(self.phase, self.a, self.a)
        phase_angle = (1 - np.arctan2(phase_grad_y, phase_grad_x)/pi) / 2
        phase_magnitude = np.hypot(phase_grad_x, phase_grad_y)
        if phase_magnitude.max() != 0:
            saturation = np.sin(phase_magnitude/phase_magnitude.max() * pi / 2)
            phase_saturation = np.dstack((saturation,)*4)
        # Color code the angle and create the holography image:
        if grad_encode == 'dark':
            rgba = self.HOLO_CMAP(phase_angle)
            rgb = (255.999 * img_holo.T * saturation.T * rgba[:, :, :3].T).T.astype(np.uint8)
        elif grad_encode == 'bright':
            rgba = self.HOLO_CMAP(phase_angle)+(1-phase_saturation)*self.HOLO_CMAP_INV(phase_angle)
            rgb = (255.999 * img_holo.T * rgba[:, :, :3].T).T.astype(np.uint8)
        elif grad_encode == 'color':
            rgba = self.HOLO_CMAP(phase_angle)
            rgb = (255.999 * img_holo.T * rgba[:, :, :3].T).T.astype(np.uint8)
        elif grad_encode == 'none':
            rgba = self.HOLO_CMAP(phase_angle)+self.HOLO_CMAP_INV(phase_angle)
            rgb = (255.999 * img_holo.T * rgba[:, :, :3].T).T.astype(np.uint8)
        else:
            raise AssertionError('Gradient encoding not recognized!') 
        holo_image = Image.fromarray(rgb)
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        # Plot the image on a black background and set axes:
        axis.patch.set_facecolor('black')
        axis.imshow(holo_image, origin='lower', interpolation=interpolation)
        # Set the title and the axes labels:
        axis.set_title(title)
        plt.tick_params(axis='both', which='major', labelsize=14)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel('x-axis [px]', fontsize=15)
        axis.set_ylabel('y-axis [px]', fontsize=15)
        axis.set_xlim(0, self.dim[1])
        axis.set_ylim(0, self.dim[0])
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        # Show Plot:
        if show:
            plt.show()
        # Return plotting axis:
        return axis
    
    def display_combined(self, density=1, title='Combined Plot', interpolation='none',
                         grad_encode='dark'):
        '''Display the phase map and the resulting color coded holography image in one plot.
    
        Parameters
        ----------
        density : float, optional
            The gain factor for determining the number of contour lines. The default is 1.
        title : string, optional
            The title of the plot. The default is 'Combined Plot'.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method for the holographic contour map.
            No interpolation is used in the default case.
    
        Returns
        -------
        phase_axis, holo_axis: :class:`~matplotlib.axes.AxesSubplot`
            The axes on which the graphs are plotted.
    
        '''# TODO: Docstring grad_encode!
        self.log.info('Calling display_combined')
        # Create combined plot and set title:
        fig = plt.figure(figsize=(16, 7))
        fig.suptitle(title, fontsize=20)
        # Plot holography image:
        holo_axis = fig.add_subplot(1, 2, 1, aspect='equal')
        self.display_holo(density=density, axis=holo_axis, interpolation=interpolation,
                          show=False, grad_encode=grad_encode)
        # Plot phase map:
        phase_axis = fig.add_subplot(1, 2, 2, aspect='equal')
        fig.subplots_adjust(right=0.85)
        self.display_phase(axis=phase_axis, show=False)
        plt.show()
        # Return the plotting axes:
        return phase_axis, holo_axis

    @classmethod
    def make_color_wheel(cls):
        '''Display a color wheel to illustrate the color coding of the gradient direction.
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None
    
        '''
        cls.log.info('Calling make_color_wheel')
        x = np.linspace(-256, 256, num=512)
        y = np.linspace(-256, 256, num=512)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx ** 2 + yy ** 2)
        # Create the wheel:
        color_wheel_magnitude = (1 - np.cos(r * pi/360)) / 2
        color_wheel_magnitude *= 0 * (r > 256) + 1 * (r <= 256)
        color_wheel_angle = (1 - np.arctan2(xx, -yy)/pi) / 2
        # Color code the angle and create the holography image:
        rgba = cls.HOLO_CMAP(color_wheel_angle)
        rgb = (255.999 * color_wheel_magnitude.T * rgba[:, :, :3].T).T.astype(np.uint8)
        color_wheel = Image.fromarray(rgb)
        # Plot the color wheel:
        fig = plt.figure(figsize=(4, 4))
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(color_wheel, origin='lower')
        axis.xaxis.set_major_locator(NullLocator())
        axis.yaxis.set_major_locator(NullLocator())
        plt.show()


PhaseMap.make_color_wheel()

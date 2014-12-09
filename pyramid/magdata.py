# -*- coding: utf-8 -*-
"""This module provides the :class:`~.MagData` class for storing of magnetization data."""


import os

import numpy as np
from numpy.linalg import norm
from scipy.ndimage.interpolation import zoom

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.ticker import MaxNLocator

from numbers import Number

import netCDF4

import logging


__all__ = ['MagData']


class MagData(object):

    '''Class for storing magnetization data.

    Represents 3-dimensional magnetic distributions with 3 components which are stored as a
    2-dimensional numpy array in `magnitude`, but which can also be accessed as a vector via
    `mag_vec`. :class:`~.MagData` objects support negation, arithmetic operators
    (``+``, ``-``, ``*``) and their augmented counterparts (``+=``, ``-=``, ``*=``), with numbers
    and other :class:`~.MagData` objects, if their dimensions and grid spacings match. It is
    possible to load data from NetCDF4 or LLG (.txt) files or to save the data in these formats.
    Plotting methods are also provided.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    dim: tuple (N=3)
        Dimensions (z, y, x) of the grid.
    magnitude: :class:`~numpy.ndarray` (N=4)
        The `x`-, `y`- and `z`-component of the magnetization vector for every 3D-gridpoint
        as a 4-dimensional numpy array (first dimension has to be 3, because of the 3 components).
    mag_vec: :class:`~numpy.ndarray` (N=1)
        Vector containing the magnetic distribution.

    '''

    _log = logging.getLogger(__name__+'.MagData')

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = float(a)

    @property
    def dim(self):
        return self._dim

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, magnitude):
        assert isinstance(magnitude, np.ndarray), 'Magnitude has to be a numpy array!'
        assert len(magnitude.shape) == 4, 'Magnitude has to be 4-dimensional!'
        assert magnitude.shape[0] == 3, 'First dimension of the magnitude has to be 3!'
        self._magnitude = np.asarray(magnitude, dtype=np.float32)
        self._dim = magnitude.shape[1:]

    @property
    def mag_vec(self):
        return np.reshape(self.magnitude, -1)

    @mag_vec.setter
    def mag_vec(self, mag_vec):
        assert isinstance(mag_vec, np.ndarray), 'Vector has to be a numpy array!'
        assert np.size(mag_vec) == 3*np.prod(self.dim), \
            'Vector has to match magnitude dimensions! {} {}'.format(mag_vec.shape,
                                                                     3*np.prod(self.dim))
        self.magnitude = mag_vec.reshape((3,)+self.dim)

    def __init__(self, a, magnitude):
        self._log.debug('Calling __init__')
        self.a = a
        self.magnitude = magnitude
        self._log.debug('Created '+str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, magnitude=%r)' % (self.__class__, self.a, self.magnitude)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'MagData(a=%s, dim=%s)' % (self.a, self.dim)

    def __neg__(self):  # -self
        self._log.debug('Calling __neg__')
        return MagData(self.a, -self.magnitude)

    def __add__(self, other):  # self + other
        self._log.debug('Calling __add__')
        assert isinstance(other, (MagData, Number)), \
            'Only MagData objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, MagData):
            self._log.debug('Adding two MagData objects')
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.magnitude.shape == (3,)+self.dim, \
                'Added magnitude has to have the same dimensions!'
            return MagData(self.a, self.magnitude+other.magnitude)
        else:  # other is a Number
            self._log.debug('Adding an offset')
            return MagData(self.a, self.magnitude+other)

    def __sub__(self, other):  # self - other
        self._log.debug('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        assert isinstance(other, Number), 'MagData objects can only be multiplied by numbers!'
        return MagData(self.a, other*self.magnitude)

    def __radd__(self, other):  # other + self
        self._log.debug('Calling __radd__')
        return self.__add__(other)

    def __rsub__(self, other):  # other - self
        self._log.debug('Calling __rsub__')
        return -self.__sub__(other)

    def __rmul__(self, other):  # other * self
        self._log.debug('Calling __rmul__')
        return self.__mul__(other)

    def __iadd__(self, other):  # self += other
        self._log.debug('Calling __iadd__')
        return self.__add__(other)

    def __isub__(self, other):  # self -= other
        self._log.debug('Calling __isub__')
        return self.__sub__(other)

    def __imul__(self, other):  # self *= other
        self._log.debug('Calling __imul__')
        return self.__mul__(other)

    def copy(self):
        '''Returns a copy of the :class:`~.MagData` object

        Parameters
        ----------
        None

        Returns
        -------
        mag_data: :class:`~.MagData`
            A copy of the :class:`~.MagData`.

        '''
        self._log.debug('Calling copy')
        return MagData(self.a, self.magnitude.copy())

    def scale_down(self, n=1):
        '''Scale down the magnetic distribution by averaging over two pixels along each axis.

        Parameters
        ----------
        n : int, optional
            Number of times the magnetic distribution is scaled down. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        Only possible, if each axis length is a power of 2!

        '''
        self._log.debug('Calling scale_down')
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        self.a = self.a * 2**n
        for t in range(n):
            # Pad if necessary:
            pz, py, px = self.dim[0] % 2, self.dim[1] % 2, self.dim[2] % 2
            if pz != 0 or py != 0 or px != 0:
                self.magnitude = np.pad(self.magnitude, ((0, 0), (0, pz), (0, py), (0, px)),
                                        mode='constant')
            # Create coarser grid for the magnetization:
            self.magnitude = self.magnitude.reshape(
                3, self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2).mean(axis=(6, 4, 2))

    def scale_up(self, n=1, order=0):
        '''Scale up the magnetic distribution using spline interpolation of the requested order.

        Parameters
        ----------
        n : int, optional
            Power of 2 with which the grid is scaled. Default is 1, which means every axis is
            increased by a factor of ``2**1 = 2``.
        order : int, optional
            The order of the spline interpolation, which has to be in the range between 0 and 5
            and defaults to 0.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        '''
        self._log.debug('Calling scale_up')
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, (int, long)), \
            'order must be a positive integer between 0 and 5!'
        self.a = self.a / 2**n
        self.magnitude = np.array((zoom(self.magnitude[0], zoom=2**n, order=order),
                                   zoom(self.magnitude[1], zoom=2**n, order=order),
                                   zoom(self.magnitude[2], zoom=2**n, order=order)))

    def pad(self, x_pad, y_pad, z_pad):
        '''Pad the current magnetic distribution with zeros for each individual axis.

        Parameters
        ----------
        x_pad : int
            Number of zeros which should be padded on both sides of the x-axis.
        y_pad : int
            Number of zeros which should be padded on both sides of the y-axis.
        z_pad : int
            Number of zeros which should be padded on both sides of the z-axis.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions accordingly.
        '''
        assert x_pad >= 0 and isinstance(x_pad, (int, long)), 'x_pad must be a positive integer!'
        assert y_pad >= 0 and isinstance(y_pad, (int, long)), 'y_pad must be a positive integer!'
        assert z_pad >= 0 and isinstance(z_pad, (int, long)), 'z_pad must be a positive integer!'
        self.magnitude = np.pad(self.magnitude,
                                ((0, 0), (z_pad, z_pad), (y_pad, y_pad), (x_pad, x_pad)),
                                mode='constant')

    def get_mask(self, threshold=0):
        '''Mask all pixels where the amplitude of the magnetization lies above `threshold`.

        Parameters
        ----------
        threshold : float, optional
            A pixel only gets masked, if it lies above this threshold . The default is 0.

        Returns
        -------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Mask of the pixels where the amplitude of the magnetization lies above `threshold`.

        '''
        return np.sqrt(np.sum(np.array(self.magnitude)**2, axis=0)) > threshold

    def get_vector(self, mask):
        '''Returns the magnetic components arranged in a vector, specified by a mask.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Masks the pixels from which the components should be taken.

        Returns
        -------
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing magnetization components of the specified pixels.
            Order is: first all `x`-, then all `y`-, then all `z`-components.

        '''
        if mask is not None:
            return np.reshape([self.magnitude[0][mask],
                           self.magnitude[1][mask],
                           self.magnitude[2][mask]], -1)
        else:
            return self.mag_vec


    def set_vector(self, vector, mask=None):
        '''Set the magnetic components of the masked pixels to the values specified by `vector`.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean), optional
            Masks the pixels from which the components should be taken.
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing magnetization components of the specified pixels.
            Order is: first all `x`-, then all `y-, then all `z`-components.

        Returns
        -------
        None

        '''
        assert np.size(vector) % 3 == 0, 'Vector has to contain all 3 components for every pixel!'
        count = np.size(vector)/3
        if mask is not None:
            self.magnitude[0][mask] = vector[:count]  # x-component
            self.magnitude[1][mask] = vector[count:2*count]  # y-component
            self.magnitude[2][mask] = vector[2*count:]  # z-component
        else:
            self.mag_vec = vector

    def save_to_llg(self, filename='..\output\magdata_output.txt'):
        '''Save magnetization data in a file with LLG-format.

        Parameters
        ----------
        filename : string, optional
            The name of the LLG-file in which to store the magnetization data.
            The default is '..\output\magdata_output.txt'.

        Returns
        -------
        None

        '''
        self._log.debug('Calling save_to_llg')
        a = self.a * 1.0E-9 / 1.0E-2  # from nm to cm
        # Create 3D meshgrid and reshape it and the magnetization into a list where x varies first:
        zz, yy, xx = np.mgrid[a/2:(self.dim[0]*a-a/2):self.dim[0]*1j,
                              a/2:(self.dim[1]*a-a/2):self.dim[1]*1j,
                              a/2:(self.dim[2]*a-a/2):self.dim[2]*1j].reshape(3, -1)
        x_vec, y_vec, z_vec = self.magnitude.reshape(3, -1)
        # Save data to file:
        data = np.array([xx, yy, zz, x_vec, y_vec, z_vec]).T
        with open(filename, 'w') as mag_file:
            mag_file.write('LLGFileCreator: %s\n' % filename.replace('.txt', ''))
            mag_file.write('    %d    %d    %d\n' % (self.dim[2], self.dim[1], self.dim[0]))
            mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell)
                                          for cell in row) for row in data))

    @classmethod
    def load_from_llg(cls, filename):
        '''Construct :class:`~.MagData` object from LLG-file.

        Parameters
        ----------
        filename : string
            The name of the LLG-file from which to load the data.

        Returns
        -------
        mag_data: :class:`~.MagData`
            A :class:`~.MagData` object containing the loaded data.

        '''
        cls._log.debug('Calling load_from_llg')
        SCALE = 1.0E-9 / 1.0E-2  # From cm to nm
        data = np.genfromtxt(filename, skip_header=2)
        dim = tuple(np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:, 0])))
        a = (data[1, 0] - data[0, 0]) / SCALE
        magnitude = data[:, 3:6].T.reshape((3,)+dim)
        return MagData(a, magnitude)

    def save_to_netcdf4(self, filename='..\output\magdata_output.nc'):
        '''Save magnetization data in a file with NetCDF4-format.

        Parameters
        ----------
        filename : string, optional
            The name of the NetCDF4-file in which to store the magnetization data.
            The default is '..\output\magdata_output.nc'.

        Returns
        -------
        None

        '''
        self._log.debug('Calling save_to_netcdf4')
        mag_file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        mag_file.a = self.a
        mag_file.createDimension('comp', 3)  # Number of components
        mag_file.createDimension('z_dim', self.dim[0])
        mag_file.createDimension('y_dim', self.dim[1])
        mag_file.createDimension('x_dim', self.dim[2])
        magnitude = mag_file.createVariable('magnitude', 'f', ('comp', 'z_dim', 'y_dim', 'x_dim'))
        magnitude[...] = self.magnitude
        mag_file.close()

    @classmethod
    def load_from_netcdf4(cls, filename):
        '''Construct :class:`~.DataMag` object from NetCDF4-file.

        Parameters
        ----------
        filename : string
            The name of the NetCDF4-file from which to load the data. Standard format is '\*.nc'.

        Returns
        -------
        mag_data: :class:`~.MagData`
            A :class:`~.MagData` object containing the loaded data.

        '''
        cls._log.debug('Calling copy')
        mag_file = netCDF4.Dataset(filename, 'r', format='NETCDF4')
        a = mag_file.a
        magnitude = mag_file.variables['magnitude'][...]
        mag_file.close()
        return MagData(a, magnitude)

    def quiver_plot(self, title='Magnetization Distribution', axis=None, proj_axis='z',
                    ar_dens=1, ax_slice=None, log=False, scaled=True):  # TODO: Doc ar_dens
        '''Plot a slice of the magnetization as a quiver plot.

        Parameters
        ----------
        title : string, optional
            The title for the plot.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        log : boolean, optional
            Takes the Default is False.
        scaled : boolean, optional
            Normalizes the plotted arrows in respect to the highest one. Default is True.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        '''
        self._log.debug('Calling quiver_plot')
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = int(self.dim[0]/2.)
            plot_u = np.copy(self.magnitude[0][ax_slice, ...])  # x-component
            plot_v = np.copy(self.magnitude[1][ax_slice, ...])  # y-component
            u_label = 'x [px]'
            v_label = 'y [px]'
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = int(self.dim[1]/2.)
            plot_u = np.copy(self.magnitude[0][:, ax_slice, :])  # x-component
            plot_v = np.copy(self.magnitude[2][:, ax_slice, :])  # z-component
            u_label = 'x [px]'
            v_label = 'z [px]'
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = int(self.dim[2]/2.)
            plot_u = np.swapaxes(np.copy(self.magnitude[2][..., ax_slice]), 0, 1)  # z-component
            plot_v = np.swapaxes(np.copy(self.magnitude[1][..., ax_slice]), 0, 1)  # y-component
            u_label = 'z [px]'
            v_label = 'y [px]'
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=(8.5, 7))
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        angles = np.angle(plot_u+1j*plot_v, deg=True)
        # Take the logarithm of the arrows to clearly show directions (if specified):
        if log:
            cutoff = 10
            amp = np.round(np.hypot(plot_u, plot_v), decimals=cutoff)
            min_value = amp[np.nonzero(amp)].min()
            plot_u = np.round(plot_u, decimals=cutoff) / min_value
            plot_u = np.log10(np.abs(plot_u)+1) * np.sign(plot_u)
            plot_v = np.round(plot_v, decimals=cutoff) / min_value
            plot_v = np.log10(np.abs(plot_v)+1) * np.sign(plot_v)
        # Scale the magnitude of the arrows to the highest one (if specified):
        if scaled:
            plot_u /= np.hypot(plot_u, plot_v).max()
            plot_v /= np.hypot(plot_u, plot_v).max()
        # Setup quiver:
        dim_uv = plot_u.shape
        ad = ar_dens
        xx, yy = np.meshgrid(np.arange(dim_uv[0]), np.arange(dim_uv[1]))
        axis.quiver(xx[::ad, ::ad], yy[::ad, ::ad], plot_u[::ad, ::ad], plot_v[::ad, ::ad],
                    pivot='middle', units='xy', angles=angles[::ad, ::ad],
                    scale_units='xy', scale=1./ad, headwidth=6, headlength=7)
        axis.set_xlim(-1, dim_uv[1])
        axis.set_ylim(-1, dim_uv[0])
        axis.set_title(title, fontsize=18)
        axis.set_xlabel(u_label, fontsize=15)
        axis.set_ylabel(v_label, fontsize=15)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        return axis

    # TODO: Switch with mayavi or combine
    def quiver_plot3d_matplotlib(self, title='Magnetization Distribution', ar_dens=1):  # TODO: Doc ar_dens
        from mpl_toolkits.mplot3d import axes3d  #analysis:ignore
        # TODO: more arguments like in the other plots and document!!!
        a = self.a
        dim = self.dim
        ad = ar_dens
        # Create points and vector components as lists:
        zz, yy, xx = np.mgrid[a/2:(dim[0]*a-a/2):dim[0]*1j,
                              a/2:(dim[1]*a-a/2):dim[1]*1j,
                              a/2:(dim[2]*a-a/2):dim[2]*1j]
        xx = xx[::ad, ::ad, ::ad]
        yy = yy[::ad, ::ad, ::ad]
        zz = zz[::ad, ::ad, ::ad]
        x_mag = self.magnitude[0, ::ad, ::ad, ::ad]
        y_mag = self.magnitude[1, ::ad, ::ad, ::ad]
        z_mag = self.magnitude[2, ::ad, ::ad, ::ad]
        # Plot them as vectors:
        fig = plt.figure(figsize=(8.5, 7))
        axis = fig.add_subplot(1, 1, 1)
        axis = fig.gca(projection='3d')
        axis.quiver(xx, yy, zz, x_mag, y_mag, z_mag)
        axis.set_title(title, fontsize=18)
        # TODO: add colorbar!
#        mlab.colorbar(None, label_fmt='%.2f')
#        mlab.colorbar(None, orientation='vertical')
        return axis

    def quiver_plot3d(self, title='Magnetization Distribution', ar_dens=1):  # TODO: Doc ar_dens
        '''Plot the magnetization as 3D-vectors in a quiverplot.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        self._log.debug('Calling quiver_plot3D')
        from mayavi import mlab
        a = self.a
        dim = self.dim
        ad = ar_dens
        # Create points and vector components as lists:
        zz, yy, xx = np.mgrid[a/2:(dim[0]*a-a/2):dim[0]*1j,
                              a/2:(dim[1]*a-a/2):dim[1]*1j,
                              a/2:(dim[2]*a-a/2):dim[2]*1j]
        xx = xx[::ad, ::ad, ::ad].flatten()#reshape(-1)
        yy = yy[::ad, ::ad, ::ad].flatten()#.reshape(-1)
        zz = zz[::ad, ::ad, ::ad].flatten()#.reshape(-1)
        x_mag = self.magnitude[0][::ad, ::ad, ::ad].flatten()#, (-1))
        y_mag = self.magnitude[1][::ad, ::ad, ::ad].flatten()#, (-1))
        z_mag = self.magnitude[2][::ad, ::ad, ::ad].flatten()#, (-1))
        # Plot them as vectors:
        mlab.figure()
        plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow')
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.title(title, height=0.95, size=0.35)
        mlab.colorbar(None, label_fmt='%.2f')
        mlab.colorbar(None, orientation='vertical')

    def save_to_x3d(self, filename='..\..\output\magdata_output.x3d', maximum=1):
        '''Output the magnetization in the .x3d format for the Fraunhofer InstantReality Player.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        self._log.debug('Calling save_to_x3d')
        from lxml import etree

        dim = self.dim
        # Create points and vector components as lists:
        zz, yy, xx = np.mgrid[0.5:(dim[0]-0.5):dim[0]*1j,
                              0.5:(dim[1]-0.5):dim[1]*1j,
                              0.5:(dim[2]-0.5):dim[2]*1j]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        zz = zz.reshape(-1)
        x_mag = np.reshape(self.magnitude[0], (-1))
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[2], (-1))
        # Load template, load tree and write viewpoint information:
        template = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'template.x3d')
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(template, parser)
        scene = tree.find('Scene')
        etree.SubElement(scene, 'Viewpoint', position='0 0 {}'.format(1.5*dim[0]),
                         fieldOfView='1')
        # Write each "spin"-tag separately:
        for i in range(np.prod(dim)):
            mag = np.sqrt(x_mag[i]**2+y_mag[i]**2+z_mag[i]**2)
            if mag != 0:
                spin_position = (xx[i]-dim[2]/2., yy[i]-dim[1]/2., zz[i]-dim[0]/2.)
                sx_ref = 0
                sy_ref = 1
                sz_ref = 0
                rot_x = sy_ref*z_mag[i] - sz_ref*y_mag[i]
                rot_y = sz_ref*x_mag[i] - sx_ref*z_mag[i]
                rot_z = sx_ref*y_mag[i] - sy_ref*x_mag[i]
                angle = np.arccos(y_mag[i]/mag)
                if norm((rot_x, rot_y, rot_z)) < 1E-10:
                    rot_x, rot_y, rot_z = 1, 0, 0
                spin_rotation = (rot_x, rot_y, rot_z, angle)
                spin_color = cmx.RdYlGn(mag/maximum)[:3]
                spin_scale = (1., 1., 1.)
                spin = etree.SubElement(scene, 'ProtoInstance',
                                        DEF='Spin {}'.format(i), name='Spin_Proto')
                etree.SubElement(spin, 'fieldValue', name='spin_position',
                                 value='{} {} {}'.format(*spin_position))
                etree.SubElement(spin, 'fieldValue', name='spin_rotation',
                                 value='{} {} {} {}'.format(*spin_rotation))
                etree.SubElement(spin, 'fieldValue', name='spin_color',
                                 value='{} {} {}'.format(*spin_color))
                etree.SubElement(spin, 'fieldValue', name='spin_scale',
                                 value='{} {} {}'.format(*spin_scale))
        # Write the tree into the file in pretty print format:
        tree.write(filename, pretty_print=True)

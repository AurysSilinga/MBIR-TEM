# -*- coding: utf-8 -*-
"""Class for the storage of magnetizatin data."""


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mayavi import mlab

from numbers import Number

import netCDF4


class MagData(object):

    '''Class for storing magnetization data.

    Represents 3-dimensional magnetic distributions with 3 components which are stored as a
    2-dimensional numpy array in `magnitude`, but which can also be accessed as a vector via
    `mag_vec`. :class:`~.MagData` objects support arithmetic operators (``+``, ``-``, ``*``, ``/``)
    and their augmented counterparts (``+=``, ``-=``, ``*=``, ``/=``), with numbers and other
    :class:`~.MagData` objects, if their dimensions and grid spacings match. It is possible to load
    data from NetCDF4 or LLG (.txt) files or to save the data in these formats. Plotting methods 
    are also provided.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    dim : tuple (N=3)
        Dimensions (z, y, x) of the grid.
    magnitude : :class:`~numpy.ndarray` (N=4)
        The `x`-, `y`- and `z`-component of the magnetization vector for every 3D-gridpoint
        as a 4-dimensional numpy array (first dimension has to be 3, because of the 3 components).
    mag_vec: :class:`~numpy.ndarray` (N=1)
        Vector containing the magnetic distribution.

    '''

    # TODO: Implement: __str__ or __repr__

    # TODO: Implement: logging

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
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, magnitude):
        assert isinstance(magnitude, np.ndarray), 'Magnitude has to be a numpy array!'
        assert len(magnitude.shape) == 4, 'Magnitude has to be 4-dimensional!'
        assert magnitude.shape[0] == 3, 'First dimension of the magnitude has to be 3!'
        self._magnitude = magnitude
        self._dim = magnitude.shape[1:]
    
    @property
    def mag_vec(self):
        return np.reshape(self.magnitude, -1)

    @mag_vec.setter
    def mag_vec(self, mag_vec):
        assert isinstance(mag_vec, np.ndarray), 'Vector has to be a numpy array!'
        assert np.size(mag_vec) == 3*np.prod(self.dim), 'Vector has to match magnitude dimensions!'
        self.magnitude = mag_vec.reshape((3,)+self.dim)

    def __init__(self, a, magnitude):
        self.a = a
        self.magnitude = magnitude

    def __neg__(self):  # -self
        return MagData(self.a, -self.magnitude)
        
    def __add__(self, other):  # self + other
        assert isinstance(other, (MagData, Number)), \
            'Only MagData objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, MagData):
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.magnitude.shape == (3,)+self.dim, \
                'Added magnitude has to have the same dimensions!'
            return MagData(self.a, self.magnitude+other.magnitude)
        else:  # other is a Number
            return MagData(self.a, self.magnitude+other)

    def __sub__(self, other):  # self - other
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        assert isinstance(other, Number), 'MagData objects can only be multiplied by numbers!'
        return MagData(self.a, other*self.magnitude)

    def __div__(self, other):  # self / other 
        assert other != 0, 'Division by zero!'
        return self.__mul__(1./other)
        # TODO: Does not work, but why?

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __rsub__(self, other):  # other - self
        return -self.__sub__(other)

    def __rmul__(self, other):  # other * self
        return self.__mul__(other)
    
    def __iadd__(self, other):  # self += other
        return self.__add__(other)

    def __isub__(self, other):  # self -= other
        return self.__sub__(other)

    def __imul__(self, other):  # self *= other
        return self.__mul__(other)

    def __idiv__(self, other):  # self /= other
        return self.__div__(other)
    
    def copy(self):
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
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        assert np.all([d % 2**n == 0 for d in self.dim]), 'Dimensions must a multiples of 2!'
        for t in range(n):
            # Create coarser grid for the magnetization:
            self.magnitude = self.magnitude.reshape(
                3, self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2).mean(axis=(6, 4, 2))

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
        mag_file = netCDF4.Dataset(filename, 'r', format='NETCDF4')
        a = mag_file.a
        magnitude =  mag_file.variables['magnitude'][...]
        mag_file.close()
        return MagData(a, magnitude)

    def quiver_plot(self, title='Magnetic Distribution', filename=None, axis=None,
                    proj_axis='z', ax_slice=None):
        '''Plot a slice of the magnetization as a quiver plot.

        Parameters
        ----------
        title : string, optional
            The title for the plot.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        filename : string, optional
            The filename, specifying the location where the image is saved. If not specified,
            the image is shown instead.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        '''
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            if ax_slice is None:
                ax_slice = int(self.dim[0]/2)
            mag_slice_u = self.magnitude[0][ax_slice, ...]  # x-component
            mag_slice_v = self.magnitude[1][ax_slice, ...]  # y-component
            u_label = 'x [px]'
            v_label = 'y [px]'
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            if ax_slice is None:
                ax_slice = int(self.dim[1]/2)
            mag_slice_u = self.magnitude[0][:, ax_slice, :]  # x-component
            mag_slice_v = self.magnitude[2][:, ax_slice, :]  # z-component
            u_label = 'x [px]'
            v_label = 'z [px]'
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            if ax_slice is None:
                ax_slice = int(self.dim[2]/2)
            mag_slice_u = self.magnitude[1][..., ax_slice]  # y-component
            mag_slice_v = self.magnitude[2][..., ax_slice]  # z-component
            u_label = 'y [px]'
            v_label = 'z [px]'
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=(8.5, 7))
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.quiver(mag_slice_u, mag_slice_v, pivot='middle', angles='xy', scale_units='xy',
                   scale=1, headwidth=6, headlength=7)
        axis.set_xlim(-1, np.shape(mag_slice_u)[1])
        axis.set_ylim(-1, np.shape(mag_slice_u)[0])
        axis.set_title(title, fontsize=18)
        axis.set_xlabel(u_label, fontsize=15)
        axis.set_ylabel(v_label, fontsize=15)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
        plt.show()
        return axis

    def quiver_plot3d(self):
        '''Plot the magnetization as 3D-vectors in a quiverplot.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        a = self.a
        dim = self.dim
        # Create points and vector components as lists:
        zz, yy, xx = np.mgrid[a/2:(dim[0]*a-a/2):dim[0]*1j,
                              a/2:(dim[1]*a-a/2):dim[1]*1j,
                              a/2:(dim[2]*a-a/2):dim[2]*1j]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        zz = zz.reshape(-1)
        x_mag = np.reshape(self.magnitude[0], (-1))
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[2], (-1))
        # Plot them as vectors:
        mlab.figure()
        plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow')#, scale_factor=5.0)
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.colorbar()

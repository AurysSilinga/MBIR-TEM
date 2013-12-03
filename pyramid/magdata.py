# -*- coding: utf-8 -*-
"""Class for the storage of magnetizatin data.

This module provides the :class:`~.MagData` class whose instances can be used to store
magnetization distributions with 3 components for a 3-dimensional grid. It is possible to load 
data from NetCDF4 or LLG (.txt) files or to save the data in these formats. Plotting methods
are also provided. See :class:`~.MagData` for further information.

"""


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mayavi import mlab

import netCDF4


class MagData:

    '''Class for storing magnetization data.

    Represents 3-dimensional magnetic distributions with 3 components which are stored as a
    tuple of size 3 in `magnitude`. The object can be created empty by omitting the `magnitude`
    paramter. The `magnitude` can be added later by using the :func:`~.add_magnitude` function.
    This is useful, if the `magnitude` is more complex and more than one magnetized object should
    be represented by the :class:`~.MagData` object, which can be added one after another by the
    :func:`~.add_magnitude` function. The dimensions `dim` of the grid will be set as soon as the
    magnitude is specified. However, the grid spacding `a` has to be always specified at
    construction time.

    Attributes
    ----------
    a : float
        The grid spacing in nm.
    dim : tuple (N=3)
        Dimensions of the grid.
    magnitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
        The `z`-, `y`- and `x`-component of the magnetization vector for every 3D-gridpoint
        as a tuple.

    '''

    def __init__(self, a, magnitude=None):
        '''Constructor for a :class:`~.MagData` object for storing magnetization data.

        Parameters
        ----------
        a : float
            The grid spacing in nm.
        magnitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3), optional
            The `z`-, `y`- and `x`-component of the magnetization vector for every 3D-gridpoint
            as a tuple. Is zero everywhere if not specified.

        '''
        if magnitude is not None:
            dim = np.shape(magnitude[0])
            assert len(dim) == 3, 'Magnitude has to be defined for a 3-dimensional grid!'
            assert np.shape(magnitude[1]) == np.shape(magnitude[2]) == dim, \
                'Dimensions of the magnitude components do not match!'
            self.magnitude = magnitude
            self.dim = dim
        else:
            self.magnitude = None
            self.dim = None
        self.a = a

    def add_magnitude(self, magnitude):
        '''Add a given magnitude to the magnitude of the :class:`~.MagData`.

        Parameters
        ----------
        magnitude : tuple (N=3) of :class:`~numpy.ndarray` (N=3)
            The `z`-, `y`- and `x`-component of the magnetization vector for every 3D-gridpoint
            as a tuple. If the :class:`~.MagData` object already has a `magnitude`, the added
            one has to match the dimensions `dim`, otherwise the added magnitude sets `dim`.

        Returns
        -------
        None

        '''
        if self.magnitude is not None:  # Add magnitude to existing one
            assert np.shape(magnitude) == (3,) + self.dim, \
                'Added magnitude has to have the same dimensions!'
            z_mag, y_mag, x_mag = self.magnitude
            z_new, y_new, x_new = magnitude
            z_mag += z_new
            y_mag += y_new
            x_mag += x_new
            self.magnitude = (z_mag, y_mag, x_mag)
        else:  # Magnitude is set for the first time and the dimensions are calculated
            dim = np.shape(magnitude[0])
            assert len(dim) == 3, 'Magnitude has to be defined for a 3-dimensional grid!'
            assert np.shape(magnitude[1]) == np.shape(magnitude[2]) == dim, \
                'Dimensions of the magnitude components do not match!'
            self.magnitude = magnitude
            self.dim = dim

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
        return np.concatenate([self.magnitude[2][mask],
                               self.magnitude[1][mask],
                               self.magnitude[0][mask]])

    def set_vector(self, mask, vector):
        '''Set the magnetic components of the masked pixels to the values specified by `vector`.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
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
        self.magnitude[2][mask] = vector[:count]  # x-component
        self.magnitude[1][mask] = vector[count:2*count]  # y-component
        self.magnitude[0][mask] = vector[2*count:]  # z-component

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
        assert self.dim[0] % 2**n == 0 and self.dim[1] % 2**n == 0 and self.dim[2] % 2**n == 0, \
            'For downscaling, every dimension must be a multiple of 2!'
        for t in range(n):
            # Create coarser grid for the magnetization:
            z_mag = self.magnitude[0].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            y_mag = self.magnitude[1].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            x_mag = self.magnitude[2].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            self.dim = (self.dim[0]/2, self.dim[1]/2, self.dim[2]/2)
            self.a = self.a * 2
            self.magnitude = (z_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                              y_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                              x_mag.mean(axis=5).mean(axis=3).mean(axis=1))


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
        dim = self.dim
        a = self.a * 1.0E-9 / 1.0E-2  # from nm to cm
        # Create 3D meshgrid and reshape it and the magnetization into a list where x varies first:
        zz, yy, xx = np.mgrid[a/2:(dim[0]*a-a/2):dim[0]*1j,
                              a/2:(dim[1]*a-a/2):dim[1]*1j,
                              a/2:(dim[2]*a-a/2):dim[2]*1j]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        zz = zz.reshape(-1)
        x_mag = np.reshape(self.magnitude[2], (-1))
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        # Save data to file:
        data = np.array([xx, yy, zz, x_mag, y_mag, z_mag]).T
        with open(filename, 'w') as mag_file:
            mag_file.write('LLGFileCreator: %s\n' % filename.replace('.txt', ''))
            mag_file.write('    %d    %d    %d\n' % (dim[2], dim[1], dim[0]))
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
        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, skip_header=1,
                                            skip_footer=len(data[:, 0]))
        a = (data[1, 0] - data[0, 0]) / SCALE
        # Reshape in Python and Igor is different, Python fills rows first, Igor columns:
        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim) for i in range(3, 6)]
        return MagData(a, (z_mag, y_mag, x_mag))

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
        mag_file.createDimension('z_dim', self.dim[0])
        mag_file.createDimension('y_dim', self.dim[1])
        mag_file.createDimension('x_dim', self.dim[2])
        z_mag = mag_file.createVariable('z_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        y_mag = mag_file.createVariable('y_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        x_mag = mag_file.createVariable('x_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        z_mag[:] = self.magnitude[0]
        y_mag[:] = self.magnitude[1]
        x_mag[:] = self.magnitude[2]
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
        z_mag = mag_file.variables['z_mag'][:]
        y_mag = mag_file.variables['y_mag'][:]
        x_mag = mag_file.variables['x_mag'][:]
        mag_file.close()
        return MagData(a, (z_mag, y_mag, x_mag))

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
            mag_slice_u = self.magnitude[2][ax_slice, ...]
            mag_slice_v = self.magnitude[1][ax_slice, ...]
            u_label = 'x [px]'
            v_label = 'y [px]'
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            if ax_slice is None:
                ax_slice = int(self.dim[1]/2)
            mag_slice_u = self.magnitude[2][:, ax_slice, :]
            mag_slice_v = self.magnitude[0][:, ax_slice, :]
            u_label = 'x [px]'
            v_label = 'z [px]'
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            if ax_slice is None:
                ax_slice = int(self.dim[2]/2)
            mag_slice_u = self.magnitude[1][..., ax_slice]
            mag_slice_v = self.magnitude[0][..., ax_slice]
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
        x_mag = np.reshape(self.magnitude[2], (-1))
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        # Plot them as vectors:
        mlab.figure()
        plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow')#, scale_factor=5.0)
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.colorbar()

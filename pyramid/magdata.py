# -*- coding: utf-8 -*-
"""Class for creating objects to store magnetizatin data."""


import numpy as np
import tables.netcdf3 as nc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class MagData:

    '''An object storing magnetization data.'''

    def __init__(self, res, magnitude):  # TODO: electrostatic component!
        '''Constructor for a MagData object for storing magnetization data.
        Arguments:
            res       - the resolution of the grid (grid spacing) in nm
            magnitude - the z-, y- and x-component of the magnetization vector for every
                        3D-gridpoint as a tuple
        Returns:
            MagData object

        '''
        dim = np.shape(magnitude[0])
        assert len(dim) == 3, 'Magnitude has to be defined for a 3-dimensional grid!'
        assert np.shape(magnitude[1]) == np.shape(magnitude[2]) == dim, \
            'Dimensions of the magnitude components do not match!'
        self.res = res
        self.dim = dim
        self.magnitude = magnitude

    def get_vector(self, mask):
        # TODO: DOCSTRING!
        return np.concatenate([self.magnitude[2][mask],
                               self.magnitude[1][mask],
                               self.magnitude[0][mask]])

    def set_vector(self, mask, vector):
        # TODO: DOCSTRING!
        assert np.size(vector) % 3 == 0, 'Vector has to contain all 3 components for every pixel!'
        count = np.size(vector)/3
        self.magnitude[2][mask] = vector[:count]  # x-component
        self.magnitude[1][mask] = vector[count:2*count]  # y-component
        self.magnitude[0][mask] = vector[2*count:]  # z-component
        
    def get_mask(self, threshold=0):
        # TODO: DOCSTRING!
        z_mask = abs(self.magnitude[0]) > threshold
        x_mask = abs(self.magnitude[1]) > threshold
        y_mask = abs(self.magnitude[2]) > threshold
        return np.logical_or(np.logical_or(x_mask, y_mask), z_mask)

    def scale_down(self, n=1):
        # TODO: DOCSTRING!
        # Starting magnetic distribution:
        assert n >= 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        assert self.dim[0] % 2**n == 0 and self.dim[1] % 2**n == 0 and self.dim[2] % 2**n == 0, \
            'For downscaling, every dimension must be a multiple of 2!'    
        for t in range(n):
            # Create coarser grid for the magnetization:
            z_mag = self.magnitude[0].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            y_mag = self.magnitude[1].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            x_mag = self.magnitude[2].reshape(self.dim[0]/2, 2, self.dim[1]/2, 2, self.dim[2]/2, 2)
            self.dim = (self.dim[0]/2, self.dim[1]/2, self.dim[2]/2)
            self.res = self.res * 2
            self.magnitude = (z_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                              y_mag.mean(axis=5).mean(axis=3).mean(axis=1),
                              x_mag.mean(axis=5).mean(axis=3).mean(axis=1))

    @classmethod
    def load_from_llg(cls, filename):
        '''Construct DataMag object from LLG-file (classmethod).
        Arguments:
            filename - the name of the LLG-file from which to load the data
        Returns.
            MagData object

        '''
        scale = 1.0E-9 / 1.0E-2  # From cm to nm
        data = np.genfromtxt(filename, skip_header=2)
        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, skip_header=1,
                                            skip_footer=len(data[:, 0]))
        res = (data[1, 0] - data[0, 0]) / scale
        # Reshape in Python and Igor is different, Python fills rows first, Igor columns:
        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim) for i in range(3, 6)]
        return MagData(res, (z_mag, y_mag, x_mag))

    def save_to_llg(self, filename='magdata_output.txt'):
        '''Save magnetization data in a file with LLG-format.
        Arguments:
            filename - the name of the LLG-file in which to store the magnetization data
                       (default: 'magdata_output.txt')
        Returns:
            None

        '''
        dim = self.dim
        res = self.res * 1.0E-9 / 1.0E-2  # from nm to cm
        # Create 3D meshgrid and reshape it and the magnetization into a list where x varies first:
        zz, yy, xx = np.mgrid[res/2:(dim[0]*res-res/2):dim[0]*1j,
                              res/2:(dim[1]*res-res/2):dim[1]*1j,
                              res/2:(dim[2]*res-res/2):dim[2]*1j]
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
    def load_from_netcdf(cls, filename):
        '''Construct MagData object from a NetCDF-file (classmethod).
        Arguments:
            filename - name of the file from which to load the data
        Returns:
            PhaseMap object

        '''
        f = nc.NetCDFFile(filename, 'r')
        res = getattr(f, 'res')
        z_mag = f.variables['z_mag'].getValue()
        y_mag = f.variables['y_mag'].getValue()
        x_mag = f.variables['x_mag'].getValue()
        f.close()
        return MagData(res, (z_mag, y_mag, x_mag))

    def save_to_netcdf(self, filename='..\output\magdata_output.nc'):
        '''Save magnetization data in a file with NetCDF-format.
        Arguments:
            filename - the name of the file in which to store the phase map data
                       (default: 'phasemap_output.txt')
        Returns:
            None

        '''
        f = nc.NetCDFFile(filename, 'w')
        setattr(f, 'res', self.res)
        f.createDimension('z_dim', self.dim[0])
        f.createDimension('y_dim', self.dim[1])
        f.createDimension('x_dim', self.dim[2])
        z_mag = f.createVariable('z_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        y_mag = f.createVariable('y_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        x_mag = f.createVariable('x_mag', 'f', ('z_dim', 'y_dim', 'x_dim'))
        z_mag[:] = self.magnitude[0]
        y_mag[:] = self.magnitude[1]
        x_mag[:] = self.magnitude[2]
        print f
        f.close()

    def quiver_plot(self, title='Magnetic Distribution)', proj_axis='z', ax_slice=None,
                    filename=None, axis=None): # TODO!!
        '''Plot a slice of the magnetization as a quiver plot.
        Arguments:
            axis     - the axis from which a slice is plotted ('x', 'y' or 'z'), default = 'z'
            ax_slice - the slice-index of the specified axis (optional, if not specified, is
                       set to the center of the specified axis)
            filename - filename, specifying the location where the image is saved (optional, if
                       not specified, image is shown instead)
        Returns:
            None

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
        axis.set_xlim(0, np.shape(mag_slice_u)[1])
        axis.set_ylim(0, np.shape(mag_slice_u)[0])
        axis.set_title(title, fontsize=18)
        axis.set_xlabel(u_label, fontsize=15)
        axis.set_ylabel(v_label, fontsize=15)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        plt.show()

    def quiver_plot3d(self):
        '''3D-Quiver-Plot of the magnetization as vectors.
        Arguments:
            None
        Returns:
            None

        '''
        from mayavi import mlab

        res = self.res
        dim = self.dim
        # Create points and vector components as lists:
        zz, yy, xx = np.mgrid[res/2:(dim[0]*res-res/2):dim[0]*1j,
                              res/2:(dim[1]*res-res/2):dim[1]*1j,
                              res/2:(dim[2]*res-res/2):dim[2]*1j]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        zz = zz.reshape(-1)
        x_mag = np.reshape(self.magnitude[2], (-1))
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        # Plot them as vectors:
        mlab.figure()
        plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow', scale_factor=10.0)
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.colorbar()

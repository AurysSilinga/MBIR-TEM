# -*- coding: utf-8 -*-
"""Class for creating objects to store magnetizatin data."""


import numpy as np
import tables.netcdf3 as nc
import matplotlib.pyplot as plt
from mayavi import mlab


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
        # TODO: Implement!
        return np.concatenate([self.magnitude[2][mask],
                               self.magnitude[1][mask],
                               self.magnitude[0][mask]])

    def set_vector(self, mask, vector):
        # TODO: Implement!
        assert np.size(vector) % 3 == 0, 'Vector has to contain all 3 components for every pixel!'
        count = np.size(vector)/3
        self.magnitude[2][mask] = vector[:count]  # x-component
        self.magnitude[1][mask] = vector[count:2*count]  # y-component
        self.magnitude[0][mask] = vector[2*count:]  # z-component

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

    def quiver_plot(self, axis='z', ax_slice=0):
        '''Plot a slice of the magnetization as a quiver plot.
        Arguments:
            axis     - the axis from which a slice is plotted ('x', 'y' or 'z'), default = 'z'
            ax_slice - the slice-index of the specified axis
        Returns:
            None

        '''
        assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as string).'
        if axis == 'z':  # Slice of the xy-plane with z = ax_slice
            mag_slice_u = self.magnitude[2][ax_slice, ...]
            mag_slice_v = self.magnitude[1][ax_slice, ...]
        elif axis == 'y':  # Slice of the xz-plane with y = ax_slice
            mag_slice_u = self.magnitude[2][:, ax_slice, :]
            mag_slice_v = self.magnitude[0][:, ax_slice, :]
        elif axis == 'x':  # Slice of the yz-plane with x = ax_slice
            mag_slice_u = self.magnitude[1][..., ax_slice]
            mag_slice_v = self.magnitude[0][..., ax_slice]
        # Plot the magnetization vectors:
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.quiver(mag_slice_u, mag_slice_v, pivot='middle', angles='xy', scale_units='xy',
                   scale=1, headwidth=6, headlength=7)

    def quiver_plot3d(self):
        '''3D-Quiver-Plot of the magnetization as vectors.
        Arguments:
            None
        Returns:
            None

        '''
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
        plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow', scale_factor=10.0)
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.colorbar()

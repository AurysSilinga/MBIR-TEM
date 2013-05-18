# -*- coding: utf-8 -*-
"""Class for creating objects to store magnetizatin data."""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


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
        self.res       = res
        self.dim       = dim
        self.magnitude = magnitude
        
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
        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim) 
                               for i in range(3,6)]
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
        zz, yy, xx = np.mgrid[res/2 : (dim[0]*res-res/2) : dim[0]*1j,
                              res/2 : (dim[1]*res-res/2) : dim[1]*1j,
                              res/2 : (dim[2]*res-res/2) : dim[2]*1j]                          
        xx = np.reshape(xx,(-1))
        yy = np.reshape(yy,(-1))
        zz = np.reshape(zz,(-1))
        x_mag = np.reshape(self.magnitude[2], (-1))        
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        # Save data to file:
        data = np.array([xx, yy, zz, x_mag, y_mag, z_mag]).T
        with open(filename,'w') as mag_file:
            mag_file.write('LLGFileCreator2D: %s\n' % filename.replace('.txt', ''))
            mag_file.write('    %d    %d    %d\n' % (dim[2], dim[1], dim[0]))
            mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell) 
                                          for cell in row) for row in data) )
         
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
            mag_slice_1 = self.magnitude[2][ax_slice, ...]
            mag_slice_2 = self.magnitude[1][ax_slice, ...]
        elif axis == 'y': # Slice of the xz-plane with y = ax_slice
            mag_slice_1 = self.magnitude[2][:, ax_slice, :]
            mag_slice_2 = self.magnitude[0][:, ax_slice, :]
        elif axis == 'x': # Slice of the yz-plane with x = ax_slice
            mag_slice_1 = self.magnitude[1][..., ax_slice]
            mag_slice_2 = self.magnitude[0][..., ax_slice]
        # Plot the magnetization vectors:
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.quiver(mag_slice_1, mag_slice_2, pivot='middle', angles='xy', scale_units='xy', 
                   scale=1, headwidth=6, headlength=7)
                                     
    def quiver_plot3d(self):  # XXX: Still buggy, use only for small distributions!
        '''3D-Quiver-Plot of the magnetization as vectors.
        Arguments:
            None
        Returns:
            None
            
        '''
        class Arrow3D(FancyArrowPatch):
            '''Class representing one magnetization vector.'''
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs
            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)[:-1]
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)
        res = self.res
        dim = self.dim
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        ax.set_aspect("equal")
        # Create 3D meshgrid and reshape it and the magnetization into a list where x varies first:   
        zz, yy, xx = np.mgrid[res/2 : (dim[0]*res-res/2) : dim[0]*1j,
                              res/2 : (dim[1]*res-res/2) : dim[1]*1j,
                              res/2 : (dim[2]*res-res/2) : dim[2]*1j]
        xx = np.reshape(xx, (-1))
        yy = np.reshape(yy, (-1))
        zz = np.reshape(zz, (-1))
        x_mag = np.reshape(self.magnitude[2], (-1))        
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        # Add every individual magnetization vector:
        for i in range(np.size(xx)):
            xs = [xx[i] - x_mag[i]*res/2, xx[i] + x_mag[i]*res/2]
            ys = [yy[i] - y_mag[i]*res/2, yy[i] + y_mag[i]*res/2]
            zs = [zz[i] - z_mag[i]*res/2, zz[i] + z_mag[i]*res/2]
            a = Arrow3D(xs, ys, zs, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(a)
        # Rescale the axes and show plot:
        ax.set_xlim3d(0, xx[-1]+res/2)
        ax.set_ylim3d(0, yy[-1]+res/2)
        ax.set_zlim3d(0, zz[-1]+res/2)
        plt.show()
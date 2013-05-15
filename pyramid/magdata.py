# -*- coding: utf-8 -*-
"""Load magnetization data from LLG files."""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class MagData:
    
    '''An object storing magnetization data.'''
    
    
    def __init__(self, res, z_mag, y_mag, x_mag):  # TODO: electrostatic component!
        '''Load magnetization in LLG-file format.
        Arguments:
            filename - the name of the file where the data are stored
        Returns:
            None
        
        '''# TODO: Docstring
        assert np.shape(x_mag) == np.shape(y_mag) == np.shape(z_mag), 'Dimensions do not match!'
        self.res = res
        self.dim = np.shape(x_mag)
        self.magnitude = (z_mag, y_mag, x_mag)
        
        
    @classmethod
    def load_from_llg(self, filename):
        # TODO: Docstring
        scale = 1.0E-9 / 1.0E-2  #from cm to nm
        data = np.genfromtxt(filename, skip_header=2)  
        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, 
                                            skip_header=1, 
                                            skip_footer=len(data[:, 0]))
        res = (data[1, 0] - data[0, 0]) / scale
        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim) 
                               for i in range(3,6)]
        #Reshape in Python and Igor is different, Python fills rows first, Igor columns!
        return MagData(res, z_mag, y_mag, x_mag)


    def save_to_llg(self, filename='output.txt'):
        '''Create homog. magnetization data, saved in a file with LLG-format.
        Arguments:
            dim       - the dimensions of the grid, shape(y, x)
            res       - the resolution of the grid 
                        (real space distance between two points)
            beta      - the angle of the magnetization
            filename  - the name of the file in which to save the data
            mag_shape - an array of shape dim, representing the shape of the magnetic object,
                        a few are supplied in this module
        Returns:
            the the magnetic distribution as a 2D-boolean-array.
            
        ''' # TODO: Renew Docstring
        dim = self.dim
        res = self.res * 1.0E-9 / 1.0E-2  # from nm to cm     
        
        zz, yy, xx = np.mgrid[res/2 : (dim[0]*res-res/2) : dim[0]*1j,
                              res/2 : (dim[1]*res-res/2) : dim[1]*1j,
                              res/2 : (dim[2]*res-res/2) : dim[2]*1j]                          
        
        xx = np.reshape(xx,(-1))
        yy = np.reshape(yy,(-1))
        zz = np.reshape(zz,(-1))
        x_mag = np.reshape(self.magnitude[2], (-1))        
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        
        data = np.array([xx, yy, zz, x_mag, y_mag, z_mag]).T
        with open(filename,'w') as mag_file:
            mag_file.write('LLGFileCreator2D: %s\n' % filename.replace('.txt', ''))
            mag_file.write('    %d    %d    %d\n' % (dim[2], dim[1], dim[0]))
            mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell) 
                                          for cell in row) for row in data) )
         
                                                             
    def quiver_plot(self, axis='z', ax_slice=0):
        # TODO: Docstring
        assert axis == 'z' or axis == 'y' or axis == 'x', 'Axis has to be x, y or z (as string).'
        if axis == 'z':
            mag_slice_1 = self.magnitude[2][ax_slice,...]
            mag_slice_2 = self.magnitude[1][ax_slice,...]
        elif axis == 'y':
            mag_slice_1 = self.magnitude[2][:,ax_slice,:]
            mag_slice_2 = self.magnitude[0][:,ax_slice,:]
        elif axis == 'x':
            mag_slice_1 = self.magnitude[1][...,ax_slice]
            mag_slice_2 = self.magnitude[0][...,ax_slice]
            
        fig = plt.figure()
        fig.add_subplot(111, aspect='equal')
        plt.quiver(mag_slice_1, mag_slice_2, pivot='middle', angles='xy', scale_units='xy', 
                   scale=1, headwidth=6, headlength=7)
                   
                   
    def quiver_plot_3D(self):
        # TODO: Docstring 
        res = self.res
        dim = self.dim
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        
        ax.set_aspect("equal")

        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs
            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)
                
        zz, yy, xx = np.mgrid[res/2 : (dim[0]*res-res/2) : dim[0]*1j,
                              res/2 : (dim[1]*res-res/2) : dim[1]*1j,
                              res/2 : (dim[2]*res-res/2) : dim[2]*1j]
        xx = np.reshape(xx, (-1))
        yy = np.reshape(yy, (-1))
        zz = np.reshape(zz, (-1))
        x_mag = np.reshape(self.magnitude[2], (-1))        
        y_mag = np.reshape(self.magnitude[1], (-1))
        z_mag = np.reshape(self.magnitude[0], (-1))
        
        for i in range(np.size(xx)):
            xs = [xx[i] - x_mag[i]*res/2, xx[i] + x_mag[i]*res/2]
            ys = [yy[i] - y_mag[i]*res/2, yy[i] + y_mag[i]*res/2]
            zs = [zz[i] - z_mag[i]*res/2, zz[i] + z_mag[i]*res/2]
            a = Arrow3D(xs, ys, zs, mutation_scale=10, lw=1, arrowstyle="-|>", color="k")
            ax.add_artist(a)
        ax.set_xlim3d(0, xx[-1]+res/2)
        ax.set_ylim3d(0, yy[-1]+res/2)
        ax.set_zlim3d(0, zz[-1]+res/2)
        plt.show()
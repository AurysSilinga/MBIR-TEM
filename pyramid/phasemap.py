# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:00:57 2013

@author: Jan
"""

class PhaseMap:
    
    '''An object storing magnetization data.'''
    
    
    def __init__(self, phase):  # TODO: more arguments?
        '''Load magnetization in LLG-file format.
        Arguments:
            filename - the name of the file where the data are stored
        Returns:
            None
        
        '''# TODO: Docstring
        self.phase = phase
        
        
    def load_from_file(self, filename): # TODO: Implement
        pass
#        # TODO: Docstring
#        scale = 1.0E-9 / 1.0E-2  #from cm to nm
#        data = np.genfromtxt(filename, skip_header=2)  
#        x_dim, y_dim, z_dim = np.genfromtxt(filename, dtype=int, 
#                                            skip_header=1, 
#                                            skip_footer=len(data[:, 0]))
#        res = (data[1, 0] - data[0, 0]) / scale
#        x_mag, y_mag, z_mag = [data[:, i].reshape(z_dim, y_dim, x_dim) 
#                               for i in range(3,6)]
#        #Reshape in Python and Igor is different, Python fills rows first, Igor columns!
#        return MagData(res, z_mag, y_mag, x_mag)


    def save_to_file(self, filename='output.txt'): # TODO: Implement
        pass
#        '''Create homog. magnetization data, saved in a file with LLG-format.
#        Arguments:
#            dim       - the dimensions of the grid, shape(y, x)
#            res       - the resolution of the grid 
#                        (real space distance between two points)
#            beta      - the angle of the magnetization
#            filename  - the name of the file in which to save the data
#            mag_shape - an array of shape dim, representing the shape of the magnetic object,
#                        a few are supplied in this module
#        Returns:
#            the the magnetic distribution as a 2D-boolean-array.
#            
#        ''' # TODO: Renew Docstring
#        dim = self.dim
#        res = self.res * 1.0E-9 / 1.0E-2  # from nm to cm     
#        
#        zz, yy, xx = np.mgrid[res/2 : (dim[0]*res-res/2) : dim[0]*1j,
#                              res/2 : (dim[1]*res-res/2) : dim[1]*1j,
#                              res/2 : (dim[2]*res-res/2) : dim[2]*1j]                          
#        
#        xx = np.reshape(xx,(-1))
#        yy = np.reshape(yy,(-1))
#        zz = np.reshape(zz,(-1))
#        x_mag = np.reshape(self.magnitude[2], (-1))        
#        y_mag = np.reshape(self.magnitude[1], (-1))
#        z_mag = np.reshape(self.magnitude[0], (-1))
#        
#        data = np.array([xx, yy, zz, x_mag, y_mag, z_mag]).T
#        with open(filename,'w') as mag_file:
#            mag_file.write('LLGFileCreator2D: %s\n' % filename.replace('.txt', ''))
#            mag_file.write('    %d    %d    %d\n' % (dim[2], dim[1], dim[0]))
#            mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell) 
#                                          for cell in row) for row in data) )

    def display_phase(self, res, title, axis=None):
        '''Display the phasemap as a colormesh.
        Arguments:
            phase - the phasemap that should be displayed
            res   - the resolution of the phasemap
            title - the title of the plot
        Returns:
            None
            
        '''
        if axis == None:
            fig = plt.figure()
            axis = fig.add_subplot(1,1,1, aspect='equal')
        
        im = plt.pcolormesh(phase, cmap='gray')
    
        ticks = axis.get_xticks()*res
        axis.set_xticklabels(ticks)
        ticks = axis.get_yticks()*res
        axis.set_yticklabels(ticks)
    
        axis.set_title(title)
        axis.set_xlabel('x-axis [nm]')
        axis.set_ylabel('y-axis [nm]')
        
        fig = plt.gcf()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.show()
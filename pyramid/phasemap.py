# -*- coding: utf-8 -*-
"""Class for creating objects to store phase maps."""


import numpy as np
import matplotlib.pyplot as plt
import tables.netcdf3 as nc


class PhaseMap:

    '''An object storing magnetization data.'''

    def __init__(self, res, phase):
        '''Constructor for a MagData object for storing magnetization data.
        Arguments:
            res       - the resolution of the grid (grid spacing) in nm
            magnitude - the z-, y- and x-component of the magnetization vector for every
                        3D-gridpoint as a tuple
        Returns:
            MagData object

        '''
        dim = np.shape(phase)
        assert len(dim) == 2, 'Phasemap has to be 2-dimensional!'
        self.res = res
        self.dim = dim
        self.phase = phase

    @classmethod
    def load_from_txt(cls, filename):
        '''Construct PhaseMap object from a human readable txt-file (classmethod).
        Arguments:
            filename - name of the file from which to load the data
        Returns.
            PhaseMap object

        '''
        with open(filename, 'r') as f:
            f.readline()  # Headerline is not used
            res = int(f.readline()[13:-4])
            phase = np.loadtxt(filename, delimiter='\t', skiprows=2)
        return PhaseMap(res, phase)

    def save_to_txt(self, filename='..\output\phasemap_output.txt'):
        '''Save PhaseMap data in a file with txt-format.
        Arguments:
            filename - the name of the file in which to store the phase map data
                       (default: 'phasemap_output.txt')
        Returns:
            None

        '''
        with open(filename, 'w') as f:
            f.write('{}\n'.format(filename.replace('.txt', '')))
            f.write('resolution = {} nm\n'.format(self.res))
            np.savetxt(f, self.phase, fmt='%7.6e', delimiter='\t')

    @classmethod
    def load_from_netcdf(cls, filename):
        '''Construct PhaseMap object from a NetCDF-file (classmethod).
        Arguments:
            filename - name of the file from which to load the data
        Returns:
            PhaseMap object

        '''
        f = nc.NetCDFFile(filename, 'r')
        res = getattr(f, 'res')
        phase = f.variables['phase'].getValue()
        f.close()
        return PhaseMap(res, phase)

    def save_to_netcdf(self, filename='..\output\phasemap_output.nc'):
        '''Save PhaseMap data in a file with NetCDF-format.
        Arguments:
            filename - the name of the file in which to store the phase map data
                       (default: 'phasemap_output.txt')
        Returns:
            None

        '''
        f = nc.NetCDFFile(filename, 'w')
        setattr(f, 'res', self.res)
        f.createDimension('v_dim', self.dim[0])
        f.createDimension('u_dim', self.dim[1])
        phase = f.createVariable('phase', 'f', ('v_dim', 'u_dim'))
        phase[:] = self.phase
        f.close()

    def display(self, title='Phase Map', labels=('x-axis [nm]', 'y-axis [nm]', 'phase [rad]'),
                cmap='RdBu', limit=None, norm=None, axis=None):
        '''Display the phasemap as a colormesh.
        Arguments:
            title - the title of the plot (default: 'Phase Map')
            axis  - the axis on which to display the plot (default: None, a new figure is created)
            cmap  - the colormap which is used for the plot (default: 'gray')
        Returns:
            None

        ''' # TODO: Docstring!
        
        # TODO: ALWAYS CENTERED around 0
        if limit is None:
            limit = np.max(np.abs(self.phase))
        
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=(8.5, 7))
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        # Plot the phasemap:
        im = axis.pcolormesh(self.phase, cmap=cmap, vmin=-limit, vmax=limit, norm=norm)
        # Set the axes ticks and labels:
        axis.set_xlim(0, np.shape(self.phase)[1])
        axis.set_ylim(0, np.shape(self.phase)[0])
        ticks = (axis.get_xticks()*self.res).astype(int)
        axis.set_xticklabels(ticks)
        ticks = (axis.get_yticks()*self.res).astype(int)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.set_yticklabels(ticks)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel(labels[0], fontsize=15)
        axis.set_ylabel(labels[1], fontsize=15)
        # Add colorbar:
        fig = plt.gcf()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(labels[2], fontsize=15)
        
        plt.show()

    def display3d(self, title='Phase Map', labels=('x-axis [nm]', 'y-axis [nm]', 'phase [rad]'),
                cmap='RdBu', limit=None, norm=None, axis=None):
        '''Display the phasemap as a colormesh.
        Arguments:
            title - the title of the plot (default: 'Phase Map')
            axis  - the axis on which to display the plot (default: None, a new figure is created)
            cmap  - the colormap which is used for the plot (default: 'gray')
        Returns:
            None

        ''' # TODO: Docstring!
        
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = Axes3D(fig)#.gca(projection='3d')

        u = range(self.dim[1])
        v = range(self.dim[0])
        uu, vv = np.meshgrid(u, v)
        ax.plot_surface(uu, vv, self.phase, rstride=4, cstride=4, alpha=0.7, cmap='RdBu',
                        linewidth=0, antialiased=False)
        ax.contourf(uu, vv, self.phase, 15, zdir='z', offset=np.min(self.phase), cmap='RdBu')
        ax.view_init(45, -135)
        ax.set_xlabel('x-axis [px]')
        ax.set_ylabel('y-axis [px]')
        ax.set_zlabel('phase [mrad]')

        plt.show()

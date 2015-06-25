# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:26:32 2015

@author: Jan
"""


from mayavi import mlab
import numpy as np
import pyramid as py


mag_data = py.MagData.load_from_netcdf4('magdata_mc_orthogonal_thin_vortices.nc')
#mag_data = py.MagData(1, np.ones((3, 2, 2, 2)))
a = mag_data.a
dim = mag_data.dim
magnitude = mag_data.magnitude
ad = 1
limit = 1

byte = 256

def create_8bit_rgb_lut():
    xl = np.mgrid[0:byte, 0:byte, 0:byte]
    lut = np.vstack((xl[0].reshape(1, byte**3),
                     xl[1].reshape(1, byte**3),
                     xl[2].reshape(1, byte**3),
                     255 * np.ones((1, byte**3)))).T
    return lut.astype('int32')


# indexing function to above grid
def rgb_2_scalar_idx(r, g, b):
    print 'r:', r, 'g:', g, 'b:', b
    return byte**2 * r + byte * g + b

# N x 3 colors
colors = np.arange(3*np.prod(dim)).reshape((-1, 3)) % byte
print colors

# N scalars
scalars = np.zeros(np.prod(dim))
for (kp_idx, kp_c) in enumerate(colors):
    print kp_idx, kp_c
    scalars[kp_idx] = rgb_2_scalar_idx(kp_c[0], kp_c[1], kp_c[2])
print scalars

#rgb_lut = create_8bit_rgb_lut()

# Create points and vector components as lists:
zz, yy, xx = (np.indices(dim)-a/2).reshape((3,)+dim)
zz = zz[::ad, ::ad, ::ad].flatten()
yy = yy[::ad, ::ad, ::ad].flatten()
xx = xx[::ad, ::ad, ::ad].flatten()
x_mag = magnitude[0][::ad, ::ad, ::ad].flatten()
y_mag = magnitude[1][::ad, ::ad, ::ad].flatten()
z_mag = magnitude[2][::ad, ::ad, ::ad].flatten()
# Plot them as vectors:
mlab.figure(size=(750, 700))

phis = (1 - np.arctan2(y_mag, x_mag)/np.pi) / 2
thetas = np.arctan2(np.hypot(y_mag, x_mag), z_mag)/np.pi


levels = 15
N = 256
cmap = py.MagData._create_directional_colormap(levels, N)


def angle_to_rgba(phi, theta):
    level = np.floor((1-theta) * levels)
    lookup_value = (level + phi) / levels
    rgba = cmap(lookup_value)
    return tuple((np.asarray(rgba)*255).astype(np.int))

colors = []
for i in range(len(x_mag)):
    colors.append(angle_to_rgba(phis[i], thetas[i]))

plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow', colormap='jet')
from tvtk.api import tvtk
sc=tvtk.UnsignedCharArray()
sc.from_array(colors)

plot.mlab_source.dataset.point_data.scalars=colors
plot.glyph.color_mode = 'color_by_scalar'
plot.mlab_source.dataset.modified()


#plot = mlab.quiver3d(xx, yy, zz, x_mag, y_mag, z_mag, mode='arrow',
#                     colormap='jet')#py.MagData._create_directional_colormap())
#angles = (1 - np.arctan2(y_mag, x_mag)/np.pi) / 2
#plot.mlab_source.dataset.point_data.scalars = angles
#plot.glyph.color_mode = 'color_by_scalar'
#plot.glyph.glyph_source.glyph_position = 'center'
#plot.module_manager.vector_lut_manager.data_range = np.array([0, limit])
mlab.outline(plot)
mlab.axes(plot)
mlab.title('Title', height=0.95, size=0.35)
#mlab.colorbar(label_fmt='%.2f')
#mlab.colorbar(orientation='vertical')
mlab.orientation_axes()
mlab.show_pipeline()



## magic to modify lookup table
#plot.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, rgb_lut.shape[0])
#plot.module_manager.scalar_lut_manager.lut.number_of_colors = rgb_lut.shape[0]
#plot.module_manager.scalar_lut_manager.lut.table = rgb_lut


#color = np.linspace(0, 256, np.prod(dim)).reshape(dim)
#colors = np.asarray((color, color, color))

#nr_points = 6
#x,y,z=np.random.random((3,nr_points)) #some data
#colors=np.random.randint(256,size=(nr_points,3)) #some RGB or RGBA colors
#
#pts=mlab.points3d(x,y,z)
#sc=tvtk.UnsignedCharArray()
#sc.from_array(colors)
#
#
#pts.mlab_source.dataset.point_data.scalars=sc
#pts.mlab_source.dataset.modified()
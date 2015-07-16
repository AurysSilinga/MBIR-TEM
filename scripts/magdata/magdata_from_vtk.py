# -*- coding: utf-8 -*-
"""Create magnetization distributions from vtk-files."""


import os
import imp
import numpy as np
from pylab import griddata
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import vtk
from tqdm import tqdm
import pyramid as py
import logging.config


logging.config.fileConfig(py.LOGGING_CONFIG, disable_existing_loggers=False)

###################################################################################################
filename = 'gyroid_homx.vtk'
b_0 = 1.
###################################################################################################

filepath = os.path.join(py.DIR_FILES, 'vtk', filename)
# Load additional points for the z-slices (for non-convex structures):
configname = filename.replace('.vtk', '_addpoints')
configpath = os.path.join(py.DIR_FILES, 'vtk', configname+'.py')
insert_x, insert_y = [], []
condition = lambda z: False  # If no config file is found!
if os.path.isfile(configpath):
    config = imp.load_source(configname, configpath)
    condition, insert_x, insert_y = config.condition, config.insert_x, config.insert_y

print 'LOAD VTK-DATA!'
# Setting up reader:
reader = vtk.vtkDataSetReader()
reader.SetFileName(filepath)
reader.ReadAllScalarsOn()
reader.ReadAllVectorsOn()
reader.Update()
# Getting output:
output = reader.GetOutput()
# Reading points and vectors:
size = output.GetNumberOfPoints()
vtk_points = output.GetPoints().GetData()
vtk_vectors = output.GetPointData().GetVectors()
# Converting points to numpy array:
point_array = np.zeros(vtk_points.GetSize())
vtk_points.ExportToVoidPointer(point_array)
point_array = np.reshape(point_array, (-1, 3))
# Converting vectors to numpy array:
vector_array = np.zeros(vtk_points.GetSize())
vtk_vectors.ExportToVoidPointer(vector_array)
vector_array = np.reshape(vector_array, (-1, 3))
# Combining data:
data = np.hstack((point_array, vector_array)).astype(py.fft.FLOAT)
# Discard unused stuff:
del reader, output, vtk_points, vtk_vectors, point_array, vector_array
# Scatter plot of all x-y-coordinates
axis = plt.figure().add_subplot(1, 1, 1)
axis.scatter(data[:, 0], data[:, 1])
plt.show()

print 'INTERPOLATE ON REGULAR GRID!'
# Find unique z-slices:
z_uniq = np.unique(data[:, 2])
# Determine the grid spacing:
a = z_uniq[1] - z_uniq[0]
# Determine the size of object:
x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
z_min, z_max = data[:, 2].min(), data[:, 2].max()
x_diff, y_diff, z_diff = np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])
x_cent, y_cent, z_cent = x_min+x_diff/2., y_min+y_diff/2., z_min+z_diff/2.
# Create regular grid:
x = np.arange(x_cent-x_diff, x_cent+x_diff, a)
y = np.arange(y_cent-y_diff, y_cent+y_diff, a)
z = np.arange(z_min, z_max, a)
xx, yy = np.meshgrid(x, y)

def excluding_mesh(x, y, nx=30, ny=30):
    """
    Construct a grid of points, that are some distance away from points (x,
    """

    dx = x.ptp() / nx
    dy = y.ptp() / ny

    xp, yp = np.mgrid[x.min()-2*dx:x.max()+2*dx:(nx+2)*1j,
                      y.min()-2*dy:y.max()+2*dy:(ny+2)*1j]
    xp = xp.ravel()
    yp = yp.ravel()

    # Use KDTree to answer the question: "which point of set (x,y) is the
    # nearest neighbors of those in (xp, yp)"
    tree = KDTree(np.c_[x, y])
    dist, j = tree.query(np.c_[xp, yp], k=1)

    # Select points sufficiently far away
    m = (dist > np.hypot(dx, dy))
    return xp[m], yp[m]

## Prepare fake data points
#xp, yp = excluding_mesh(x, y, nx=35, ny=35)
#zp = np.nan + np.zeros_like(xp)
#
## Grid the data plus fake data points
#xi, yi = np.ogrid[-3:3:350j, -3:3:350j]
#zi = griddata((np.r_[x,xp], np.r_[y,yp]), np.r_[z, zp], (xi, yi),
#              method='linear')
#plt.imshow(zi)
#plt.show()


# Create empty magnitude:
magnitude = np.zeros((3, len(z), len(y), len(x)), dtype=py.fft.FLOAT)
print 'Mag Dimensions:', magnitude.shape[1:]


#import pdb; pdb.set_trace()
# Fill magnitude slice per slice:
for i, zi in tqdm(enumerate(z), total=len(z)):
    # Take all points that lie in one z-voxel of the new regular grid into account (use weights!):
    z_slice = data[np.abs(data[:, 2]-zi) <= a/2., :]
    weights = 1 - np.abs(z_slice[:, 2]-zi)*2/a  # If z is regular everywhere, weights are always 1!

    # Prepare fake data points
    x_nan, y_nan = excluding_mesh(z_slice[:, 0], z_slice[:, 1], nx=len(x)//10, ny=len(y)//10)
    z_nan = np.nan + np.zeros_like(x_nan)


    if i == 10:
        axis = plt.figure().add_subplot(1, 1, 1)
        axis.scatter(x_nan, y_nan)
        plt.show()

    grid_x = np.r_[z_slice[:, 0], x_nan]
    grid_y = np.r_[z_slice[:, 1], y_nan]
    for j in range(3):  # For all 3 components!
#        if condition(zi):  # use insert point if condition is True!
#            grid_x = np.concatenate([z_slice[:, 0], insert_x])
#            grid_y = np.concatenate([z_slice[:, 1], insert_y])
#            grid_z = np.concatenate([weights*z_slice[:, 3+j], np.zeros(len(insert_x))])
#        else:
        grid_z = np.r_[weights*z_slice[:, 3+j], z_nan]
        gridded_subdata = griddata(grid_x, grid_y, grid_z, xx, yy)
        magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0).astype(py.fft.FLOAT)
# Convert a to nm:
a *= 10

print 'CREATE AND SAVE MAGDATA OBJECT!'
mag_data = py.MagData(a, magnitude)
mag_data.save_to_netcdf4('magdata_vtk_{}'.format(filename.replace('.vtk', '.nc')))

py.pm(mag_data).display_combined()
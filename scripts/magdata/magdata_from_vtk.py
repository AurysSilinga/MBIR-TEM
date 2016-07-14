# -*- coding: utf-8 -*-
"""Create magnetization distributions from vtk-files."""

from time import sleep

import pyramid as pr

import matplotlib.pyplot as plt
import numpy as np
import vtk
from pylab import griddata
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm


###################################################################################################
filename = 'tube_90x30x35nm.vtk'
b_0 = 1.
###################################################################################################


def enclosing_zero(x, y, nx=30, ny=30):
    """Construct a grid of points, that are some distance away from points (x, y)

    Parameters
    ----------
    x : int

    y : int

    nx: int, optional

    ny: int, optional

    Returns
    -------
    zero_points: lists
        Two lists for the finer grid in x and y.

    """
    dx = x.ptp() / nx
    dy = y.ptp() / ny
    xp, yp = np.mgrid[x.min() - 2 * dx:x.max() + 2 * dx:(nx + 2) * 1j,
                      y.min() - 2 * dy:y.max() + 2 * dy:(ny + 2) * 1j]
    xp = xp.ravel()
    yp = yp.ravel()
    # Use KDTree to answer the question: "which point of set (x, y) is the
    # nearest neighbors of those in (xp, yp)"
    tree = KDTree(np.c_[x, y])
    dist, j = tree.query(np.c_[xp, yp], k=1)
    # Select points sufficiently far away
    m = (dist > np.hypot(dx, dy))
    return xp[m], yp[m]

print('LOAD VTK-DATA!')
# Setting up reader:
reader = vtk.vtkDataSetReader()
reader.SetFileName(filename)
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
data = np.hstack((point_array, vector_array)).astype(pr.fft.FLOAT)
# Discard unused stuff:
del reader, output, vtk_points, vtk_vectors, point_array, vector_array
# Scatter plot of all x-y-coordinates
axis = plt.figure().add_subplot(1, 1, 1)
axis.scatter(data[:, 0], data[:, 1])

print('INTERPOLATE ON REGULAR GRID!')
# Find unique z-slices:
z_uniq = np.unique(data[:, 2])
# Determine the grid spacing:
a = z_uniq[1] - z_uniq[0]
# Determine the size of object:
x_min, x_max = data[:, 0].min(), data[:, 0].max()
y_min, y_max = data[:, 1].min(), data[:, 1].max()
z_min, z_max = data[:, 2].min(), data[:, 2].max()
x_diff, y_diff, z_diff = np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])
x_cent, y_cent, z_cent = x_min + x_diff / 2., y_min + y_diff / 2., z_min + z_diff / 2.
# Create regular grid:
x = np.arange(x_cent - x_diff, x_cent + x_diff, a)
y = np.arange(y_cent - y_diff, y_cent + y_diff, a)
z = np.arange(z_min, z_max, a)
xx, yy = np.meshgrid(x, y)
# Create empty field:
magnitude = np.zeros((3, len(z), len(y), len(x)), dtype=pr.fft.FLOAT)
print('Mag Dimensions:', magnitude.shape[1:])
sleep(0.5)
# Fill field slice per slice:
for i, zi in tqdm(enumerate(z), total=len(z)):
    # Take all points that lie in one z-voxel of the new regular grid into account (use weights!):
    z_slice = data[np.abs(data[:, 2] - zi) <= a / 2., :]
    # If z is regular everywhere, weights are always 1:
    weights = 1 - np.abs(z_slice[:, 2] - zi) * 2 / a
    # Prepare fake data points
    x_nan, y_nan = enclosing_zero(z_slice[:, 0], z_slice[:, 1], nx=len(x) // 10, ny=len(y) // 10)
    z_nan = np.empty_like(x_nan)
    z_nan[:] = np.nan
    grid_x = np.r_[z_slice[:, 0], x_nan]
    grid_y = np.r_[z_slice[:, 1], y_nan]
    for j in range(3):  # For all 3 components!
        grid_z = np.r_[weights * z_slice[:, 3 + j], z_nan]
        gridded_subdata = griddata(grid_x, grid_y, grid_z, xx, yy)
        magnitude[j, i, :, :] = gridded_subdata.filled(fill_value=0).astype(pr.fft.FLOAT)

print('CREATE AND SAVE MAGDATA OBJECT!')
# Convert a to nm:
a *= 10
mag_data = pr.VectorData(a, magnitude)
mag_data.save_to_hdf5('magdata_vtk_{}'.format(filename.replace('.vtk', '.hdf5')), overwrite=True)
# Plot stuff:
pr.pm(mag_data).display_combined()
plt.show()

# -*- coding: utf-8 -*-
"""Display holography images with the gradient direction encoded in color"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyramid.phasemap import PhaseMap
from numpy import pi
from PIL import Image


CDICT = {'red':   [(0.00, 1.0, 0.0),
                   (0.25, 1.0, 1.0),
                   (0.50, 1.0, 1.0),
                   (0.75, 0.0, 0.0),
                   (1.00, 0.0, 1.0)],

         'green': [(0.00, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.50, 1.0, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.00, 0.0, 1.0)],

         'blue':  [(0.00, 1.0, 1.0),
                   (0.25, 0.0, 0.0),
                   (0.50, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.00, 1.0, 1.0)]}

HOLO_CMAP = mpl.colors.LinearSegmentedColormap('my_colormap', CDICT, 256)


def holo_image(phase_map, density=1):
    '''Returns a holography image with color-encoded gradient direction.
    Arguments:
        phase_map - a PhaseMap object storing the phase informations
        density   - the gain factor for determining the number of contour lines (default: 1)
    Returns:
        holography image

    '''
    assert isinstance(phase_map, PhaseMap), 'phase_map has to be a PhaseMap object!'
    # Calculate the holography image intensity:
    img_holo = (1 + np.cos(density * phase_map.phase * pi/2)) / 2
    # Calculate the phase gradients, expressed by magnitude and angle:
    phase_grad_y, phase_grad_x = np.gradient(phase_map.phase, phase_map.res, phase_map.res)
    phase_angle = (1 - np.arctan2(phase_grad_y, phase_grad_x)/pi) / 2
    phase_magnitude = np.hypot(phase_grad_x, phase_grad_y)
    phase_magnitude = np.sin(phase_magnitude/phase_magnitude.max() * pi / 2)
    # Color code the angle and create the holography image:
    rgba = HOLO_CMAP(phase_angle)
    rgb = (255.999 * img_holo.T * phase_magnitude.T * rgba[:, :, :3].T).T.astype(np.uint8)
    holo_image = Image.fromarray(rgb)
    return holo_image


def make_color_wheel():
    '''Display a color wheel for the gradient direction.
    Arguments:
        None
    Returns:
        None

    '''
    x = np.linspace(-256, 256, num=512)
    y = np.linspace(-256, 256, num=512)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    # Create the wheel:
    color_wheel_magnitude = (1 - np.cos(r * pi/360)) / 2
    color_wheel_magnitude *= 0 * (r > 256) + 1 * (r <= 256)
    color_wheel_angle = (1 - np.arctan2(xx, -yy)/pi) / 2
    # Color code the angle and create the holography image:
    rgba = HOLO_CMAP(color_wheel_angle)
    rgb = (255.999 * color_wheel_magnitude.T * rgba[:, :, :3].T).T.astype(np.uint8)
    color_wheel = Image.fromarray(rgb)
    # Plot the color wheel:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(color_wheel)
    ax.set_title('Color Wheel')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')


def display(holo_image, title='Holography Image', axis=None):
    '''Display the color coded holography image resulting from a given phase map.
    Arguments:
        holo_image - holography image created with the holo_image function of this module
        title      - the title of the plot (default: 'Holography Image')
        axis       - the axis on which to display the plot (default: None, a new figure is created)
    Returns:
        None

    '''
    # If no axis is specified, a new figure is created:
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
    # Plot the image and set axes:
    axis.imshow(holo_image)
    axis.set_title(title)
    axis.set_xlabel('x-axis')
    axis.set_ylabel('y-axis')


def display_combined(phase_map, density, title='Combined Plot'):
    '''Display a given phase map and the resulting color coded holography image in one plot.
    Arguments:
        phase_map - the PhaseMap object from which the holography image is calculated
        density   - the factor for determining the number of contour lines
        title     - the title of the combined plot (default: 'Combined Plot')
    Returns:
        None

    '''
    # Create combined plot and set title:
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(title, fontsize=20)
    # Plot holography image:
    holo_axis = fig.add_subplot(1, 2, 1, aspect='equal')
    display(holo_image(phase_map, density), axis=holo_axis)
    # Plot phase map:
    phase_axis = fig.add_subplot(1, 2, 2, aspect='equal')
    phase_map.display(axis=phase_axis)

# -*- coding: utf-8 -*-
"""Create holographic contour maps from a given phase map.

This module converts phase maps into holographic contour map. This basically means taking the
cosine of the (optionally amplified) phase and encoding the direction of the 2-dimensional
gradient via color. The directional encoding can be seen by using the :func:`~.make_color_wheel`
function. Use the :func:`~.holoimage` function to create these holographic contour maps. It is
possible to use these as input for the :func:`~.display` to plot them, or just pass the
:class:`~pyramid.phasemap.PhaseMap` object to the :func:`~.display_combined` function to plot
the phase map and the holographic contour map next to each other.

"""


import numpy as np
from numpy import pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, MaxNLocator
from PIL import Image

from pyramid.phasemap import PhaseMap


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
    '''Create a holographic contour map from a :class:`~pyramid.phasemap.PhaseMap` object.

    Parameters
    ----------
    phase_map : :class:`~pyramid.phasemap.PhaseMap`
        A :class:`~pyramid.phasemap.PhaseMap` object storing the phase information.
    density : float, optional
        The gain factor for determining the number of contour lines. The default is 1.

    Returns
    -------
    holo_image : :class:`~PIL.image`
        The resulting holographic contour map with color encoded gradient.

    '''
    assert isinstance(phase_map, PhaseMap), 'phase_map has to be a PhaseMap object!'
    # Calculate the holography image intensity:
    img_holo = (1 + np.cos(density * phase_map.phase)) / 2
    # Calculate the phase gradients, expressed by magnitude and angle:
    phase_grad_y, phase_grad_x = np.gradient(phase_map.phase, phase_map.res, phase_map.res)
    phase_angle = (1 - np.arctan2(phase_grad_y, phase_grad_x)/pi) / 2
    phase_magnitude = np.hypot(phase_grad_x, phase_grad_y)
    if phase_magnitude.max() != 0:
        phase_magnitude = np.sin(phase_magnitude/phase_magnitude.max() * pi / 2)
    # Color code the angle and create the holography image:
    rgba = HOLO_CMAP(phase_angle)
    rgb = (255.999 * img_holo.T * phase_magnitude.T * rgba[:, :, :3].T).T.astype(np.uint8)
    holo_image = Image.fromarray(rgb)
    return holo_image


def make_color_wheel():
    '''Display a color wheel to illustrate the color coding of the gradient direction.

    Parameters
    ----------
    None

    Returns
    -------
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
    fig = plt.figure(figsize=(4, 4))
    axis = fig.add_subplot(1, 1, 1, aspect='equal')
    axis.imshow(color_wheel, origin='lower')
    axis.xaxis.set_major_locator(NullLocator())
    axis.yaxis.set_major_locator(NullLocator())


def display(holo_image, title='Holographic Contour Map', axis=None, interpolation='none'):
    '''Display the color coded holography image.

    Parameters
    ----------
    holo_image : :class:`~PIL.image`
        The resulting holographic contour map with color encoded gradient.
    title : string, optional
        The title of the plot. The default is 'Holographic Contour Map'.
    axis : :class:`~matplotlib.axes.AxesSubplot`, optional
        Axis on which the graph is plotted. Creates a new figure if none is specified.
    interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
        Defines the interpolation method. No interpolation is used in the default case.

    Returns
    -------
    axis: :class:`~matplotlib.axes.AxesSubplot`
        The axis on which the graph is plotted.

    '''
    # If no axis is specified, a new figure is created:
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
    # Plot the image and set axes:
    axis.imshow(holo_image, origin='lower', interpolation=interpolation)
    # Set the title and the axes labels:
    axis.set_title(title)
    plt.tick_params(axis='both', which='major', labelsize=14)
    axis.set_title(title, fontsize=18)
    axis.set_xlabel('x-axis [px]', fontsize=15)
    axis.set_ylabel('y-axis [px]', fontsize=15)
    axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))

    return axis


def display_combined(phase_map, density=1, title='Combined Plot', interpolation='none'):
    '''Display a given phase map and the resulting color coded holography image in one plot.

    Parameters
    ----------
    phase_map : :class:`~pyramid.phasemap.PhaseMap`
        A :class:`~pyramid.phasemap.PhaseMap` object storing the phase information.
    density : float, optional
        The gain factor for determining the number of contour lines. The default is 1.
    title : string, optional
        The title of the plot. The default is 'Combined Plot'.
    interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
        Defines the interpolation method for the holographic contour map.
        No interpolation is used in the default case.

    Returns
    -------
    phase_axis, holo_axis: :class:`~matplotlib.axes.AxesSubplot`
        The axes on which the graphs are plotted.

    '''
    # Create combined plot and set title:
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=20)
    # Plot holography image:
    holo_axis = fig.add_subplot(1, 2, 1, aspect='equal')
    display(holo_image(phase_map, density), axis=holo_axis, interpolation=interpolation)
    holo_axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    holo_axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    # Plot phase map:
    phase_axis = fig.add_subplot(1, 2, 2, aspect='equal')
    fig.subplots_adjust(right=0.85)
    phase_map.display(axis=phase_axis)
    phase_axis.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    phase_axis.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    
    return phase_axis, holo_axis

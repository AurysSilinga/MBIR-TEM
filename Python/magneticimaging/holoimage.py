# -*- coding: utf-8 -*-
"""Display holography images with the gradient direction encoded in color"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def holo_image(phase, res, density, title):
    '''Display a holography image with color-encoded gradient direction.
    Arguments:
        phase   - the phasemap that should be displayed
        res     - the resolution of the phasemap
        density - the factor for determining the number of contour lines
        title   - the title of the plot
    Returns:
        None
        
    '''
    img_holo = (1 + np.cos(density * phase * pi/2)) /2
    
    phase_grad_y, phase_grad_x = np.gradient(phase, res, res)
    
    phase_angle = (1 - np.arctan2(phase_grad_y, phase_grad_x)/pi) / 2
    
    phase_magnitude = np.sqrt(phase_grad_x ** 2 + phase_grad_y ** 2)    
    phase_magnitude /= np.amax(phase_magnitude)
    phase_magnitude = np.sin(phase_magnitude * pi / 2)
    
    cmap = HOLO_CMAP
    rgba_img = cmap(phase_angle)
    rgb_img = np.delete(rgba_img, 3, 2)
    red, green, blue = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]    
    red   *= 255.999 * img_holo * phase_magnitude
    green *= 255.999 * img_holo * phase_magnitude
    blue  *= 255.999 * img_holo * phase_magnitude
    rgb = np.dstack((red, green, blue)).astype(np.uint8)
    img = Image.fromarray(rgb)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(img)
    ax.set_title(title + ' - Holography Image')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')


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
    color_wheel_angle = (1 - np.arctan2(xx, -yy)/pi) / 2
        
    r = np.sqrt(xx ** 2 + yy ** 2)
    color_wheel_magnitude = (1 - np.cos(r * pi/360)) / 2
    color_wheel_magnitude *= 0 * (r > 256) + 1 * (r <= 256)
    
    cmap = HOLO_CMAP
    rgba_img = cmap(color_wheel_angle)
    rgb_img = np.delete(rgba_img, 3, 2)
    red, green, blue = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]    
    red   *= 255.999 * color_wheel_magnitude
    green *= 255.999 * color_wheel_magnitude
    blue  *= 255.999 * color_wheel_magnitude
    rgb = np.dstack((red, green, blue)).astype(np.uint8)
    color_wheel = Image.fromarray(rgb)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(color_wheel)
    ax.set_title('Color Wheel')
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
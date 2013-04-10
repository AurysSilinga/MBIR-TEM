# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 13:39:07 2013

@author: Jan
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from numpy import pi
from PIL import Image


def holo_image(phase, res, density, title):
    '''Display the cosine of the phasemap (times a factor) as a colormesh.
    Arguments:
        phase   - the phasemap that should be displayed
        res     - the resolution of the phasemap
        density - the factor for determining the number of contour lines
        title   - the title of the plot
    Returns:
        None
        
    '''
    # TODO: Docstring
    img_holo = (1 + np.cos(density * phase * pi/2)) /2
    
    phase_grad_y, phase_grad_x = np.gradient(phase, res, res)
    
    phase_angle = (1 - np.arctan2(phase_grad_y, phase_grad_x)/pi) / 2 # used for colors
    # TODO: look in semper code for implementation
    
    phase_magnitude = np.sqrt(phase_grad_x ** 2 + phase_grad_y ** 2)    
    phase_magnitude /= np.amax(phase_magnitude)
    phase_magnitude = np.sin(phase_magnitude * pi / 2)
    
    cmap = plt.get_cmap('hsv')

    rgba_img = cmap(phase_angle)
    rgb_img = np.delete(rgba_img, 3, 2)

    red, green, blue = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]    
    red *= 255.999 * img_holo * phase_magnitude
    green *= 255.999 * img_holo * phase_magnitude
    blue *= 255.999 * img_holo * phase_magnitude
    rgb = np.dstack((red,green,blue)).astype(np.uint8)
    
    img = Image.fromarray(rgb)
    
    fig = plt.figure()
    plt.imshow(img)
    #plt.gca().invert_yaxis()    
    
#    cmap = my_cmap#plt.get_cmap('hsv')
#    rgba_img = cmap(color_wheel_angle)
#    rgb_img = np.delete(rgba_img, 3, 2)
#    red, green, blue = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]    
#    red *= 255.999 * color_wheel_magnitude
#    green *= 255.999 * color_wheel_magnitude
#    blue *= 255.999 *color_wheel_magnitude
#    rgb = np.dstack((red,green,blue)).astype(np.uint8)
#    
#    color_wheel = Image.fromarray(rgb)
#    
#    fig = plt.figure()
#    plt.imshow(color_wheel)
#    plt.gca().invert_yaxis()    
#    
#    
#    #rgb_img *= np.dstack((phase_magnitude, phase_magnitude, phase_magnitude))
#    
#    rgb = np.dstack((red,green,blue)).astype(np.uint8)
#    
#    img = Image.fromarray(rgb)
#    img.save('myimg.jpeg')    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.pcolormesh(phase_magnitude, cmap='gray')

    ticks = ax.get_xticks()*res
    ax.set_xticklabels(ticks.astype(int))
    ticks = ax.get_yticks()*res
    ax.set_yticklabels(ticks.astype(int))

    ax.set_title(title+' - magnitude')
    ax.set_xlabel('x-axis [nm]')
    ax.set_ylabel('y-axis [nm]')
    
    plt.colorbar()
    plt.show()    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    plt.pcolormesh(phase_angle, cmap='hsv')

    ticks = ax.get_xticks()*res
    ax.set_xticklabels(ticks.astype(int))
    ticks = ax.get_yticks()*res
    ax.set_yticklabels(ticks.astype(int))

    ax.set_title(title+' - angle')
    ax.set_xlabel('x-axis [nm]')
    ax.set_ylabel('y-axis [nm]')
    
    plt.colorbar()
    plt.show()
    
    pass



cdict = {'red':   [(0.0,  0.0, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'green': [(0.0,  0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  1.0, 1.0)]}

my_cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)


def make_color_wheel():
    
    x = np.linspace(-256, 256, num=512)
    y = np.linspace(-256, 256, num=512)
    xx, yy = np.meshgrid(x, y)
    color_wheel_angle = (1 - np.arctan2(xx, -yy)/pi) / 2
        
    r = np.sqrt(xx ** 2 + yy ** 2)
    color_wheel_magnitude = (1 - np.cos(r * pi/360)) / 2
    color_wheel_magnitude *= 0 * (r > 256) + 1 * (r <= 256)
    
    cmap = plt.get_cmap('hsv')
    rgba_img = cmap(color_wheel_angle)
    rgb_img = np.delete(rgba_img, 3, 2)
    red, green, blue = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]    
    red *= 255.999 * color_wheel_magnitude
    green *= 255.999 * color_wheel_magnitude
    blue *= 255.999 *color_wheel_magnitude
    rgb = np.dstack((red,green,blue)).astype(np.uint8)
    
    color_wheel = Image.fromarray(rgb)
    
    plt.figure()
    plt.imshow(color_wheel)
    #plt.gca().invert_yaxis()
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')   
    #plt.pcolormesh(color_wheel_angle, cmap='hsv')
    #ax.set_title('Colorwheel')
    #plt.colorbar()
    #plt.show()
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')   
    #plt.pcolormesh(color_wheel_magnitude, cmap='gray')
    #ax.set_title('Colorwheel')
    #plt.colorbar()
    #plt.show() 
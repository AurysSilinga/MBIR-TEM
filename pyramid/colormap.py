# -*- coding: utf-8 -*-
"""
Created on Wed Jul 08 15:43:06 2015

@author: Jan
"""

# TODO: Docstrings!


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import logging


__all__ = ['DirectionalColormap']


class DirectionalColormap(mpl.colors.LinearSegmentedColormap):

    _log = logging.getLogger(__name__+'.DirectionalColormap')

    CDICT = {'red': [(0.00, 0.0, 0.0),
                     (0.25, 1.0, 1.0),
                     (0.50, 1.0, 1.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 0.0, 0.0)],

             'green': [(0.00, 1.0, 1.0),
                       (0.25, 1.0, 1.0),
                       (0.50, 0.0, 0.0),
                       (0.75, 0.0, 0.0),
                       (1.00, 1.0, 1.0)],

             'blue': [(0.00, 0.0, 0.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 1.0, 1.0),
                      (1.00, 0.0, 0.0)]}

    CDICT_INV = {'red': [(0.00, 1.0, 1.0),
                         (0.25, 0.0, 0.0),
                         (0.50, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.00, 1.0, 1.0)],

                 'green': [(0.00, 0.0, 0.0),
                           (0.25, 0.0, 0.0),
                           (0.50, 1.0, 1.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 0.0)],

                 'blue': [(0.00, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.50, 1.0, 1.0),
                          (0.75, 0.0, 0.0),
                          (1.00, 1.0, 1.0)]}

    HOLO_CMAP = mpl.colors.LinearSegmentedColormap('my_colormap', CDICT, 256)
    HOLO_CMAP_INV = mpl.colors.LinearSegmentedColormap('my_colormap', CDICT_INV, 256)


    def __init__(self, inverted=False):
        self._log.debug('Calling __create_directional_colormap')
#        r, g, b = [], [], []  # RGB lists
#        # Create saturation lists to encode up and down directions via black and white colors.
#        # example for 5 levels from black (down) to color (in-plane) to white:
#        # pos_sat: [ 0.   0.5  1.   1.   1. ]
#        # neg_sat: [ 0.   0.   0.   0.5  1. ]
#        center = levels//2
#        pos_sat = np.ones(levels)
#        pos_sat[0:center] = [i/center for i in range(center)]
#        neg_sat = np.zeros(levels)
#        neg_sat[center+1:] = [(i+1)/center for i in range(center)]
#
#        # Iterate over all levels (the center level represents in-plane moments!):
#        for i in range(levels):
#            inter_points = np.linspace(i/levels, (i+1)/levels, 5)  # interval points, current level
#            # Red:
#            r.append((inter_points[0], 0, neg_sat[i]))
#            r.append((inter_points[1], pos_sat[i], pos_sat[i]))
#            r.append((inter_points[2], pos_sat[i], pos_sat[i]))
#            r.append((inter_points[3], neg_sat[i], neg_sat[i]))
#            r.append((inter_points[4], neg_sat[i], 0))
#            # Green:
#            g.append((inter_points[0], 0, neg_sat[i]))
#            g.append((inter_points[1], neg_sat[i], neg_sat[i]))
#            g.append((inter_points[2], pos_sat[i], pos_sat[i]))
#            g.append((inter_points[3], pos_sat[i], pos_sat[i]))
#            g.append((inter_points[4], neg_sat[i], 0))
#            # Blue
#            b.append((inter_points[0], 0, pos_sat[i]))
#            b.append((inter_points[1], neg_sat[i], neg_sat[i]))
#            b.append((inter_points[2], neg_sat[i], neg_sat[i]))
#            b.append((inter_points[3], neg_sat[i], neg_sat[i]))
#            b.append((inter_points[4], pos_sat[i], 0))
#        # Combine to color dictionary and return:
#        cdict = {'red': r, 'green': g, 'blue': b}
        cdict = self.CDICT_INV if inverted else self.CDICT
        super(DirectionalColormap, self).__init__('directional_colormap', cdict, N=256)
        self._log.debug('Created '+str(self))

    @classmethod
    def display_colorwheel(cls, mode='white_to_color'):
        '''Display a color wheel to illustrate the color coding of the gradient direction.

        Parameters
        ----------
        None

        Returns
        -------
        None

        ''' # TODO: mode docstring!
        cls._log.debug('Calling display_color_wheel')
        yy, xx = np.indices((512, 512)) - 256
        r = np.hypot(xx, yy)
        # Create the wheel:
        colorind = (1 + np.arctan2(yy, xx)/np.pi) / 2
        saturation = r / 256  # 0 in center, 1 at borders of circle!
        if mode == 'black_to_color':
            pass
        elif mode == 'color_to_black':
            saturation = 1 - saturation
        elif mode == 'white_to_color':
            saturation = 2 - saturation
        elif mode == 'white_to_color_to_black':
            saturation = 2 - 2*saturation
        else:
            raise Exception('Invalid color wheel mode!')
        saturation *= (r <= 256) # TODO: [r<=256]
        rgb = cls.rgb_from_colorind_and_saturation(colorind, saturation)
        color_wheel = Image.fromarray(rgb)
        # Plot the color wheel:
        fig = plt.figure(figsize=(4, 4))
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(color_wheel, origin='lower')
        plt.tick_params(axis='both', which='both', labelleft='off', labelbottom='off',
                        left='off', right='off', top='off', bottom='off')

    @classmethod  # TODO: Docstrings!
    def rgb_from_colorind_and_saturation(cls, colorind, saturation):
        c, s = colorind, saturation
        rgb_norm = (np.minimum(s, np.ones_like(s)).T * cls.HOLO_CMAP(c)[..., :3].T).T
        rgb_invs = (np.maximum(s-1, np.zeros_like(s)).T * cls.HOLO_CMAP_INV(c)[..., :3].T).T
        return (255.999 * (rgb_norm + rgb_invs)).astype(np.uint8)

    @classmethod
    def rgb_from_angles(cls, phi, theta=np.pi/2):
        colorind = (1 + phi/np.pi) / 2
        saturation = 2 * (1 - theta/np.pi)  #  goes from 0 to 2  # TODO: explain!
        return cls.rgb_from_colorind_and_saturation(colorind, saturation)

    @classmethod
    def rgb_from_direction(cls, x, y, z):
        phi = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / (r+1E-30))
        return cls.rgb_from_angles(phi, theta)

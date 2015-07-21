# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides a custom :class:`~.DirectionalColormap` colormap class which has a few
additional functions and can encode three-dimensional directions."""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

import logging


__all__ = ['DirectionalColormap']


class DirectionalColormap(mpl.colors.LinearSegmentedColormap):

    '''Colormap subclass for encoding 3D-directions with colors..

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.LinearSegmentedColormap`
    class with a few classmethods which can be used for convenience. The
    :method:`.display_colorwheel` method can be used to display a colorhweel which is used for
    the directional encoding and three `rgb_from_...` methods are used to calculate rgb tuples
    from 3D-directions, angles or directly from a colorindex and a saturation value. This is
    useful for quiverplots where the arrow colors can be set individually by providing said rgb-
    tuples. Arrows in plane will be encoded with full color saturation and arrow pointing down
    will gradually lose saturation until they are black when pointing down. Arrows pointing up
    will 'oversaturate' (the saturation value will go from 1 up to 2), which in consequence will
    partially add the inverse colormap to the arrows turning them white if they point up (rgb-
    tuple: 255, 255, 255).

    Attributes
    ----------
    inverted: boolean, optional
        Flag which is used to invert the internal colormap (default is False).
        Just influences the classical use as a normal colormap, not the classmethods!

    '''

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
        cdict = self.CDICT_INV if inverted else self.CDICT
        super(DirectionalColormap, self).__init__('directional_colormap', cdict, N=256)
        self._log.debug('Created '+str(self))

    @classmethod
    def display_colorwheel(cls, mode='white_to_color'):
        '''Display a color wheel to illustrate the color coding of the gradient direction.

        Parameters
        ----------
        mode : {'white_to_color', 'color_to_black', 'black_to_color', 'white_to_color_to_black'}
            Optional string for determining the color scheme of the color wheel. Describes the
            order of colors from the center to the outline.

        Returns
        -------
        None

        '''
        cls._log.debug('Calling display_color_wheel')
        yy, xx = np.indices((512, 512)) - 256
        r = np.hypot(xx, yy)
        # Create the wheel:
        colorind = (1 + np.arctan2(yy, xx)/np.pi) / 2
        saturation = r / 256  # 0 in center, 1 at borders of circle!
        if mode == 'black_to_color':
            pass  # no further modification necessary!
        elif mode == 'color_to_black':
            saturation = 1 - saturation
        elif mode == 'white_to_color':
            saturation = 2 - saturation
        elif mode == 'white_to_color_to_black':
            saturation = 2 - 2*saturation
        else:
            raise Exception('Invalid color wheel mode!')
        saturation *= np.where(r <= 256, 1, 0)  # Cut out the wheel!
        rgb = cls.rgb_from_colorind_and_saturation(colorind, saturation)
        color_wheel = Image.fromarray(rgb)
        # Plot the color wheel:
        fig = plt.figure(figsize=(4, 4))
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(color_wheel, origin='lower')
        plt.tick_params(axis='both', which='both', labelleft='off', labelbottom='off',
                        left='off', right='off', top='off', bottom='off')

    @classmethod
    def rgb_from_colorind_and_saturation(cls, colorind, saturation):
        '''Construct a rgb tuple from a colorindex and a saturation value.

        Parameters
        ----------
        colorind : float, [0, 1)
            Colorindex specifying the directional color according to the CDICT colormap.
            The colormap is periodic so that a value of 1 is equivalent to 0 again.
        saturation : [0, 2]float, optional
            Saturation value for the color. The lower hemisphere uses values from 0 to 1 in a
            traditional sense of saturation with no saturation for directions pointing down (black)
            and full saturation in plane (full colors). Higher values (between 1 and 2) add the
            inverse colormap described in CDICT_INV to gradually increase the complementary colors
            so that arrows pointing up appear white.
            Azimuthal angle of the desired direction to encode (in rad). Default: pi/2 (in-plane).

        Returns
        -------
        rgb : tuple (N=3)
            rgb tuple with the encoded direction color.

        Notes
        -----
            Also works with numpy arrays as input! Always returns array of shape (..., 3)!

        '''
        cls._log.debug('Calling rgb_from_colorind_and_saturation')
        c, s = np.ravel(colorind), np.ravel(saturation)
        rgb_norm = (np.minimum(s, np.ones_like(s)).T * cls.HOLO_CMAP(c)[..., :3].T).T
        rgb_invs = (np.maximum(s-1, np.zeros_like(s)).T * cls.HOLO_CMAP_INV(c)[..., :3].T).T
        return (255.999 * (rgb_norm + rgb_invs)).astype(np.uint8)

    @classmethod
    def rgb_from_angles(cls, phi, theta=np.pi/2):
        '''Construct a rgb tuple from two angles representing a 3D direction.

        Parameters
        ----------
        phi : float
            Polar angle of the desired direction to encode (in rad).
        theta : float, optional
            Azimuthal angle of the desired direction to encode (in rad). Default: pi/2 (in-plane).

        Returns
        -------
        rgb : tuple (N=3)
            rgb tuple with the encoded direction color.

        Notes
        -----
            Also works with numpy arrays as input!

        '''
        cls._log.debug('Calling rgb_from_angles')
        colorind = (1 + phi/np.pi) / 2
        saturation = 2 * (1 - theta/np.pi)  # goes from 0 to 2!
        return cls.rgb_from_colorind_and_saturation(colorind, saturation)

    @classmethod
    def rgb_from_direction(cls, x, y, z):
        '''Construct a rgb tuple from three coordinates representing a 3D direction.

        Parameters
        ----------
        x : float
            x-coordinate of the desired direction to encode.
        y : float
            y-coordinate of the desired direction to encode.
        z : float
            z-coordinate of the desired direction to encode.

        Returns
        -------
        rgb : tuple (N=3)
            rgb tuple with the encoded direction color.

        Notes
        -----
            Also works with numpy arrays as input!

        '''
        cls._log.debug('Calling rgb_from_direction')
        phi = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / (r+1E-30))
        return cls.rgb_from_angles(phi, theta)

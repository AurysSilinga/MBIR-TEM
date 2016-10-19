# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides a custom :class:`~.HLSTriadicColormap` colormap class which has a few
additional functions and can encode three-dimensional directions."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import colors
import colorsys

__all__ = ['TransparentColormap', 'HLSTriadicColormap', 'HLSTetradicColormap', 'hls_triadic_cmap',
           'hls_tetradic_cmap', 'plot_colorwheel', 'rgb_from_hls',
           'hls_from_vector', 'rgb_from_vector']
_log = logging.getLogger(__name__)


class TransparentColormap(colors.LinearSegmentedColormap):
    """Colormap subclass for including transparency.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.LinearSegmentedColormap`
    class with integrated support for transparency. The colormap is unicolor and varies only in
    transparency.

    Attributes
    ----------
    r: float, optional
        Intensity of red in the colormap. Has to be between 0. and 1.
    g: float, optional
        Intensity of green in the colormap. Has to be between 0. and 1.
    b: float, optional
        Intensity of blue in the colormap. Has to be between 0. and 1.
    alpha_range : list (N=2) of float, optional
        Start and end alpha value. Has to be between 0. and 1.

    """

    _log = logging.getLogger(__name__ + '.TransparentColormap')

    def __init__(self, r=1., g=0., b=0., alpha_range=None):
        self._log.debug('Calling __init__')
        if alpha_range is None:
            alpha_range = [0., 1.]
        red = [(0., 0., r), (1., r, 1.)]
        green = [(0., 0., g), (1., g, 1.)]
        blue = [(0., 0., b), (1., b, 1.)]
        alpha = [(0., 0., alpha_range[0]), (1., alpha_range[1], 1.)]
        cdict = {'red': red, 'green': green, 'blue': blue, 'alpha': alpha}
        super().__init__('transparent', cdict, N=256)
        self._log.debug('Created ' + str(self))


class HLSTriadicColormap(colors.ListedColormap):
    """Colormap subclass for encoding directions with colors.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.ListedColormap`
    class. The class follows the HSL ('hue', 'saturation', 'lightness') 'Double Hexcone' Model
    with the saturation always set to 1 (moving on the surface of the color
    cylinder) with a luminance of 0.5 (full color). The three prime colors (`rgb`) are spaced
    equidistant with 120° space in between, according to a triadic arrangement.

    """

    _log = logging.getLogger(__name__ + '.HLSTriadicColormap')

    def __init__(self):
        self._log.debug('Calling __init__')
        h = np.linspace(0, 1, 256)
        l = 0.5 * np.ones_like(h)
        s = np.ones_like(h)
        rgb = rgb_from_hls(h, l, s)
        colors = [(ri / 255, gi / 255, bi / 255) for ri, gi, bi in rgb]
        super().__init__(colors, 'hlscm', N=256)
        self._log.debug('Created ' + str(self))


class HLSTetradicColormap(colors.LinearSegmentedColormap):
    """Colormap subclass for encoding directions with colors.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.LinearSegmentedColormap`
    class. The class follows the HSL ('hue', 'saturation', 'lightness') 'Double
    Hexcone' Model with the saturation always set to 1 (moving on the surface of the color
    cylinder) with a luminance of 0.5 (full color). The colors follow a tetradic arrangement with
    four colors (red, green, blue and yellow) arranged with 90° spacing in between.

    """

    _log = logging.getLogger(__name__ + '.HLSTetradicColormap')

    CDICT = {'red': [(0.00, 1.0, 1.0),
                     (0.25, 0.0, 0.0),
                     (0.50, 0.0, 0.0),
                     (0.75, 1.0, 1.0),
                     (1.00, 1.0, 1.0)],

             'green': [(0.00, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 0.0)],

             'blue': [(0.00, 0.0, 0.0),
                      (0.25, 1.0, 1.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 0.0, 0.0)]}

    def __init__(self):
        self._log.debug('Calling __init__')
        super().__init__('holo', self.CDICT, N=256)
        self._log.debug('Created ' + str(self))


def plot_colorwheel(mode='triadic'):
    """Display a color wheel to illustrate the color coding of vector gradient directions.

    Parameters
    ----------
    mode : {'triadic', 'tetradic'}
        Optional string for determining the hue scheme of the color wheel. The default is the
        standard HLS scheme with the three primary colors (red, blue, green) spaced with 120°
        between them (triadic arrangement). The other option is a tetradic scheme commonly
        used in holography where red, green, blue and yellow are used with 90° spacing. In both
        cases, the saturation decreases to the center of the circle.

    Returns
    -------
    None

    """
    _log.debug('Calling display_color_wheel')
    # Construct the colorwheel:
    yy, xx = np.indices((512, 512)) - 256
    rr = np.hypot(xx, yy)
    xx = np.where(rr <= 256, xx, 0)
    yy = np.where(rr <= 256, yy, 0)
    zz = np.where(rr <= 256, 0, -1)
    h, l, s = hls_from_vector(xx, yy, zz)
    rgb = rgb_from_hls(h, l, s, mode=mode)
    color_wheel = Image.fromarray(rgb)
    # Plot the color wheel:
    fig = plt.figure(figsize=(4, 4))
    axis = fig.add_subplot(1, 1, 1, aspect='equal')
    axis.imshow(color_wheel, origin='lower')
    plt.tick_params(axis='both', which='both', labelleft='off', labelbottom='off',
                    left='off', right='off', top='off', bottom='off')
    return


def rgb_from_hls(h, l, s, mode='triadic'):
    """Construct a rgb tuple from hue, luminance and saturation.

    Parameters
    ----------
    h : float, [0, 1)
        Colorindex specifying the directional color according to the `mode` colormap scheme.
        The colormap is periodic so that a value of 1 is equivalent to 0 again.
    l : [0, 1] float
        Luminance or lightness is the radiant power of the color. In the HLS model, it is defined
        as the average between the largest and smallest color component. This definition puts
        the primary (rgb) and secondary (ymc) colors into a plane halfway between black and
        white for a luminance of 0.5. Directions pointing upwards will be encoded in white,
        downwards in black.
    s : [0, 1] float
        The chroma scaled to fill the interval between 0 and 1. This means that smaller
        amplitudes are encoded in grey, while large amplitudes will be in full color.
    mode : {'triadic', 'tetradic'}
        Optional string for determining the hue scheme of the color wheel. The default is the
        standard HLS scheme with the three primary colors (red, blue, green) spaced with 120°
        between them (triadic arrangement). The other option is a tetradic scheme commonly
        used in holography where red, green, blue and yellow are used with 90° spacing. In both
        cases, the saturation decreases to the center of the circle.

    Returns
    -------
    rgb : tuple (N=3)
        rgb tuple with the encoded direction color.

    Notes
    -----
        Also works with numpy arrays as input! Always returns array of shape (..., 3)!

    """
    _log.debug('Calling rgb_from_hls')
    h = np.asarray(h)
    l = np.asarray(l)
    s = np.asarray(s)
    hls_to_rgb = np.vectorize(colorsys.hls_to_rgb)
    rgb_to_hls = np.vectorize(colorsys.rgb_to_hls)
    if mode == 'triadic':
        r, g, b = hls_to_rgb(h, l, s)
    elif mode == 'tetradic':
        rgb = hls_tetradic_cmap(h)
        rh, gh, bh = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        rh = np.asarray(255 * rh).astype(np.uint8)
        gh = np.asarray(255 * gh).astype(np.uint8)
        bh = np.asarray(255 * bh).astype(np.uint8)
        h_standard = rgb_to_hls(rh, gh, bh)[0]
        r, g, b = hls_to_rgb(h_standard, l, s)
    else:
        raise ValueError('Mode not defined!')
    return (255 * np.stack((r, g, b), axis=-1)).astype(np.uint8)


def hls_from_vector(x, y, z):
    """Construct a hls tuple from three coordinates representing a 3D direction.

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
    hls : tuple (N=3)
        hls tuple with the encoded direction color.

    Notes
    -----
        Also works with numpy arrays as input!

    """
    _log.debug('Calling hls_from_vector')
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    phi = np.asarray(np.arctan2(y, x))
    phi[phi < 0] += 2 * np.pi
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / (r + 1E-30))
    h = phi / (2 * np.pi)
    l = 1 - theta / np.pi
    s = r / r.max()
    return h, l, s


def rgb_from_vector(x, y, z, mode='triadic'):
    """Construct a rgb tuple from three coordinates representing a 3D direction.

    Parameters
    ----------
    x : float
        x-coordinate of the desired direction to encode.
    y : float
        y-coordinate of the desired direction to encode.
    z : float
        z-coordinate of the desired direction to encode.
    mode : {'triadic', 'tetradic'}
        Optional string for determining the hue scheme of the color wheel. The default is the
        standard HLS scheme with the three primary colors (red, blue, green) spaced with 120°
        between them (triadic arrangement). The other option is a tetradic scheme commonly
        used in holography where red, green, blue and yellow are used with 90° spacing. In both
        cases, the saturation decreases to the center of the circle.

    Returns
    -------
    rgb : tuple (N=3)
        rgb tuple with the encoded direction color.

    Notes
    -----
        Also works with numpy arrays as input!

    """
    _log.debug('Calling rgb_from_vector')
    return rgb_from_hls(*hls_from_vector(x, y, z), mode=mode)


transparent_cmap = TransparentColormap(0.2, 0.3, 0.2, [0.75, 0.])
hls_tetradic_cmap = HLSTetradicColormap()
hls_triadic_cmap = HLSTriadicColormap()

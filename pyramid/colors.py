# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides a number of custom colormaps, which also have capabilities for 3D plotting.
If this is the case, the :class:`~.Colormap3D` colormap class is a parent class. In `cmaps`, a
number of specialised colormaps is available for convenience. If the default for angular colormaps
(used for 3D plotting) should be changed, set it via `CMAP_CMAP_ANGULAR_DEFAULT`.
For general questions about colors see:
http://www.poynton.com/PDFs/GammaFAQ.pdf
http://www.poynton.com/PDFs/ColorFAQ.pdf
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import colors

import colorsys

import abc

__all__ = ['ColormapCubehelix', 'ColormapPerception', 'ColormapHLS', 'ColormapClassic',
           'ColormapTransparent', 'cmaps', 'CMAP_ANGULAR_DEFAULT']
_log = logging.getLogger(__name__)


class Colormap3D(colors.Colormap, metaclass=abc.ABCMeta):
    """Colormap subclass for encoding directions with colors.

    This abstract class is used as a superclass/interface for 3D vector plotting capabilities.
    In general, a circular colormap should be used to encode the in-plane angle (hue). The
    perpendicular angle is encoded via luminance variation (up: white, down: black). Finally,
    the length of a vector is encoded via saturation. Decreasing vector length causes a desaturated
    color. Subclassing colormaps get access to routines to plot a colorwheel (which should
    ideally be located in the 50% luminance plane, which depends strongly on the underlying map),
    a convenience function to interpolate color tuples and a function to return rgb triples for a
    given vector. The :class:`~.Colormap3D` class itself subclasses the matplotlib base colormap.

    """

    _log = logging.getLogger(__name__ + '.Colormap3D')

    @staticmethod
    def interpolate_color(fraction, start, end):
        """Interpolate linearly between two color tuples (e.g. RGB).

        Parameters
        ----------
        fraction: float or :class:`~numpy.ndarray`
            Interpolation fraction between 0 and 1, which determines the position of the
            interpolation between `start` and `end`.
        start: tuple (N=3) or :class:`~numpy.ndarray`
            Start of the interpolation as a tuple of three numbers or a numpy array, where the last
            dimension should have length 3 and contain the color tuples.
        end: tuple (N=3) or :class:`~numpy.ndarray`
            End of the interpolation as a tuple of three numbers or a numpy array, where the last
            dimension should have length 3 and contain the color tuples.

        Returns
        -------
        result: tuple (N=3) or :class:`~numpy.ndarray`
            Result of the interpolation as a tuple of three numbers or a numpy array, where the
            last dimension should has length 3 and contains the color tuples.

        """
        _log.debug('Calling interpolate_color')
        start, end = np.asarray(start), np.asarray(end)
        r1 = start[..., 0] + (end[..., 0] - start[..., 0]) * fraction
        r2 = start[..., 1] + (end[..., 1] - start[..., 1]) * fraction
        r3 = start[..., 2] + (end[..., 2] - start[..., 2]) * fraction
        return r1, r2, r3

    def rgb_from_vector(self, vector):
        """Construct a hls tuple from three coordinates representing a 3D direction.

        Parameters
        ----------
        vector: tuple (N=3) or :class:`~numpy.ndarray`
            Vector containing the x, y and z component, or a numpy array encompassing the
            components as three lists.z-coordinate of the desired direction to encode.

        Returns
        -------
        rgb:  :class:`~numpy.ndarray`
            Numpy array containing the calculated color tuples.

        """
        self._log.debug('Calling rgb_from_vector')
        x, y, z = np.asarray(vector)
        # Calculate spherical coordinates:
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        phi = np.asarray(np.arctan2(y, x))
        phi[phi < 0] += 2 * np.pi
        theta = np.arccos(z / (r + 1E-30))
        # Calculate color deterministics:
        hue = phi / (2 * np.pi)
        lum = 1 - theta / np.pi
        sat = r / r.max()
        # Calculate RGB from hue with colormap:
        rgba = np.asarray(self(hue))
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        # Interpolate saturation:
        r, g, b = self.interpolate_color(sat, (0.5, 0.5, 0.5), np.stack((r, g, b), axis=-1))
        # Interpolate luminance:
        lum_target = np.where(lum < 0.5, 0, 1)
        lum_target = np.stack([lum_target] * 3, axis=-1)
        fraction = np.where(lum < 0.5, 1 - 2 * lum, 2 * (lum - 0.5))
        r, g, b = self.interpolate_color(fraction, np.stack((r, g, b), axis=-1), lum_target)
        # Return RGB:
        return np.asarray(255 * np.stack((r, g, b), axis=-1), dtype=np.uint8)

    def plot_colorwheel(self, figsize=(4, 4)):
        """Display a color wheel to illustrate the color coding of vector gradient directions.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_colorwheel')
        # Construct the colorwheel:
        yy, xx = (np.indices((512, 512)) - 256) / 256
        rr = np.hypot(xx, yy)
        xx = np.where(rr <= 1, xx, 0)
        yy = np.where(rr <= 1, yy, 0)
        zz = np.where(rr <= 1, 0, -1)
        # Create color wheel:
        color_wheel = Image.fromarray(self.rgb_from_vector(np.asarray((xx, yy, zz))))
        # Plot the color wheel:
        fig = plt.figure(figsize=figsize)
        axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(color_wheel, origin='lower')
        plt.tick_params(axis='both', which='both', labelleft='off', labelbottom='off',
                        left='off', right='off', top='off', bottom='off')
        # Return axis:
        return axis


class ColormapCubehelix(colors.LinearSegmentedColormap, Colormap3D):
    """A full implementation of Dave Green's "cubehelix" for Matplotlib.

    Based on the FORTRAN 77 code provided in D.A. Green, 2011, BASI, 39, 289.
    http://adsabs.harvard.edu/abs/2011arXiv1108.5083G
    Also see:
    http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
    http://davidjohnstone.net/pages/cubehelix-gradient-picker
    User can adjust all parameters of the cubehelix algorithm. This enables much greater
    flexibility in choosing color maps. Default color map settings produce the standard cubehelix.
    Create color map in only blues by setting rot=0 and start=0. Create reverse (white to black)
    backwards through the rainbow once by setting rot=1 and reverse=True, etc. Furthermore, the
    algorithm was tuned, so that constant luminance values can be used (e.g. to create a truly
    isoluminant colorwheel). The `rot` parameter is also tuned to hold true for these cases.
    Of the here presented colorwheels, only this one manages to solely navigate through the L*=50
    plane, which can be seen here:
    https://upload.wikimedia.org/wikipedia/commons/2/21/Lab_color_space.png

    Parameters
    ----------
    start : scalar, optional
        Sets the starting position in the color space. 0=blue, 1=red,
        2=green. Defaults to 0.5.
    rot : scalar, optional
        The number of rotations through the rainbow. Can be positive
        or negative, indicating direction of rainbow. Negative values
        correspond to Blue->Red direction. Defaults to -1.5.
    gamma : scalar, optional
        The gamma correction for intensity. Defaults to 1.0.
    reverse : boolean, optional
        Set to True to reverse the color map. Will go from black to
        white. Good for density plots where shade~density. Defaults to False.
    nlev : scalar, optional
        Defines the number of discrete levels to render colors at.
        Defaults to 256.
    sat : scalar, optional
        The saturation intensity factor. Defaults to 1.2
        NOTE: this was formerly known as `hue` parameter
    minSat : scalar, optional
        Sets the minimum-level saturation. Defaults to 1.2.
    maxSat : scalar, optional
        Sets the maximum-level saturation. Defaults to 1.2.
    startHue : scalar, optional
        Sets the starting color, ranging from [0, 360], as in
        D3 version by @mbostock.
        NOTE: overrides values in start parameter.
    endHue : scalar, optional
        Sets the ending color, ranging from [0, 360], as in
        D3 version by @mbostock
        NOTE: overrides values in rot parameter.
    minLight : scalar, optional
        Sets the minimum lightness value. Defaults to 0.
    maxLight : scalar, optional
        Sets the maximum lightness value. Defaults to 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap object

    Revisions
    ---------
    2014-04 (@jradavenport) Ported from IDL version
    2014-04 (@jradavenport) Added kwargs to enable similar to D3 version,
                            changed name of `hue` parameter to `sat`.
    2016-11 (@jan.caron) Added support for isoluminant cubehelices while making
                         sure `rot` works as intended.
    """

    _log = logging.getLogger(__name__ + '.ColormapCubehelix')

    def __init__(self, start=0.5, rot=-1.5, gamma=1.0, reverse=False, nlev=256,
                 minSat=1.2, maxSat=1.2, minLight=0., maxLight=1., **kwargs):
        self._log.debug('Calling __init__')
        # Override start and rot if startHue and endHue are set:
        if kwargs is not None:
            if 'startHue' in kwargs:
                start = (kwargs.get('startHue') / 360. - 1.) * 3.
            if 'endHue' in kwargs:
                rot = kwargs.get('endHue') / 360. - start / 3. - 1.
            if 'sat' in kwargs:
                minSat = kwargs.get('sat')
                maxSat = kwargs.get('sat')
        self.nlev = nlev
        # Set up the parameters:
        maxLight += 1E-10  # For the edge case that minLight == maxLight!
        self.fract = np.linspace(minLight, maxLight, nlev)
        dL = maxLight - minLight  # Used to scale `fract`, so that # of rotations stays the same!
        angle = 2.0 * np.pi * (start / 3.0 + rot * self.fract / dL + 1.)
        self.fract = self.fract**gamma
        satar = np.linspace(minSat, maxSat, nlev)
        amp = np.asarray(satar * self.fract * (1. - self.fract) / 2)
        # Compute the RGB vectors according to main equations:
        self.red = self.fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
        self.grn = self.fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
        self.blu = self.fract + amp * (1.97294 * np.cos(angle))
        # Find where RBB are outside the range [0,1], clip:
        self.red[np.where((self.red > 1.))] = 1.
        self.grn[np.where((self.grn > 1.))] = 1.
        self.blu[np.where((self.blu > 1.))] = 1.
        self.red[np.where((self.red < 0.))] = 0.
        self.grn[np.where((self.grn < 0.))] = 0.
        self.blu[np.where((self.blu < 0.))] = 0.
        # Optional color reverse
        if reverse is True:
            self.red = self.red[::-1]
            self.blu = self.blu[::-1]
            self.grn = self.grn[::-1]
        # Put in to tuple & dictionary structures needed
        rr = []
        bb = []
        gg = []
        for k in range(0, int(nlev)):
            rr.append((float(k) / (nlev - 1), self.red[k], self.red[k]))
            bb.append((float(k) / (nlev - 1), self.blu[k], self.blu[k]))
            gg.append((float(k) / (nlev - 1), self.grn[k], self.grn[k]))
        cdict = {'red': rr, 'blue': bb, 'green': gg}
        super().__init__('cubehelix', cdict, N=256)
        self._log.debug('Created ' + str(self))

    def plot_helix(self):
        """Display the RGB and luminance plots for the chosen cubehelix.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_helix')
        fig, axis = plt.subplots()
        axis.plot(self.fract, 'k', linewidth=2)
        axis.plot(self.red, 'r', linewidth=2)
        axis.plot(self.grn, 'g', linewidth=2)
        axis.plot(self.blu, 'b', linewidth=2)
        axis.set_xlim(0, self.nlev)
        axis.set_ylim(0, 1)
        axis.set_title('Cubehelix', fontsize=18)
        axis.set_xlabel('color', fontsize=15)
        axis.set_ylabel('amplitude', fontsize=15)
        # Return axis:
        return axis


class ColormapPerception(colors.LinearSegmentedColormap, Colormap3D):
    """A perceptual colormap based on face-based luminance matching.

    Based on a publication by Kindlmann et. al.
    http://www.cs.utah.edu/~gk/papers/vis02/FaceLumin.pdf
    This colormap tries to achieve an isoluminant perception by using a list of colors acquired
    through face recognition studies. It is a lot better than the HLS colormap, but still not
    completely isoluminant (despite its name). Also it appears a bit dark.

    """

    _log = logging.getLogger(__name__ + '.HLSTetradicColormap')

    CDICT = {'red': [(0/6, 0.847, 0.847),
                     (1/6, 0.527, 0.527),
                     (2/6, 0.000, 0.000),
                     (3/6, 0.000, 0.000),
                     (4/6, 0.316, 0.316),
                     (5/6, 0.718, 0.718),
                     (6/6, 0.847, 0.847)],

             'green': [(0/6, 0.057, 0.057),
                       (1/6, 0.527, 0.527),
                       (2/6, 0.592, 0.592),
                       (3/6, 0.559, 0.559),
                       (4/6, 0.316, 0.316),
                       (5/6, 0.000, 0.000),
                       (6/6, 0.057, 0.057)],

             'blue': [(0/6, 0.057, 0.057),
                      (1/6, 0.000, 0.000),
                      (2/6, 0.000, 0.000),
                      (3/6, 0.559, 0.559),
                      (4/6, 0.991, 0.991),
                      (5/6, 0.718, 0.718),
                      (6/6, 0.057, 0.057)]}

    def __init__(self):
        self._log.debug('Calling __init__')
        super().__init__('perception', self.CDICT, N=256)
        self._log.debug('Created ' + str(self))


class ColormapHLS(colors.ListedColormap, Colormap3D):
    """Colormap subclass for encoding directions with colors.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.ListedColormap`
    class. The class follows the HSL ('hue', 'saturation', 'lightness') 'Double Hexcone' Model
    with the saturation always set to 1 (moving on the surface of the color
    cylinder) with a lightness of 0.5 (full color). The three prime colors (`rgb`) are spaced
    equidistant with 120° space in between, according to a triadic arrangement.
    Even though the lightness is constant in the plane, the luminance (which is a weighted sum
    of the RGB components which encompasses human perception) is not, which can lead to
    artifacts like reliefs. Converting the map to a grayscale show spokes at the secondary colors.
    For more information see:
    https://vis4.net/blog/posts/avoid-equidistant-hsv-colors/
    http://www.workwithcolor.com/color-luminance-2233.htm
    http://blog.asmartbear.com/color-wheels.html

    """

    _log = logging.getLogger(__name__ + '.HLSTriadicColormap')

    def __init__(self):
        self._log.debug('Calling __init__')
        h = np.linspace(0, 1, 256)
        l = 0.5 * np.ones_like(h)
        s = np.ones_like(h)
        r, g, b = np.vectorize(colorsys.hls_to_rgb)(h, l, s)
        colors = [(r[i], g[i], b[i]) for i in range(len(r))]
        super().__init__(colors, 'hls', N=256)
        self._log.debug('Created ' + str(self))


class ColormapClassic(colors.LinearSegmentedColormap, Colormap3D):
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
        super().__init__('classic', self.CDICT, N=256)
        self._log.debug('Created ' + str(self))


class ColormapTransparent(colors.LinearSegmentedColormap):
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

    _log = logging.getLogger(__name__ + '.ColormapTransparent')

    def __init__(self, r=0., g=0., b=0., alpha_range=None):
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

cmaps = {'cubehelix_standard': ColormapCubehelix(),
         'cubehelix_reverse': ColormapCubehelix(reverse=True),
         'cubehelix_angular': ColormapCubehelix(start=0, rot=1, minLight=0.5, maxLight=0.5, sat=2),
         'perception_angular': ColormapPerception(),
         'hls_angular': ColormapHLS(),
         'classic_angular': ColormapClassic(),
         'transparent_black': ColormapTransparent(0, 0, 0, [0, 1.]),
         'transparent_white': ColormapTransparent(1, 1, 1, [0, 1.]),
         'transparent_confidence': ColormapTransparent(0.2, 0.3, 0.2, [0.75, 0.])}

CMAP_ANGULAR_DEFAULT = cmaps['cubehelix_angular']

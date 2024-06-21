# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#

# TODO: Own small package? Use viscm (with colorspacious)?
# TODO: Also add cmocean "phase" colormap? Make optional (try importing, fall back to RdBu!)
"""This module provides a number of custom colormaps, which also have capabilities for 3D plotting.
If this is the case, the :class:`~.Colormap3D` colormap class is a parent class. In `cmaps`, a
number of specialised colormaps is available for convenience. If the default for circular colormaps
(used for 3D plotting) should be changed, set it via `CMAP_CMAP_ANGULAR_DEFAULT`.
For general questions about colors see:
http://www.poynton.com/PDFs/GammaFAQ.pdf
http://www.poynton.com/PDFs/ColorFAQ.pdf
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter as FuncForm
from matplotlib.ticker import FixedLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Circle

import numpy as np
from PIL import Image
from matplotlib import colors

from skimage import color as skcolor

import colorsys

import abc

from . import plottools

# TODO: categorize colormaps as sequential, divergent, or cyclic!

__all__ = ['Colormap3D', 'ColormapCubehelix', 'ColormapPerception', 'ColormapHLS',
           'ColormapClassic', 'ColormapTransparent', 'cmaps', 'CMAP_CIRCULAR_DEFAULT',
           'ColorspaceCIELab', 'ColorspaceCIELuv', 'ColorspaceCIExyY', 'ColorspaceYPbPr',
           'interpolate_color', 'rgb_to_brightness', 'colormap_brightness_comparison']
_log = logging.getLogger(__name__)


# TODO: DOCSTRINGS!!!


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

    def rgb_from_vector(self, vector, vmax=None):
        """Construct a hls tuple from three coordinates representing a 3D direction.

        Parameters
        ----------
        vector: tuple (N=3) or :class:`~numpy.ndarray`
            Vector containing the x, y and z component, or a numpy array encompassing the
            components as three lists.

        Returns
        -------
        rgb:  :class:`~numpy.ndarray`
            Numpy array containing the calculated color tuples.

        """
        self._log.debug('Calling rgb_from_vector')
        x, y, z = np.asarray(vector)
        R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        R_max = vmax if vmax is not None else R.max() + 1E-30
        # FIRST color dimension: HUE (1D ring/angular direction)
        phi = np.asarray(np.arctan2(y, x))
        phi[phi < 0] += 2 * np.pi
        hue = phi / (2 * np.pi)
        rgba = np.asarray(self(hue))
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        # SECOND color dimension: SATURATION (2D, in-plane)
        rho = np.sqrt(x ** 2 + y ** 2)
        sat = rho / R_max
        r, g, b = interpolate_color(sat, (0.5, 0.5, 0.5), np.stack((r, g, b), axis=-1))
        # THIRD color dimension: LUMINANCE (3D, color sphere)
        theta = np.arccos(z / R_max)
        lum = 1 - theta / np.pi  # goes from 0 (black) over 0.5 (grey) to 1 (white)!
        lum_target = np.where(lum < 0.5, 0, 1)  # Separate upper(white)/lower(black) hemispheres!
        lum_target = np.stack([lum_target] * 3, axis=-1)  # [0, 0, 0] -> black / [1, 1, 1] -> white!
        fraction = 2 * np.abs(lum - 0.5)  # 0.5: difference from grey, 2: scale to range (0, 1)!
        r, g, b = interpolate_color(fraction, np.stack((r, g, b), axis=-1), lum_target)
        # Return RGB:
        return np.asarray(255 * np.stack((r, g, b), axis=-1), dtype=np.uint8)

    def make_colorwheel(self, size=256, alpha=1, bgcolor=None):
        # TODO: Strange arrows are not straight...
        self._log.debug('Calling make_colorwheel')
        # Construct the colorwheel:
        yy, xx = (np.indices((size, size)) - size/2 + 0.5)
        rr = np.hypot(xx, yy)
        xx = np.where(rr <= size/2-2, xx, 0)
        yy = np.where(rr <= size/2-2, yy, 0)
        zz = np.where(rr <= size/2-2, 0, -1)  # color inside, black outside
        aa = np.where(rr >= size/2-2, 255*alpha, 255).astype(dtype=np.uint8)
        rgba = np.dstack((self.rgb_from_vector(np.asarray((xx, yy, zz))), aa))
        if bgcolor:
            if bgcolor == 'w':  # TODO: Matplotlib get color tuples from string?
                bgcolor = (1, 1, 1)
            if len(bgcolor) == 3 and not isinstance(bgcolor, str):  # Only you have tuple!
                r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
                r = np.where(rr <= size / 2 - 2, r, 255*bgcolor[0]).astype(dtype=np.uint8)
                g = np.where(rr <= size / 2 - 2, g, 255*bgcolor[1]).astype(dtype=np.uint8)
                b = np.where(rr <= size / 2 - 2, b, 255*bgcolor[2]).astype(dtype=np.uint8)
                rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
        # Create color wheel:
        return Image.fromarray(rgba)

    def plot_colorwheel(self, axis=None, size=512, alpha=1, arrows=False, greyscale=False,
                        figsize=(4, 4), bgcolor=None, **kwargs):
        """Display a color wheel to illustrate the color coding of vector gradient directions.

        Parameters
        ----------
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_colorwheel')
        # Construct the colorwheel:
        color_wheel = self.make_colorwheel(size=size, alpha=alpha, bgcolor=bgcolor)
        if greyscale:
            color_wheel = color_wheel.convert('L')
        # Plot the color wheel:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(color_wheel, origin='lower', **kwargs)
        axis.add_patch(Circle(xy=(size/2-0.5, size/2-0.5), radius=size/2-2, linewidth=2,
                              edgecolor='k', facecolor='none'))
        if arrows:
            plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False,
                            left=False, right=False, top=False, bottom=False)
            axis.arrow(size/2, size/2, 0, 0.15*size, head_width=9, head_length=20,
                       fc='k', ec='k', lw=1, width=2)
            axis.arrow(size/2, size/2, 0, -0.15*size, head_width=9, head_length=20,
                       fc='k', ec='k', lw=1, width=2)
            axis.arrow(size/2, size/2, 0.15*size, 0, head_width=9, head_length=20,
                       fc='k', ec='k', lw=1, width=2)
            axis.arrow(size/2, size/2, -0.15*size, 0, head_width=9, head_length=20,
                       fc='k', ec='k', lw=1, width=2)
        # Return axis:
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        for tic in axis.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.label1.set_visible(False)
        for tic in axis.yaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
            tic.label1.set_visible(False)
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
    2016-11 (@jan.caron) Added support for isoluminant cubehelices while making sure
                         `rot` works as intended. Decoded the plane-vectors a bit.
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
        self.fract = np.linspace(minLight, maxLight, nlev)
        angle = 2.0 * np.pi * (start / 3.0 + rot * np.linspace(0, 1, nlev))
        self.fract = self.fract**gamma
        satar = np.linspace(minSat, maxSat, nlev)
        amp = np.asarray(satar * self.fract * (1. - self.fract) / 2)
        # Set RGB color coefficients (Luma is calculated in RGB Rec.601, so choose those),
        # the original version of Dave green used (0.30, 0.59, 0.11) and REc.709 is
        # c709 = (0.2126, 0.7152, 0.0722) but would not produce correct YPbPr Luma.
        c601 = (0.299, 0.587, 0.114)
        cr, cg, cb = c601
        cw = -0.90649  # Chosen to comply with Dave Greens implementation.
        k = -1.6158 / cr / cw  # k has to balance out cw so nothing gets out of RGB gamut (> 1).
        # Calculate the vectors v and w spanning the plane of constant perceived intensity.
        # v and w have to solve v x w = k(cr, cg, cb) (normal vector of the described plane) and
        # v * w = 0 (scalar product, v and w have to be perpendicular).
        # 6 unknown and 4 equations --> Chose wb = 0 and wg = cw (constant).
        v = np.array((k * cr ** 2 * cb / (cw * (cr ** 2 + cg ** 2)),
                      k * cr * cg * cb / (cw * (cr ** 2 + cg ** 2)), -k * cr / cw))
        w = np.array((-cw * cg / cr, cw, 0))
        # Calculate components:
        self.red = self.fract + amp * (v[0] * np.cos(angle) + w[0] * np.sin(angle))
        self.grn = self.fract + amp * (v[1] * np.cos(angle) + w[1] * np.sin(angle))
        self.blu = self.fract + amp * (v[2] * np.cos(angle) + w[2] * np.sin(angle))
        # Original formulas with original v and w:
        # self.red = self.fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
        # self.grn = self.fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
        # self.blu = self.fract + amp * (1.97294 * np.cos(angle))
        # Find where RBG are outside the range [0,1], clip:
        self.red = np.clip(self.red, 0, 1)
        self.grn = np.clip(self.grn, 0, 1)
        self.blu = np.clip(self.blu, 0, 1)
        # Optional color reverse:
        if reverse is True:
            self.red = self.red[::-1]
            self.blu = self.blu[::-1]
            self.grn = self.grn[::-1]
        # Put in to tuple & dictionary structures needed:
        rr, bb, gg = [], [], []
        for k in range(0, int(nlev)):
            rr.append((float(k) / (nlev - 1), self.red[k], self.red[k]))
            bb.append((float(k) / (nlev - 1), self.blu[k], self.blu[k]))
            gg.append((float(k) / (nlev - 1), self.grn[k], self.grn[k]))
        cdict = {'red': rr, 'blue': bb, 'green': gg}
        super().__init__('cubehelix', cdict, N=256)
        self._log.debug('Created ' + str(self))

    def plot_helix(self, figsize=None, **kwargs):
        """Display the RGB and luminance plots for the chosen cubehelix.

        Parameters
        ----------
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        None

        """
        self._log.debug('Calling plot_helix')
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[8, 1])
        # Main plot:
        axis = plt.subplot(gs[0])
        axis.plot(self.fract, 'k', linewidth=2)
        axis.plot(self.red, 'r', linewidth=2)
        axis.plot(self.grn, 'g', linewidth=2)
        axis.plot(self.blu, 'b', linewidth=2)
        axis.set_xlim(0, self.nlev)
        axis.set_ylim(0, 1)
        axis.set_title('Cubehelix', fontsize=18)
        axis.set_xlabel('Color index', fontsize=15)
        axis.set_ylabel('Brightness / RGB', fontsize=15)
        axis.xaxis.set_major_locator(FixedLocator(locs=np.linspace(0, self.nlev, 5)))
        axis.yaxis.set_major_locator(FixedLocator(locs=[0, 0.5, 1]))
        # Colorbar horizontal:
        caxis = plt.subplot(gs[1])
        rgb = self(np.linspace(0, 1, 256))[None, ...]
        rgb = np.asarray(255.9999 * rgb, dtype=np.uint8)
        rgb = np.repeat(rgb, 30, axis=0)
        im = Image.fromarray(rgb, aspect='auto')
        caxis.imshow(im)
        plt.tick_params(axis='both', which='both', labelleft=False, labelbottom=False,
                        left=False, right=False, top=False, bottom=False)
        return plottools.format_axis(axis, scalebar=False, keep_labels=True, **kwargs)


class ColormapPerception(colors.LinearSegmentedColormap, Colormap3D):
    """A perceptual colormap based on face-based luminance matching.

    Based on a publication by Kindlmann et. al.
    http://www.cs.utah.edu/~gk/papers/vis02/FaceLumin.pdf
    This colormap tries to achieve an isoluminant perception by using a list of colors acquired
    through face recognition studies. It is a lot better than the HLS colormap, but still not
    completely isoluminant (despite its name). Also it appears a bit dark.

    """

    _log = logging.getLogger(__name__ + '.ColormapPerception')

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

    _log = logging.getLogger(__name__ + '.ColormapHLS')

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

    _log = logging.getLogger(__name__ + '.ColormapClassic')

    CDICT = {'red': [(0/4, 1.0, 1.0),
                     (1/4, 0.0, 0.0),
                     (2/4, 0.0, 0.0),
                     (3/4, 1.0, 1.0),
                     (4/4, 1.0, 1.0)],

             'green': [(0/4, 0.0, 0.0),
                       (1/4, 0.0, 0.0),
                       (2/4, 1.0, 1.0),
                       (3/4, 1.0, 1.0),
                       (4/4, 0.0, 0.0)],

             'blue': [(0/4, 0.0, 0.0),
                      (1/4, 1.0, 1.0),
                      (2/4, 0.0, 0.0),
                      (3/4, 0.0, 0.0),
                      (4/4, 0.0, 0.0)]}

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


class ColorspaceCIELab(object):  # TODO: Superclass?
    """Class representing the CIELab colorspace."""

    _log = logging.getLogger(__name__ + '.ColorspaceCIELab')

    def __init__(self, dim=(500, 500), extent=(-100, 100, -100, 100), cut_gamut=False, clip=True):
        self._log.debug('Calling __init__')
        self.dim = dim
        self.extent = extent
        self.cut_out_gamut = cut_gamut
        self.clip = clip
        self._log.debug('Created ' + str(self))

    def plot(self, L=53.4, axis=None, figsize=None, **kwargs):
        self._log.debug('Calling plot')
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        a = np.linspace(ext[0], ext[1], dim[1])
        b = np.linspace(ext[2], ext[3], dim[0])
        aa, bb = np.meshgrid(a, b)
        LL = np.full(dim, L, dtype=int)
        Lab = np.stack((LL, aa, bb), axis=-1)
        # Convert to XYZ colorspace:
        XYZ = skcolor.lab2xyz(Lab)
        # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
        rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
        # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
        rgb[~mask] *= 12.92
        # Determine gamut:
        gamut = np.logical_or(rgb < 0, rgb > 1)
        gamut = np.sum(gamut, axis=-1, dtype=bool)
        gamut_mask = np.stack((gamut, gamut, gamut), axis=-1)
        # Cut out gamut (set out of bound colors to gray) if necessary:
        if self.cut_out_gamut:
            rgb[gamut_mask] = 0.5
        # Clip out of gamut colors:
        if self.clip:
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        # Plot colorspace:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(rgb, origin='lower', interpolation='none', extent=(0, dim[0], 0, dim[1]))
        axis.contour(gamut, levels=[0], colors='k', linewidths=1.5)
        axis.set_xlabel('a', fontsize=15)
        axis.set_ylabel('b', fontsize=15)
        axis.set_title('CIELab (L = {:g})'.format(L), fontsize=18)
        axis.xaxis.set_major_locator(FixedLocator(np.linspace(0, dim[1], 5)))
        axis.yaxis.set_major_locator(FixedLocator(np.linspace(0, dim[0], 5)))
        fx = FuncForm(lambda x, pos: '{:.3g}'.format(x / dim[1] * (ext[1] - ext[0]) + ext[0]))
        axis.xaxis.set_major_formatter(fx)
        fy = FuncForm(lambda y, pos: '{:.3g}'.format(y / dim[0] * (ext[3] - ext[2]) + ext[2]))
        axis.yaxis.set_major_formatter(fy)
        plottools.format_axis(axis, scalebar=False, keep_labels=True, **kwargs)

    def plot_colormap(self, cmap, N=256, L='auto', figsize=None, cbar_lim=None, brightness=True,
                      input_rec=None):
        self._log.debug('Calling plot_colormap')
        dim, ext = self.dim, self.extent
        # Calculate rgb values:
        rgb = cmap(np.linspace(0, 1, N))[None, :, :3]  # These are R'G'B' values!
        if input_rec == 601:
            rgb = RGBConverter('Rec601', 'Rec709')(rgb)
        # Convert to Lab space:
        Lab = np.squeeze(skcolor.rgb2lab(rgb))
        LL, aa, bb = Lab.T
        aa = (aa - ext[0]) / (ext[1] - ext[0]) * dim[1]
        bb = (bb - ext[2]) / (ext[3] - ext[2]) * dim[0]
        # Determine number of images / luma levels:
        LL_min, LL_max = np.round(np.min(LL), 1), np.round(np.max(LL), 1)
        if L == 'auto':
            if LL_max - LL_min < 0.1:  # Just one image:
                L = LL_min
            else:  # Two images:
                L = np.asarray((LL_max, np.mean(LL), LL_min))
        L_list = np.atleast_1d(L)
        # Determine colorbar limits:
        if cbar_lim is not None:  # Overwrite limits!
            LL_min, LL_max = cbar_lim
        elif not brightness or LL_max - LL_min < 0.1:  # Just one value, full range for colormap:
            LL_min, LL_max = 0, 1
        # Creat grid:
        if figsize is None:
            figsize = (len(L_list) * 5 + 2, 7)
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(L_list)), axes_pad=0.4, share_all=False,
                         cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.25)
        # Plot:
        if brightness:
            c = LL
            cmap = 'gray'
        else:
            c = np.linspace(0, 1, N)
        for i, axis in enumerate(grid):
            self.plot(L=L_list[i], axis=axis)
            im = axis.scatter(aa, bb, c=c, cmap=cmap, edgecolors='none',
                              vmin=LL_min, vmax=LL_max)
            axis.set_xlim(0, self.dim[1])
            axis.set_ylim(0, self.dim[0])
            axis.cax.colorbar(im, ticks=np.linspace(LL_min, LL_max, 9))

    def plot3d(self, N=9):
        self._log.debug('Calling plot3d')
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        a = np.linspace(ext[0], ext[1], dim[1])
        b = np.linspace(ext[2], ext[3], dim[0])
        aa, bb = np.meshgrid(a, b)
        import visvis  # TODO: If VisPy is ever ready, switch every plot to that!
        for i in range(1, N):
            LL = np.full(dim, i * 100 / N, dtype=int)
            Lab = np.stack((LL, aa, bb), axis=-1)
            # Convert to XYZ colorspace:
            XYZ = skcolor.lab2xyz(Lab)
            # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
            rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
            # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
            mask = rgb > 0.0031308
            rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
            rgb[~mask] *= 12.92
            # Determine gamut:
            gamut = np.logical_or(rgb < 0, rgb > 1)
            gamut = np.sum(gamut, axis=-1, dtype=bool)
            # Alpha:
            alpha = 1.
            a = np.full(dim + (1,), alpha)
            a *= np.logical_not(gamut[..., None])
            rgba = np.asarray(255 * np.dstack((rgb, a)), dtype=np.uint8)
            # Visvis plot:
            obj = visvis.functions.surf(aa, bb, i * 100. / N * np.ones_like(aa), rgba, aa=0)
            obj.parent.light0.ambient = 1.
            obj.parent.light0.diffuse = 0.


class ColorspaceCIELuv(object):
    """Class representing the CIELuv colorspace."""

    _log = logging.getLogger(__name__ + '.ColorspaceCIELuv')

    def __init__(self, dim=(500, 500), extent=(-100, 100, -100, 100), cut_gamut=False, clip=True):
        self._log.debug('Calling __init__')
        self.dim = dim
        self.extent = extent
        self.cut_out_gamut = cut_gamut
        self.clip = clip
        self._log.debug('Created ' + str(self))

    def plot(self, L=53.4, axis=None, figsize=None, **kwargs):
        self._log.debug('Calling plot')
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        u = np.linspace(ext[0], ext[1], dim[1])
        v = np.linspace(ext[2], ext[3], dim[0])
        uu, vv = np.meshgrid(u, v)
        LL = np.full(dim, L, dtype=int)
        Luv = np.stack((LL, uu, vv), axis=-1)
        # Convert to XYZ colorspace:
        XYZ = skcolor.luv2xyz(Luv)
        # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
        rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
        # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
        rgb[~mask] *= 12.92
        # Determine gamut:
        gamut = np.logical_or(rgb < 0, rgb > 1)
        gamut = np.sum(gamut, axis=-1, dtype=bool)
        gamut_mask = np.stack((gamut, gamut, gamut), axis=-1)
        # Cut out gamut (set out of bound colors to gray) if necessary:
        if self.cut_out_gamut:
            rgb[gamut_mask] = 0.5
        # Clip out of gamut colors:
        if self.clip:
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        # Plot colorspace:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(rgb, origin='lower', interpolation='none', extent=(0, dim[0], 0, dim[1]))
        axis.contour(gamut, levels=[0], colors='k', linewidths=1.5)
        axis.set_xlabel('u', fontsize=15)
        axis.set_ylabel('v', fontsize=15)
        axis.set_title('CIELuv (L = {:g})'.format(L), fontsize=18)
        axis.xaxis.set_major_locator(FixedLocator(np.linspace(0, dim[1], 5)))
        axis.yaxis.set_major_locator(FixedLocator(np.linspace(0, dim[0], 5)))
        fx = FuncForm(lambda x, pos: '{:.3g}'.format(x / dim[1] * (ext[1] - ext[0]) + ext[0]))
        axis.xaxis.set_major_formatter(fx)
        fy = FuncForm(lambda y, pos: '{:.3g}'.format(y / dim[0] * (ext[3] - ext[2]) + ext[2]))
        axis.yaxis.set_major_formatter(fy)
        plottools.format_axis(axis, scalebar=False, keep_labels=True, **kwargs)

    def plot_colormap(self, cmap, N=256, L='auto', figsize=None, cbar_lim=None, brightness=True,
                      input_rec=None):
        self._log.debug('Calling plot_colormap')
        dim, ext = self.dim, self.extent
        # Calculate rgb values:
        rgb = cmap(np.linspace(0, 1, N))[None, :, :3]
        if input_rec == 601:
            rgb = RGBConverter('Rec601', 'Rec709')(rgb)
        # Convert to Lab space:
        Luv = np.squeeze(skcolor.rgb2luv(rgb))
        LL, uu, vv = Luv.T
        uu = (uu - ext[0]) / (ext[1] - ext[0]) * dim[1]
        vv = (vv - ext[2]) / (ext[3] - ext[2]) * dim[0]
        # Determine number of images / luma levels:
        LL_min, LL_max = np.round(np.min(LL), 1), np.round(np.max(LL), 1)
        if L == 'auto':
            if LL_max - LL_min < 0.1:  # Just one image:
                L = LL_min
            else:  # Two images:
                L = np.asarray((LL_max, np.mean(LL), LL_min))
        L_list = np.atleast_1d(L)
        # Determine colorbar limits:
        if cbar_lim is not None:  # Overwrite limits!
            LL_min, LL_max = cbar_lim
        elif not brightness or LL_max - LL_min < 0.1:  # Just one value, full range for colormap:
            LL_min, LL_max = 0, 1
        # Creat grid:
        if figsize is None:
            figsize = (len(L_list) * 5 + 2, 7)
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(L_list)), axes_pad=0.4, share_all=False,
                         cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.25)
        # Plot:
        if brightness:
            c = LL
            cmap = 'gray'
        else:
            c = np.linspace(0, 1, N)
        for i, axis in enumerate(grid):
            self.plot(L=L_list[i], axis=axis)
            im = axis.scatter(uu, vv, c=c, cmap=cmap, edgecolors='none',
                              vmin=LL_min, vmax=LL_max)
            axis.set_xlim(0, self.dim[1])
            axis.set_ylim(0, self.dim[0])
            axis.cax.colorbar(im, ticks=np.linspace(LL_min, LL_max, 9))

    def plot3d(self, N=9):
        self._log.debug('Calling plot3d')
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        u = np.linspace(ext[0], ext[1], dim[1])
        v = np.linspace(ext[2], ext[3], dim[0])
        uu, vv = np.meshgrid(u, v)
        import visvis
        for i in range(1, N):
            LL = np.full(dim, i * 100 / N, dtype=int)
            Luv = np.stack((LL, uu, vv), axis=-1)
            # Convert to XYZ colorspace:
            XYZ = skcolor.luv2xyz(Luv)
            # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
            rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
            # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
            mask = rgb > 0.0031308
            rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
            rgb[~mask] *= 12.92
            # Determine gamut:
            gamut = np.logical_or(rgb < 0, rgb > 1)
            gamut = np.sum(gamut, axis=-1, dtype=bool)
            # Alpha:
            alpha = 1.
            a = np.full(dim + (1,), alpha)
            a *= np.logical_not(gamut[..., None])
            rgba = np.asarray(255 * np.dstack((rgb, a)), dtype=np.uint8)
            # Visvis plot:
            obj = visvis.functions.surf(uu, vv, i * 100. / N * np.ones_like(uu), rgba, aa=0)
            obj.parent.light0.ambient = 1.
            obj.parent.light0.diffuse = 0.


class ColorspaceCIExyY(object):
    """Class representing the CIExyY colorspace."""

    _log = logging.getLogger(__name__ + '.ColorspaceCIExyY')

    def __init__(self, dim=(500, 500), extent=(0, 0.8, 0, 0.8), cut_gamut=False, clip=True):
        self._log.debug('Calling __init__')
        self.dim = dim
        self.extent = extent
        self.cut_out_gamut = cut_gamut
        self.clip = clip
        self._log.debug('Created ' + str(self))

    def plot(self, Y=0.214, axis=None, figsize=None, **kwargs):
        self._log.debug('Calling plot')
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        x = np.linspace(ext[0], ext[1], dim[1])
        y = np.linspace(ext[2], ext[3], dim[0])
        xx, yy = np.meshgrid(x, y)
        YY = np.full(dim, Y)
        # Convert to XYZ:
        XX = YY / (yy + 1e-30) * xx
        ZZ = YY / (yy + 1e-30) * (1 - xx - yy)
        XYZ = np.stack((XX, YY, ZZ), axis=-1)
        # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
        rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
        # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
        rgb[~mask] *= 12.92
        # Determine gamut:
        gamut = np.logical_or(rgb < 0, rgb > 1)
        gamut = np.sum(gamut, axis=-1, dtype=bool)
        gamut_mask = np.stack((gamut, gamut, gamut), axis=-1)
        # Cut out gamut (set out of bound colors to gray) if necessary:
        if self.cut_out_gamut:
            rgb[gamut_mask] = 0.5
        # Clip out of gamut colors:
        if self.clip:
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        # Plot colorspace:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(rgb, origin='lower', interpolation='none', extent=(0, dim[0], 0, dim[1]))
        axis.contour(gamut, levels=[0], colors='k', linewidths=1.5)
        axis.set_xlabel('x', fontsize=15)
        axis.set_ylabel('y', fontsize=15)
        axis.set_title('CIExyY (Y = {:g})'.format(Y), fontsize=18)
        axis.xaxis.set_major_locator(FixedLocator(np.linspace(0, dim[1], 5)))
        axis.yaxis.set_major_locator(FixedLocator(np.linspace(0, dim[0], 5)))
        fx = FuncForm(lambda x, pos: '{:.3g}'.format(x / dim[1] * (ext[1] - ext[0]) + ext[0]))
        axis.xaxis.set_major_formatter(fx)
        fy = FuncForm(lambda y, pos: '{:.3g}'.format(y / dim[0] * (ext[3] - ext[2]) + ext[2]))
        axis.yaxis.set_major_formatter(fy)
        plottools.format_axis(axis, scalebar=False, keep_labels=True, **kwargs)

    def plot_colormap(self, cmap, N=256, Y='auto', figsize=None, cbar_lim=None, brightness=True,
                      input_rec=None):
        self._log.debug('Calling plot_colormap')
        dim, ext = self.dim, self.extent
        # Calculate rgb values:
        rgb = cmap(np.linspace(0, 1, N))[None, :, :3]
        if input_rec == 601:
            rgb = RGBConverter('Rec601', 'Rec709')(rgb)
        # Convert to XYZ space:
        XYZ = np.squeeze(skcolor.rgb2xyz(rgb))
        XX, YY, ZZ = XYZ.T
        # Convert to xyY space:
        xx = XX / (XX + YY + ZZ)
        yy = YY / (XX + YY + ZZ)
        xx = (xx - ext[0]) / (ext[1] - ext[0]) * dim[1]
        yy = (yy - ext[2]) / (ext[3] - ext[2]) * dim[0]
        # Determine number of images / luma levels:
        YY_min, YY_max = np.round(np.min(YY), 2), np.round(np.max(YY), 2)
        if Y == 'auto':
            if YY_max - YY_min < 0.01:  # Just one image:
                Y = YY_min
            else:  # Two images:
                Y = np.asarray((YY_max, np.mean(YY), YY_min))
        Y_list = np.atleast_1d(Y)
        # Determine colorbar limits:
        if cbar_lim is not None:  # Overwrite limits!
            YY_min, YY_max = cbar_lim
        elif not brightness or YY_max - YY_min < 0.01:  # Just one value, full range for colormap:
            YY_min, YY_max = 0, 1
        # Creat grid:
        if figsize is None:
            figsize = (len(Y_list) * 5 + 2, 7)
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(Y_list)), axes_pad=0.4, share_all=False,
                         cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.25)
        # Plot:
        if brightness:
            c = YY
            cmap = 'gray'
        else:
            c = np.linspace(0, 1, N)
        for i, axis in enumerate(grid):
            self.plot(Y=Y_list[i], axis=axis)
            im = axis.scatter(xx, yy, c=c, cmap=cmap, edgecolors='none',
                              vmin=YY_min, vmax=YY_max)
            axis.set_xlim(0, self.dim[1])
            axis.set_ylim(0, self.dim[0])
            axis.cax.colorbar(im, ticks=np.linspace(YY_min, YY_max, 9))

    def plot3d(self, N=9):
        self._log.debug('Calling plot3d')
        dim, ext = self.dim, self.extent
        # Create Lab colorspace:
        x = np.linspace(ext[0], ext[1], dim[1])
        y = np.linspace(ext[2], ext[3], dim[0])
        xx, yy = np.meshgrid(x, y)
        import visvis
        for i in range(1, N):
            YY = np.full(dim, i * 1. / N)
            # Convert to XYZ:
            XX = YY / (yy + 1e-30) * xx
            ZZ = YY / (yy + 1e-30) * (1 - xx - yy)
            XYZ = np.stack((XX, YY, ZZ), axis=-1)
            # Convert to RGB colorspace following algorithm from http://www.easyrgb.com/index.php:
            rgb = skcolor.colorconv._convert(skcolor.colorconv.rgb_from_xyz, XYZ)
            # Gamma correction (gamma encoding) rgb are now nonlinear (R'G'B')!:
            mask = rgb > 0.0031308
            rgb[mask] = 1.055 * np.power(rgb[mask], 1 / 2.4) - 0.055
            rgb[~mask] *= 12.92
            # Determine gamut:
            gamut = np.logical_or(rgb < 0, rgb > 1)
            gamut = np.sum(gamut, axis=-1, dtype=bool)
            # Alpha:
            alpha = 1.
            a = np.full(dim + (1,), alpha)
            a *= np.logical_not(gamut[..., None])
            rgba = np.asarray(255 * np.dstack((rgb, a)), dtype=np.uint8)
            # Visvis plot:
            obj = visvis.functions.surf(xx, yy, i / N * np.ones_like(xx), rgba, aa=0)
            obj.parent.light0.ambient = 1.
            obj.parent.light0.diffuse = 0.


class ColorspaceYPbPr(object):
    """Class representing the YPbPr colorspace."""

    _log = logging.getLogger(__name__ + '.ColorspaceYPbPr')

    def __init__(self, dim=(500, 500), extent=(-0.8, 0.8, -0.8, 0.8), cut_gamut=False, clip=True):
        self._log.debug('Calling __init__')
        self.dim = dim
        self.extent = extent
        self.cut_out_gamut = cut_gamut
        self.clip = clip
        self._log.debug('Created ' + str(self))

    def plot(self, Y=0.5, axis=None, figsize=None, **kwargs):
        self._log.debug('Calling plot')
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        dim, ext = self.dim, self.extent
        # Create YPbPr colorspace:
        pb = np.linspace(ext[0], ext[1], dim[1])
        pr = np.linspace(ext[2], ext[3], dim[0])
        ppb, ppr = np.meshgrid(pb, pr)
        YY = np.full(dim, Y)  # This is luma, not relative luminance (Y', not Y)!
        # Convert to RGB colorspace (this is the nonlinear R'G'B' space!):
        rr = YY + 1.402 * ppr
        gg = YY - 0.344136 * ppb - 0.7141136 * ppr
        bb = YY + 1.772 * ppb
        rgb = np.stack((rr, gg, bb), axis=-1)
        # Determine gamut:
        gamut = np.logical_or(rgb < 0, rgb > 1)
        gamut = np.sum(gamut, axis=-1, dtype=bool)
        gamut_mask = np.stack((gamut, gamut, gamut), axis=-1)
        # Cut out gamut (set out of bound colors to gray) if necessary:
        if self.cut_out_gamut:
            rgb[gamut_mask] = 0.5
        # Clip out of gamut colors:
        if self.clip:
            rgb[rgb < 0] = 0
            rgb[rgb > 1] = 1
        # Plot colorspace:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1, aspect='equal')
        axis.imshow(rgb, origin='lower', interpolation='none',
                    extent=(0, dim[0], 0, dim[1]))
        axis.contour(gamut, levels=[0], colors='k', linewidths=1.5)
        axis.set_xlabel('Pb', fontsize=15)
        axis.set_ylabel('Pr', fontsize=15)
        axis.set_title("Y'PbPr (Y' = {:g})".format(Y), fontsize=18)
        axis.xaxis.set_major_locator(FixedLocator(np.linspace(0, dim[1], 5)))
        axis.yaxis.set_major_locator(FixedLocator(np.linspace(0, dim[0], 5)))
        fx = FuncForm(lambda x, pos: '{:.3g}'.format(x / dim[1] * (ext[1] - ext[0]) + ext[0]))
        axis.xaxis.set_major_formatter(fx)
        fy = FuncForm(lambda y, pos: '{:.3g}'.format(y / dim[0] * (ext[3] - ext[2]) + ext[2]))
        axis.yaxis.set_major_formatter(fy)
        plottools.format_axis(axis, scalebar=False, keep_labels=True, **kwargs)

    def plot_colormap(self, cmap, N=256, Y='auto', figsize=None, cbar_lim=None, brightness=True,
                      input_rec=None):
        self._log.debug('Calling plot_colormap')
        dim, ext = self.dim, self.extent
        # Calculate rgb values:
        rgb = cmap(np.linspace(0, 1, N))[None, :, :3]
        if input_rec == 709:
            rgb = RGBConverter('Rec709', 'Rec601')(rgb)
        rr, gg, bb = rgb.T
        # Convert to YPbPr space:
        k_r, k_g, k_b = 0.299, 0.587, 0.114  # Constants Rec.601!
        YY = k_r * rr + k_g * gg + k_b * bb
        ppb = (bb - YY) / (2 * (1 - k_b))
        ppr = (rr - YY) / (2 * (1 - k_r))
        ppb = (ppb - ext[0]) / (ext[1] - ext[0]) * dim[1]
        ppr = (ppr - ext[2]) / (ext[3] - ext[2]) * dim[0]
        # Determine number of images / luma levels:
        YY_min, YY_max = np.round(np.min(YY), 2), np.round(np.max(YY), 2)
        if Y == 'auto':
            if YY_max - YY_min < 0.01:  # Just one image:
                Y = YY_min
            else:  # Two images:
                Y = np.asarray((YY_max, np.mean(YY), YY_min))
        Y_list = np.atleast_1d(Y)
        # Determine colorbar limits:
        if cbar_lim is not None:  # Overwrite limits!
            YY_min, YY_max = cbar_lim
        elif not brightness or YY_max - YY_min < 0.01:  # Just one value, full range for colormap:
            YY_min, YY_max = 0, 1
        # Creat grid:
        if figsize is None:
            figsize = (len(Y_list) * 5 + 2, 7)
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(Y_list)), axes_pad=0.4, share_all=False,
                         cbar_location="right", cbar_mode="single", cbar_size="5%", cbar_pad=0.25)
        # Plot:
        if brightness:
            c = YY
            cmap = 'gray'
        else:
            c = np.linspace(0, 1, N)
        for i, axis in enumerate(grid):
            self.plot(Y=Y_list[i], axis=axis)
            im = axis.scatter(ppb, ppr, c=c, cmap=cmap, edgecolors='none',
                              vmin=YY_min, vmax=YY_max)
            axis.set_xlim(0, self.dim[1])
            axis.set_ylim(0, self.dim[0])
            axis.cax.colorbar(im, ticks=np.linspace(YY_min, YY_max, 9))

    def plot3d(self, N=9):
        self._log.debug('Calling plot3d')
        dim, ext = self.dim, self.extent
        # Create YPbPr colorspace:
        pb = np.linspace(ext[0], ext[1], dim[1])
        pr = np.linspace(ext[2], ext[3], dim[0])
        ppb, ppr = np.meshgrid(pb, pr)
        import visvis
        for i in range(1, N):
            YY = np.full(dim, i * 1. / N)  # This is luma, not relative luminance (Y', not Y)!
            # Convert to RGB colorspace (this is the nonlinear R'G'B' space!):
            rr = YY + 1.402 * ppr
            gg = YY - 0.344136 * ppb - 0.7141136 * ppr
            bb = YY + 1.772 * ppb
            rgb = np.stack((rr, gg, bb), axis=-1)
            # Determine gamut:
            gamut = np.logical_or(rgb < 0, rgb > 1)
            gamut = np.sum(gamut, axis=-1, dtype=bool)
            # Alpha:
            alpha = 1.
            a = np.full(dim + (1,), alpha)
            a *= np.logical_not(gamut[..., None])
            rgba = np.asarray(255 * np.dstack((rgb, a)), dtype=np.uint8)
            # Visvis plot:
            obj = visvis.functions.surf(ppb, ppr, i / N * np.ones_like(ppb), rgba, aa=0)
            obj.parent.light0.ambient = 1.
            obj.parent.light0.diffuse = 0.


class RGBConverter(object):
    """Class for the conversion of RGB values from one RGB space to another.

    Notes
    -----
    This operates only on NONLINEAR R'G'B' values, normalised to a range of [0, 1]!
    Convert from linear RGB values beforehand, if necessary!

    """

    rgb601_to_ypbpr = np.array([[+0.299000, +0.587000, +0.114000],
                                [-0.168736, -0.331264, +0.500000],
                                [+0.500000, -0.418688, -0.081312]])
    ypbr_to_rgb709 = np.array([[1, +0.0000, +1.5701],
                               [1, -0.1870, -0.4664],
                               [1, +1.8556, +0.0000]])
    rgb601_to_rgb709 = ypbr_to_rgb709.dot(rgb601_to_ypbpr)
    rgb709_to_rgb601 = np.linalg.inv(rgb601_to_rgb709)

    def __init__(self, source='Rec601', target='Rec709'):
        if source == 'Rec601' and target == 'Rec709':
            self.convert_matrix = self.rgb601_to_rgb709
        elif source == 'Rec709' and target == 'Rec601':
            self.convert_matrix = self.rgb709_to_rgb601
        else:
            raise KeyError('Conversion from {} to {} not found!'.format(source, target))

    def __call__(self, rgb):
        """Convert from one RGB space to another.

        Parameters
        ----------
        rgb: :class:`~numpy.ndarray`
            Numpy array containing the RGB source values (last dimension: 3).

        Returns
        -------
        rgb_result: :class:`~numpy.ndarray`
            The resulting RGB values in the target space.

        """
        rgb_out = rgb.reshape((-1, 3)).T
        rgb_out = self.convert_matrix.dot(rgb_out)
        return rgb_out.T.reshape(rgb.shape)


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


def rgb_to_brightness(rgb, mode="Y'", input_rec=None):

    import colorspacious  # TODO: Use for everything!
    c = {601: [0.299, 0.587, 0.114], 709: [0.2125, 0.7154, 0.0721]}  # Image.convert('L') uses 601!
    if input_rec is None:  # Not specified, use in all cases:
        rgbp601 = rgb
        rgbp709 = rgb
    elif input_rec == 601:
        rgbp601 = rgb
        rgbp709 = RGBConverter('Rec601', 'Rec709')(rgb)
    elif input_rec == 709:
        rgbp601 = RGBConverter('Rec601', 'Rec709')(rgb)
        rgbp709 = rgb
    else:
        raise KeyError('Input RGB type {} not understood!'.format(input_rec))
    if mode in ("Y'", 'Luma'):
        rp601, gp601, bp601 = rgbp601.T
        brightness = c[601][0] * rp601 + c[601][1] * gp601 + c[601][2] * bp601
    elif mode in ('Y', 'Luminance'):
        rgb709 = colorspacious.cspace_converter('sRGB1', 'sRGB1-linear')(rgbp709)
        r709, g709, b709 = rgb709.T
        brightness = c[709][0] * r709 + c[709][1] * g709 + c[709][2] * b709
    elif mode in ('L*', 'LightnessLab'):
        lab = colorspacious.cspace_converter('sRGB1', 'CIELab')(rgbp709)
        brightness = lab[0, :, 0]
    elif mode in ('I', 'Intensity', 'Average'):
        brightness = np.mean(rgb, axis=-1)
    elif mode in ('V', 'Value', 'Maximum'):
        brightness = np.max(rgb, axis=-1)
    elif mode in ('L', 'LightnessHSL'):
        brightness = (np.max(rgb, axis=-1) + np.min(rgb, axis=-1)) / 2
    else:
        raise KeyError('Brightness request {} not understood!'.format(mode))
    return brightness


def colormap_brightness_comparison(cmap, input_rec=None, figsize=(18, 8)):

    # Create R'G'B' values from colormap:
    x = np.linspace(0, 1, 1000)
    rgbp = cmap(x)[None, :, :3]
    # Calculate different brightness measures:
    luma = rgb_to_brightness(rgbp, mode="Y'", input_rec=input_rec)
    luminance = rgb_to_brightness(rgbp, mode='Y', input_rec=input_rec)
    lightness_lab = rgb_to_brightness(rgbp, mode='L*', input_rec=input_rec)
    intensity = rgb_to_brightness(rgbp, mode='I', input_rec=input_rec)
    value = rgb_to_brightness(rgbp, mode='V', input_rec=input_rec)
    lightness_hls = rgb_to_brightness(rgbp, mode='L', input_rec=input_rec)
    # Plot:
    fig, grid = plt.subplots(2, 3, figsize=figsize)
    plt.title(cmap.name)
    axis = grid[0, 0]
    axis.scatter(x, luma, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=0.5, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title("Luma $Y$ '")
    axis = grid[0, 1]
    axis.scatter(x, luminance, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=0.214, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Relative Luminance $Y$')
    axis = grid[0, 2]
    axis.scatter(x, lightness_lab, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=53.39, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 100)
    axis.set_title('Lightness $L^*$ (CIELab)')
    axis = grid[1, 0]
    axis.scatter(x, intensity, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=53.39, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Intensity $I$ (HSI Component Average)')
    axis = grid[1, 1]
    axis.scatter(x, value, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=53.39, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Value $V$ (HSV Component Maximum)')
    axis = grid[1, 2]
    axis.scatter(x, lightness_hls, c=x, cmap=cmap, s=200, linewidths=0.)
    axis.axhline(y=53.39, color='k', ls='--')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title('Lightness $L$ (HSL Min-Max-Average)')


cmaps = {'cubehelix_standard': ColormapCubehelix(),
         'cubehelix_reverse': ColormapCubehelix(reverse=True),
         'cubehelix_circular': ColormapCubehelix(start=1, rot=1,
                                                 minLight=0.5, maxLight=0.5, sat=2),
         'perception_circular': ColormapPerception(),
         'hls_circular': ColormapHLS(),
         'classic_circular': ColormapClassic(),
         'transparent_black': ColormapTransparent(0, 0, 0, [0, 1.]),
         'transparent_white': ColormapTransparent(1, 1, 1, [0, 1.]),
         'transparent_confidence': ColormapTransparent(0.2, 0.3, 0.2, [0.75, 0.])}

CMAP_CIRCULAR_DEFAULT = cmaps['cubehelix_circular']

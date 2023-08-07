# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.PhaseMap` class for storing phase map data."""

import logging

from numbers import Number

import numpy as np

from PIL import Image

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from mpl_toolkits.mplot3d import Axes3D

import cmocean

from scipy import ndimage

from . import colors
from . import plottools

__all__ = ['PhaseMap']


# TODO: check out pint for units and stuff!

class PhaseMap(object):
    """Class for storing phase map data.

    Represents 2-dimensional phase maps. The phase information itself is stored as a 2-dimensional
    matrix in `phase`, but can also be accessed as a vector via `phase_vec`. :class:`~.PhaseMap`
    objects support negation, arithmetic operators (``+``, ``-``, ``*``) and their augmented
    counterparts (``+=``, ``-=``, ``*=``), with numbers and other :class:`~.PhaseMap`
    objects, if their dimensions and grid spacings match. It is possible to load data from HDF5
    or textfiles or to save the data in these formats. Methods for plotting the phase or a
    corresponding holographic contour map are provided. Holographic contour maps are created by
    taking the cosine of the (optionally amplified) phase and encoding the direction of the
    2-dimensional gradient via color. The directional encoding can be seen by using the
    :func:`~.make_color_wheel` function. Use the :func:`~.plot_combined` function to plot the
    phase map and the holographic contour map next to each other.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    phase: :class:`~numpy.ndarray` (N=2)
        Array containing the phase shift.
    mask: :class:`~numpy.ndarray` (boolean, N=2, optional)
        Mask which determines the projected magnetization distribution, gotten from MIP images or
        otherwise acquired. Defaults to an array of ones (all pixels are considered).
    confidence: :class:`~numpy.ndarray` (N=2, optional)
        Confidence array which determines the trust of specific regions of the phasemap. A value
        of 1 means the pixel is trustworthy, a value of 0 means it is not. Defaults to an array of
        ones (full trust for all pixels). Can be used for the construction of Se_inv.

    """

    _log = logging.getLogger(__name__)

    UNITDICT = {u'rad': 1E0,
                u'mrad': 1E3,
                u'µrad': 1E6,
                u'nrad': 1E9,
                u'1/rad': 1E0,
                u'1/mrad': 1E-3,
                u'1/µrad': 1E-6,
                u'1/nrad': 1E-9}

    @property
    def a(self):
        """Grid spacing in nm."""
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = float(a)

    @property
    def dim_uv(self):
        """Dimensions of the grid."""
        return self._dim_uv

    @property
    def phase(self):
        """Array containing the phase shift."""
        return self._phase

    @phase.setter
    def phase(self, phase):
        assert isinstance(phase, np.ndarray), 'Phase has to be a numpy array!'
        assert len(phase.shape) == 2, 'Phase has to be 2-dimensional, not {}!'.format(phase.shape)
        self._phase = phase.astype(dtype=np.float32)
        self._dim_uv = phase.shape

    @property
    def phase_vec(self):
        """Vector containing the phase shift."""
        return self.phase.ravel()

    @phase_vec.setter
    def phase_vec(self, phase_vec):
        assert isinstance(phase_vec, np.ndarray), 'Vector has to be a numpy array!'
        assert np.size(phase_vec) == np.prod(self.dim_uv), 'Vector size has to match phase!'
        self.phase = phase_vec.reshape(self.dim_uv)

    @property
    def mask(self):
        """Mask which determines the projected magnetization distribution"""
        return self._mask

    @mask.setter
    def mask(self, mask):
        if mask is not None:
            assert mask.shape == self.phase.shape, 'Mask and phase dimensions must match!!'
        else:
            mask = np.ones_like(self.phase, dtype=bool)
        self._mask = mask.astype(np.bool)

    @property
    def confidence(self):
        """Confidence array which determines the trust of specific regions of the phasemap."""
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        if confidence is not None:
            assert confidence.shape == self.phase.shape, \
                'Confidence and phase dimensions must match!'
            confidence = confidence.astype(dtype=np.float32)
            confidence /= confidence.max() + 1E-30  # Normalise!
        else:
            confidence = np.ones_like(self.phase, dtype=np.float32)
        self._confidence = confidence

    def __init__(self, a, phase, mask=None, confidence=None):
        self._log.debug('Calling __init__')
        self.a = a
        self.phase = phase
        self.mask = mask
        self.confidence = confidence
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, phase=%r, mask=%r, confidence=%r)' % \
               (self.__class__, self.a, self.phase, self.mask, self.confidence)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMap(a=%s, dim_uv=%s, mask=%s)' % (self.a, self.dim_uv, not np.all(self.mask))

    def __neg__(self):  # -self
        self._log.debug('Calling __neg__')
        return PhaseMap(self.a, -self.phase, self.mask, self.confidence)

    def __add__(self, other):  # self + other
        self._log.debug('Calling __add__')
        assert isinstance(other, (PhaseMap, Number)), \
            'Only PhaseMap objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, PhaseMap):
            self._log.debug('Adding two PhaseMap objects')
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.phase.shape == self.dim_uv, \
                'Added field has to have the same dimensions!'
            mask_comb = np.logical_or(self.mask, other.mask)  # masks combine
            conf_comb = np.minimum(self.confidence, other.confidence)  # use minimum confidence!
            return PhaseMap(self.a, self.phase + other.phase, mask_comb, conf_comb)
        else:  # other is a Number
            self._log.debug('Adding an offset')
            return PhaseMap(self.a, self.phase + other, self.mask, self.confidence)

    def __sub__(self, other):  # self - other
        self._log.debug('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        assert (isinstance(other, Number) or (isinstance(other, np.ndarray)
                                              and other.shape == self.dim_uv)), \
            'PhaseMap objects can only be multiplied by scalar numbers or fitting arrays!'
        return PhaseMap(self.a, self.phase * other, self.mask, self.confidence)

    def __truediv__(self, other):  # self / other
        self._log.debug('Calling __truediv__')
        assert (isinstance(other, Number) or (isinstance(other, np.ndarray)
                                              and other.shape == self.dim_uv)), \
            'PhaseMap objects can only be divided by scalar numbers or fitting arrays!'
        return PhaseMap(self.a, self.phase / other, self.mask, self.confidence)

    def __floordiv__(self, other):  # self // other
        self._log.debug('Calling __floordiv__')
        assert (isinstance(other, Number) or (isinstance(other, np.ndarray)
                                              and other.shape == self.dim_uv)), \
            'PhaseMap objects can only be divided by scalar numbers or fitting arrays!'
        return PhaseMap(self.a, self.phase // other, self.mask, self.confidence)

    def __radd__(self, other):  # other + self
        self._log.debug('Calling __radd__')
        return self.__add__(other)

    def __rsub__(self, other):  # other - self
        self._log.debug('Calling __rsub__')
        return -self.__sub__(other)

    def __rmul__(self, other):  # other * self
        self._log.debug('Calling __rmul__')
        return self.__mul__(other)

    def __iadd__(self, other):  # self += other
        self._log.debug('Calling __iadd__')
        return self.__add__(other)

    def __isub__(self, other):  # self -= other
        self._log.debug('Calling __isub__')
        return self.__sub__(other)

    def __imul__(self, other):  # self *= other
        self._log.debug('Calling __imul__')
        return self.__mul__(other)

    def __itruediv__(self, other):  # self /= other
        self._log.debug('Calling __itruediv__')
        return self.__truediv__(other)

    def __ifloordiv__(self, other):  # self //= other
        self._log.debug('Calling __ifloordiv__')
        return self.__floordiv__(other)

    def __getitem__(self, item):
        return PhaseMap(self.a, self.phase[item], self.mask[item], self.confidence[item])

    def __array__(self, dtype=None):  # Used for numpy ufuncs, together with __array_wrap__!
        if dtype:
            return self.phase.astype(dtype)
        else:
            return self.phase

    def __array_wrap__(self, array, _=None):  # _ catches the context, which is not used.
        return PhaseMap(self.a, array, self.mask, self.confidence)

    def copy(self):
        """Returns a copy of the :class:`~.PhaseMap` object

        Returns
        -------
        phasemap: :class:`~.PhaseMap`
            A copy of the :class:`~.PhaseMap`.

        """
        self._log.debug('Calling copy')
        return PhaseMap(self.a, self.phase.copy(), self.mask.copy(),
                        self.confidence.copy())

    # TODO: ALL NOT IN PLACE!!!

    def scale_down(self, n=1):
        """Scale down the phase map by averaging over two pixels along each axis.

        Parameters
        ----------
        n : int, optional
            Number of times the phase map is scaled down. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        Only possible, if each axis length is a power of 2!

        """
        self._log.debug('Calling scale_down')
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        a_new = self.a * 2 ** n
        phase_new = self.phase
        mask_new = self.mask
        confidence_new = self.confidence
        for t in range(n):
            dim_uv = phase_new.shape
            # Pad if necessary:
            pv, pu = (0, dim_uv[0] % 2), (0, dim_uv[1] % 2)
            if pv != 0 or pu != 0:
                phase_new = np.pad(phase_new, (pv, pu), mode='edge')
                confidence_new = np.pad(confidence_new, (pv, pu), mode='edge')
                mask_new = np.pad(mask_new, (pv, pu), mode='edge')
                dim_uv = phase_new.shape  # Update dimensions!
            # Create coarser grid for the phase image:
            phase_new = phase_new.reshape((dim_uv[0] // 2, 2, dim_uv[1] // 2, 2)).mean(axis=(3, 1))
            mask = mask_new.reshape(dim_uv[0] // 2, 2, dim_uv[1] // 2, 2)
            mask_new = mask[:, 0, :, 0] & mask[:, 1, :, 0] & mask[:, 0, :, 1] & mask[:, 1, :, 1]
            confidence_new = confidence_new.reshape(dim_uv[0] // 2, 2,
                                                    dim_uv[1] // 2, 2).min(axis=(3, 1))
        return PhaseMap(a_new, phase_new, mask_new, confidence_new)

    def scale_up(self, n=1, order=0):
        """Scale up the phase map using spline interpolation of the requested order.

        Parameters
        ----------
        n : int, optional
            Power of 2 with which the grid is scaled. Default is 1, which means every axis is
            increased by a factor of ``2**1 = 2``.
        order : int, optional
            The order of the spline interpolation, which has to be in the range between 0 and 5
            and defaults to 0.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.

        """
        self._log.debug('Calling scale_up')
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, int), \
            'order must be a positive integer between 0 and 5!'
        a_new = self.a / 2 ** n
        phase_new = ndimage.zoom(self.phase, zoom=2 ** n, order=order)
        mask_new = ndimage.zoom(self.mask, zoom=2 ** n, order=0)
        confidence_new = ndimage.zoom(self.confidence, zoom=2 ** n, order=order)
        return PhaseMap(a_new, phase_new, mask_new, confidence_new)

    def pad(self, pad_values, mode='constant', masked=False, **kwargs):
        """Pad the current phase map with zeros for each individual axis.

        Parameters
        ----------
        pad_values : tuple of int
            Number of zeros which should be padded. Provided as a tuple where each entry
            corresponds to an axis. An entry can be one int (same padding for both sides) or again
            a tuple which specifies the pad values for both sides of the corresponding axis.
        mode: string or function
            A string values or a user supplied function. ‘constant’ pads with zeros. ‘edge’ pads
            with the edge values of array. See the numpy pad function for an in depth guide.
        masked: boolean, optional
            Determines if the padded areas should be masked or not. `True` creates a 'buffer
            zone' for the magnetization distribution in the reconstruction. Default is `False`

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions accordingly.
        The confidence of the padded areas is set to zero!

        """
        self._log.debug('Calling pad')
        assert len(pad_values) == 2, 'Pad values for each dimension have to be provided!'
        pval = np.zeros(4, dtype=np.int)
        for i, values in enumerate(pad_values):
            assert np.shape(values) in [(), (2,)], 'Only one or two values per axis can be given!'
            pval[2 * i:2 * (i + 1)] = values
        phase_pad = np.pad(self.phase, ((pval[0], pval[1]), (pval[2], pval[3])),
                           mode=mode, **kwargs)
        confidence_pad = np.pad(self.confidence, ((pval[0], pval[1]), (pval[2], pval[3])),
                                mode=mode, **kwargs)
        if masked:
            mask_kwds = {'mode': 'constant', 'constant_values': True}
        else:
            mask_kwds = {'mode': mode}
        mask_pad = np.pad(self.mask, ((pval[0], pval[1]), (pval[2], pval[3])), **mask_kwds)
        return PhaseMap(self.a, phase_pad, mask_pad, confidence_pad)

    def crop(self, crop_values):
        """Pad the current phase map with zeros for each individual axis.

        Parameters
        ----------
        crop_values : tuple of int
            Number of zeros which should be cropped. Provided as a tuple where each entry
            corresponds to an axis. An entry can be one int (same cropping for both sides) or again
            a tuple which specifies the crop values for both sides of the corresponding axis.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions accordingly.

        """
        self._log.debug('Calling crop')
        assert len(crop_values) == 2, 'Crop values for each dimension have to be provided!'
        cv = np.zeros(4, dtype=np.int)
        for i, values in enumerate(crop_values):
            assert np.shape(values) in [(), (2,)], 'Only one or two values per axis can be given!'
            cv[2 * i:2 * (i + 1)] = values
        cv *= np.resize([1, -1], len(cv))
        cv = np.where(cv == 0, None, cv)
        phase_crop = self.phase[cv[0]:cv[1], cv[2]:cv[3]]
        mask_crop = self.mask[cv[0]:cv[1], cv[2]:cv[3]]
        confidence_crop = self.confidence[cv[0]:cv[1], cv[2]:cv[3]]
        return PhaseMap(self.a, phase_crop, mask_crop, confidence_crop)

    def flip(self, axis='u'):
        """Flip/mirror the phase map around the specified axis.

        Parameters
        ----------
        axis: {'u', 'v'}, optional
            The axis around which the phase map is flipped.

        Returns
        -------
        phasemap_flip: :class:`~.PhaseMap`
           A flipped copy of the :class:`~.PhaseMap` object.

        """
        self._log.debug('Calling flip')
        if axis == 'u':
            return PhaseMap(self.a, np.flipud(self.phase), np.flipud(self.mask),
                            np.flipud(self.confidence))
        if axis == 'v':
            return PhaseMap(self.a, np.fliplr(self.phase), np.fliplr(self.mask),
                            np.fliplr(self.confidence))
        else:
            raise ValueError("Wrong input! 'u', 'v' allowed!")

    def rotate(self, angle):
        """Rotate the phase map (right hand rotation).

        Parameters
        ----------
        angle: float
            The angle around which the phase map is rotated.

        Returns
        -------
        phasemap_rot: :class:`~.PhaseMap`
           A rotated copy of the :class:`~.PhaseMap` object.

        """
        self._log.debug('Calling rotate')
        phase_rot = ndimage.rotate(self.phase, angle, reshape=False)
        mask_rot = ndimage.rotate(self.mask, angle, reshape=False, order=0)
        conf_rot = ndimage.rotate(self.confidence, angle, reshape=False)
        return PhaseMap(self.a, phase_rot, mask_rot, conf_rot)

    def shift(self, shift):
        """Shift the phase map (subpixel accuracy).

        Parameters
        ----------
        shift : float or sequence, optional
            The shift along the axes. If a float, shift is the same for each axis.
            If a sequence, shift should contain one value for each axis.

        Returns
        -------
        phasemap_shift: :class:`~.PhaseMap`
           A shifted copy of the :class:`~.PhaseMap` object.

        """
        self._log.debug('Calling shift')
        phase_rot = ndimage.shift(self.phase, shift, mode='constant', cval=0)
        mask_rot = ndimage.shift(self.mask, shift, mode='constant', cval=False, order=0)
        conf_rot = ndimage.shift(self.confidence, shift, mode='constant', cval=0)
        return PhaseMap(self.a, phase_rot, mask_rot, conf_rot)

    @classmethod
    def from_signal(cls, signal):
        """Convert a :class:`~hyperspy.signals.Image` object to a :class:`~.PhaseMap` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Image`
            The :class:`~hyperspy.signals.Image` object which should be converted to a PhaseMap.

        Returns
        -------
        phasemap: :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object containing the loaded data.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        cls._log.debug('Calling from_signal')
        # Extract phase:
        phase = signal.data
        # Extract properties:
        a = signal.axes_manager.signal_axes[0].scale
        try:
            mask = signal.metadata.Signal.mask
            confidence = signal.metadata.Signal.confidence
        except AttributeError:
            mask = None
            confidence = None
        return cls(a, phase, mask, confidence)

    def to_signal(self):
        """Convert :class:`~.PhaseMap` data into a HyperSpy Signal.

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal2D`
            Representation of the :class:`~.PhaseMap` object as a HyperSpy Signal.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        self._log.debug('Calling to_signal')
        try:  # Try importing HyperSpy:
            # noinspection PyUnresolvedReferences
            import hyperspy.api as hs
        except ImportError:
            self._log.error('This method recquires the hyperspy package!')
            return
        # Create signal:
        signal = hs.signals.Signal2D(self.phase)
        # Set axes:
        signal.axes_manager.signal_axes[0].name = 'x-axis'
        signal.axes_manager.signal_axes[0].units = 'nm'
        signal.axes_manager.signal_axes[0].scale = self.a
        signal.axes_manager.signal_axes[1].name = 'y-axis'
        signal.axes_manager.signal_axes[1].units = 'nm'
        signal.axes_manager.signal_axes[1].scale = self.a
        # Set metadata:
        signal.metadata.Signal.title = 'PhaseMap'
        signal.metadata.Signal.unit = 'rad'
        signal.metadata.Signal.mask = self.mask
        signal.metadata.Signal.confidence = self.confidence
        # Create and return signal:
        return signal

    def save(self, filename, save_mask=False, save_conf=False, pyramid_format=True, **kwargs):
        """Saves the phasemap in the specified format.

        The function gets the format from the extension:
            - hdf5 for HDF5.
            - rpl for Ripple (useful to export to Digital Micrograph).
            - unf for SEMPER unf binary format.
            - txt format.
            - Many image formats such as png, tiff, jpeg...

        If no extension is provided, 'hdf5' is used. Most formats are
        saved with the HyperSpy package (internally the phasemap is first
        converted to a HyperSpy Signal.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename: str, optional
            Name of the file which the phasemap is saved into. The extension
            determines the saving procedure.
        save_mask: boolean, optional
            If True, the `mask` is saved, too. For all formats, except HDF5, a separate file will
            be created. HDF5 always saves the `mask` in the metadata, independent of this flag. The
            default is False.
        save_conf: boolean, optional
            If True, the `confidence` is saved, too. For all formats, except HDF5, a separate file
            will be created. HDF5 always saves the `confidence` in the metadata, independent of
            this flag. The default is False
        pyramid_format: boolean, optional
            Only used for saving to '.txt' files. If this is True, the grid spacing is saved
            in an appropriate header. Otherwise just the phase is written with the
            corresponding `kwargs`.

        """
        from .file_io.io_phasemap import save_phasemap
        save_phasemap(self, filename, save_mask, save_conf, pyramid_format, **kwargs)

    def plot_phase(self, unit='auto', vmin=None, vmax=None, sigma_clip=None, symmetric=True,
                   show_mask=True, show_conf=True, norm=None, cbar=True,  # specific to plot_phase!
                   cmap=None, interpolation='none', axis=None, figsize=None, **kwargs):
        """Display the phasemap as a colormesh.

        Parameters
        ----------
        unit: {'rad', 'mrad', 'µrad', '1/rad', '1/mrad', '1/µrad'}, optional
            The plotting unit of the phase map. The phase is scaled accordingly before plotting.
            Inverse radians should be used for gain maps!
        vmin : float, optional
            Minimum value used for determining the plot limits. If not set, it will be
            determined by the minimum of the phase directly.
        vmax : float, optional
            Maximum value used for determining the plot limits. If not set, it will be
            determined by the minimum of the phase directly.
        sigma_clip : int, optional
            If this is not `None`, the values outside `sigma_clip` times the standard deviation
            will be clipped for the calculation of the plotting `limit`.
        symmetric : boolean, optional
            If True (default), a zero symmetric colormap is assumed and a zero value (which
            will always be present) will be set to the central color of the colormap.
        show_mask : bool, optional
            A switch determining if the mask should be plotted or not. Default is True.
        show_conf : float, optional
            A switch determining if the confidence should be plotted or not. Default is True.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
        cbar : bool, optional
            If True (default), a colorbar will be plotted.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method for the holographic contour map.
            No interpolation is used in the default case.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        axis, cbar: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        Notes
        -----
        Uses :func:`~.plottools.format_axis` at the end. According keywords can also be given here.

        """
        self._log.debug('Calling plot_phase')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        # Take units into consideration:
        if unit == 'auto':  # Try to automatically determine unit (recommended):
            for key, value in self.UNITDICT.items():
                if not key.startswith('1/'):
                    order = np.floor(np.log10(np.abs(self.phase).max() * value))
                    if -1 <= order < 2:
                        unit = key
            if unit == 'auto':   # No fitting unit was found:
                unit = 'rad'
        # Scale phase and make last check if order is okay:
        phase = self.phase * self.UNITDICT[unit]
        order = np.floor(np.log10(np.abs(phase).max()))
        unit_orderless=unit
        #if order > 2 or order < -6:  # Display would look bad # removed because prevent normalisation
        #    unit = '{} x 1E{:g}'.format(unit, order)
        #    phase /= 10 ** order
        # Calculate limits if necessary (not necessary if both limits are already set):
        if vmin is None or vmax is None:
            phase_lim = phase
            # Clip non-trustworthy regions for the limit calculation:
            if show_conf:
                phase_trust = np.where(self.confidence > 0.9, phase_lim, np.nan)
                phase_min, phase_max = np.nanmin(phase_trust), np.nanmax(phase_trust)
                phase_lim = np.clip(phase_lim, phase_min, phase_max)
            # Cut outlier beyond a certain sigma-margin:
            if sigma_clip is not None:
                outlier = np.abs(phase_lim - np.mean(phase_lim)) < sigma_clip * np.std(phase_lim)
                phase_sigma = np.where(outlier, phase_lim, np.nan)
                phase_min, phase_max = np.nanmin(phase_sigma), np.nanmax(phase_sigma)
                phase_lim = np.clip(phase_lim, phase_min, phase_max)
            # Calculate the limits if necessary (zero has to be present!):
            if vmin is None:
                vmin = np.min(phase_lim)
            if vmax is None:
                vmax = np.max(phase_lim)
        else:  # If vmin and vmax are set by the user, they have to be unit-scaled as well:
            vmin, vmax = vmin * self.UNITDICT[unit_orderless], vmax * self.UNITDICT[unit_orderless]
        # Configure colormap and fix white to zero if colormap is symmetric:
        if cmap is None:
            cmap = cmocean.cm.balance
            # TODO: use cmocean.cm.balance (flipped colours!)
            # TODO: get default from "colors" or "plots" package
            # TODO: make flexible, cmocean and matplotlib...
        elif isinstance(cmap, str):  # Get colormap if given as string:
            cmap = plt.get_cmap(cmap)
        if symmetric:
            vmin, vmax = np.min([vmin, -0]), np.max([0, vmax])  # Ensure zero is present!
            limit = np.max(np.abs([vmin, vmax]))
            # A symmetric colormap only has zero at white (the symmetry point) if the values
            # of the corresponding mappable go from -limit to +limit (symmetric bounds)!
            # Calculate the colors of this symmetric colormap for the range vmin to vmax:
            start = 0.5 + vmin/(2*limit)  # 0 for symmetric bounds, >0: unused colors at lower end!
            end = 0.5 + vmax/(2*limit)  # 1 for symmetric bounds, <1: unused colors at upper end!
            cmap_colors = cmap(np.linspace(start, end, 256))
            # Use calculated colors to create custom (asymmetric) colormap with white at zero:
            cmap = LinearSegmentedColormap.from_list('custom', cmap_colors)
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Plot the phasemap:
        im = axis.imshow(phase, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation,
                         norm=norm, origin='lower', extent=(0, self.dim_uv[1], 0, self.dim_uv[0]))
        if show_mask or show_conf:
            vv, uu = np.indices(self.dim_uv) + 0.5
            if show_conf and not np.all(self.confidence == 1.0):
                colormap = colors.cmaps['transparent_confidence']
                axis.imshow(self.confidence, cmap=colormap, interpolation=interpolation,
                            origin='lower', extent=(0, self.dim_uv[1], 0, self.dim_uv[0]))
            if show_mask and not np.all(self.mask):  # Plot mask if desired and not trivial!
                axis.contour(uu, vv, self.mask, levels=[0.5], colors='k', linestyles='dotted',
                             linewidths=2)
        # Determine colorbar title:
        cbar_label = kwargs.pop('cbar_label', None)
        cbar_mappable = None
        if cbar:
            cbar_mappable = im
            if cbar_label is None:
                if unit.startswith('1/'):
                    cbar_name = 'gain'
                else:
                    cbar_name = 'phase'
                if mpl.rcParams['text.usetex'] and 'µ' in unit:  # Make sure µ works in latex:
                    mpl.rc('text.latex', preamble=R'\usepackage{txfonts},\usepackage{lmodern}')
                    unit = unit.replace('µ', R'$\muup$')  # Upright µ!
                cbar_label = u'{} [{}]'.format(cbar_name, unit)
        # Return formatted axis:
        return plottools.format_axis(axis, sampling=a, cbar_mappable=cbar_mappable,
                                     cbar_label=cbar_label, **kwargs)

    def plot_holo(self, gain='auto', colorwheel=True,  # specific to plot_holo!
                  cmap=None, interpolation='none', axis=None, figsize=None, sigma_clip=2,
                  **kwargs):
        """Display the color coded holography image.

        Parameters
        ----------
        gain : float or 'auto', optional
            The gain factor for determining the number of contour lines. The default is 'auto',
            which means that the gain will be determined automatically to look pretty.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method for the holographic contour map.
            No interpolation is used in the default case.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        Notes
        -----
        Uses :func:`~.plottools.format_axis` at the end. According keywords can also be given here.

        """
        self._log.debug('Calling plot_holo')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        # Calculate gain if 'auto' is selected:
        if gain == 'auto':
            gain = 4 * 2 * np.pi / (np.abs(self.phase).max() + 1E-30)
            gain = round(gain, -int(np.floor(np.log10(abs(gain)))))
        # Calculate the holography image intensity:
        holo = np.cos(gain * self.phase)
        holo += 1  # Shift to positive values
        holo /= 2  # Rescale to [0, 1]
        # Calculate the phase gradients:
        # B = rot(A)  --> B_x =  grad_y(A_z),   B_y = -grad_x(A_z); phi_m ~ -int(A_z)
        # for projection along +z:    sign switch --> B_x = -grad_y(phi_m), B_y =  grad_x(phi_m)
        # for projection along -z: NO sign switch --> B_x = grad_y(phi_m), B_y =  -grad_x(phi_m)
        grad_y, grad_x = np.gradient(self.phase, self.a, self.a)
        # Clip outliers:
        outlier_x = np.abs(grad_x - np.mean(grad_x)) < sigma_clip * np.std(grad_x)
        grad_x_sigma = np.where(outlier_x, grad_x, np.nan)
        grad_x_min, grad_x_max = np.nanmin(grad_x_sigma), np.nanmax(grad_x_sigma)
        grad_x = np.clip(grad_x, grad_x_min, grad_x_max)
        outlier_y = np.abs(grad_y - np.mean(grad_y)) < sigma_clip * np.std(grad_y)
        grad_y_sigma = np.where(outlier_y, grad_y, np.nan)
        grad_y_min, grad_y_max = np.nanmin(grad_y_sigma), np.nanmax(grad_y_sigma)
        grad_y = np.clip(grad_y, grad_y_min, grad_y_max)
        # Calculate colors:
        if cmap is None:
            cmap = colors.CMAP_CIRCULAR_DEFAULT
        vector = np.asarray((grad_y, -grad_x, np.zeros_like(grad_x)))
        rgb = cmap.rgb_from_vector(vector)
        rgb = (holo.T * rgb.T).T.astype(np.uint8)
        holo_image = Image.fromarray(rgb)
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Plot the image and set axes:
        axis.imshow(holo_image, origin='lower', interpolation=interpolation,
                    extent=(0, self.dim_uv[1], 0, self.dim_uv[0]))
        note = kwargs.pop('note', None)
        if note is None:
            note = 'gain: {:g}'.format(gain)
        stroke = kwargs.pop('stroke', 'k')  # Default for holo is white with black outline!
        return plottools.format_axis(axis, sampling=a, note=note, colorwheel=colorwheel,
                                     stroke=stroke, **kwargs)

    def plot_combined(self, title='', phase_title='', holo_title='', figsize=None,
                      colorwheel=True, **kwargs):
        """Display the phase map and the resulting color coded holography image in one plot.

        Parameters
        ----------
        title : string, optional
            The super title of the plot. The default is 'Combined Plot'.
        phase_title : string, optional
            The title of the phase map.
        holo_title : string, optional
            The title of the holographic contour map
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        phase_axis, holo_axis: :class:`~matplotlib.axes.AxesSubplot`
            The axes on which the graphs are plotted.

        Notes
        -----
        Uses :func:`~.plottools.format_axis` at the end. According keywords can also be given here.

        """
        self._log.debug('Calling plot_combined')
        # Create combined plot and set title:
        if figsize is None:
            figsize = (plottools.FIGSIZE_DEFAULT[0]*2 + 1, plottools.FIGSIZE_DEFAULT[1])
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)
        # Only phase is annotated, holo will show gain:
        note = kwargs.pop('note', None)
        # Plot holography image:
        holo_axis = fig.add_subplot(1, 2, 1, aspect='equal')
        self.plot_holo(axis=holo_axis, title=holo_title, note=None, colorwheel=colorwheel, **kwargs)
        # Plot phase map:
        phase_axis = fig.add_subplot(1, 2, 2, aspect='equal')
        self.plot_phase(axis=phase_axis, title=phase_title, note=note, **kwargs)
        # Return the plotting axes:
        return phase_axis, holo_axis

    def plot_phase_with_hist(self, bins='auto', unit='rad',
                             title='', phase_title='', hist_title='', figsize=None, **kwargs):
        """Display the phase map and a histogram of the phase values of all pixels.

        Parameters
        ----------
        bins : int or sequence of scalars or str, optional
            Bin argument that goes to the matplotlib.hist function (more documentation there).
            The default is 'auto', which tries to pick something nice.
        unit: {'rad', 'mrad', 'µrad', '1/rad', '1/mrad', '1/µrad'}, optional
            The plotting unit of the phase map. The phase is scaled accordingly before plotting.
            Inverse radians should be used for gain maps!
        title : string, optional
            The super title of the plot. The default is 'Combined Plot'.
        phase_title : string, optional
            The title of the phase map.
        hist_title : string, optional
            The title of the histogram.
        figsize : tuple of floats (N=2)
            Size of the plot figure.

        Returns
        -------
        phase_axis, holo_axis: :class:`~matplotlib.axes.AxesSubplot`
            The axes on which the graphs are plotted.

        Notes
        -----
        Uses :func:`~.plottools.format_axis` at the end. According keywords can also be given here.

        """
        self._log.debug('Calling plot_phase_with_hist')
        # Create combined plot and set title:
        if figsize is None:
            figsize = (plottools.FIGSIZE_DEFAULT[0]*2 + 1, plottools.FIGSIZE_DEFAULT[1])
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title, fontsize=20)
        # Plot histogram:
        hist_axis = fig.add_subplot(1, 2, 1)
        vec = self.phase_vec * self.UNITDICT[unit]  # Take units into consideration:
        # TODO: This is bad! Discard low confidence values completely instead! Otherwise peak at 0!
        # TODO: Set to nan and then discard with np.isnan()?
        vec *= np.where(self.confidence > 0.5, 1, 0).ravel()  # Discard low confidence points!
        hist_axis.hist(vec, bins=bins, histtype='stepfilled', color='g')
        # Format histogram:
        x0, x1 = hist_axis.get_xlim()
        y0, y1 = hist_axis.get_ylim()
        # TODO: Why the next line? Seems bad if you want to change things later´!
        hist_axis.set(aspect=np.abs(x1 - x0) / np.abs(y1 - y0) * 0.94)  # Last value because cbar!
        fontsize = kwargs.get('fontsize', 16)
        hist_axis.tick_params(axis='both', which='major', labelsize=fontsize)
        hist_axis.set_title(hist_title, fontsize=fontsize)
        hist_axis.set_xlabel('phase [{}]'.format(unit), fontsize=fontsize)
        hist_axis.set_ylabel('count', fontsize=fontsize)
        # Plot phase map:
        phase_axis = fig.add_subplot(1, 2, 2, aspect=1)
        self.plot_phase(unit=unit, axis=phase_axis, title=phase_title, **kwargs)
        # Return the plotting axes:
        return phase_axis, hist_axis

    def plot_phase3d(self, title='Phase Map', unit='rad', cmap='RdBu'):
        """Display the phasemap as a 3D surface with contourplots.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Phase Map'.
        unit: {'rad', 'mrad', 'µrad'}, optional
            The plotting unit of the phase map. The phase is scaled accordingly before plotting.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        """
        self._log.debug('Calling plot_phase3d')
        # Take units into consideration:
        phase = self.phase * self.UNITDICT[unit]
        # Create figure and axis:
        fig = plt.figure()
        axis = Axes3D(fig)
        # Plot surface and contours:
        vv, uu = np.indices(self.dim_uv)
        axis.plot_surface(uu, vv, phase, rstride=4, cstride=4, alpha=0.7, cmap=cmap,
                          linewidth=0, antialiased=False)
        axis.contourf(uu, vv, phase, 15, zdir='z', offset=np.min(phase), cmap=cmap)
        axis.set_title(title)
        axis.view_init(45, -135)
        axis.set_xlabel('u-axis [px]')
        axis.set_ylabel('v-axis [px]')
        axis.set_zlabel('phase shift [{}]'.format(unit))
        if self.dim_uv[0] >= self.dim_uv[1]:
            u_bin, v_bin = np.max((2, np.floor(9 * self.dim_uv[1] / self.dim_uv[0]))), 9
        else:
            u_bin, v_bin = 9, np.max((2, np.floor(9 * self.dim_uv[0] / self.dim_uv[1])))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=u_bin, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=v_bin, integer=True))
        # Return plotting axis:
        return axis

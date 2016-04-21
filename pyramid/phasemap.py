# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.PhaseMap` class for storing phase map data."""

import logging
import os
from numbers import Number

from pyramid.colormap import DirectionalColormap, TransparentColormap

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import zoom

_log = logging.getLogger(__name__)
try:  # Try importing HyperSpy:
    import hyperspy.api as hs
except ImportError:
    hs = None
    _log.error('Could not load hyperspy package!')

__all__ = ['PhaseMap']


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
    :func:`~.make_color_wheel` function. Use the :func:`~.display_combined` function to plot the
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
        Confidence array which determines the trust of specific regions of the phase_map. A value
        of 1 means the pixel is trustworthy, a value of 0 means it is not. Defaults to an array of
        ones (full trust for all pixels). Can be used for the construction of Se_inv.
    unit: {'rad', 'mrad'}, optional
        Set the unit of the phase map. This is important for the :func:`display` function,
        because the phase is scaled accordingly. Does not change the phase itself, which is
        always in `rad`.

    """

    _log = logging.getLogger(__name__)

    UNITDICT = {u'rad': 1E0,
                u'mrad': 1E3,
                u'µrad': 1E6}

    CDICT = {'red': [(0.00, 1.0, 0.0),
                     (0.25, 1.0, 1.0),
                     (0.50, 1.0, 1.0),
                     (0.75, 0.0, 0.0),
                     (1.00, 0.0, 1.0)],

             'green': [(0.00, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.50, 1.0, 1.0),
                       (0.75, 1.0, 1.0),
                       (1.00, 0.0, 1.0)],

             'blue': [(0.00, 1.0, 1.0),
                      (0.25, 0.0, 0.0),
                      (0.50, 0.0, 0.0),
                      (0.75, 0.0, 0.0),
                      (1.00, 1.0, 1.0)]}

    CDICT_INV = {'red': [(0.00, 0.0, 1.0),
                         (0.25, 0.0, 0.0),
                         (0.50, 0.0, 0.0),
                         (0.75, 1.0, 1.0),
                         (1.00, 1.0, 0.0)],

                 'green': [(0.00, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 0.0, 0.0),
                           (1.00, 1.0, 0.0)],

                 'blue': [(0.00, 0.0, 0.0),
                          (0.25, 1.0, 1.0),
                          (0.50, 1.0, 1.0),
                          (0.75, 1.0, 1.0),
                          (1.00, 0.0, 0.0)]}

    HOLO_CMAP = LinearSegmentedColormap('my_colormap', CDICT, 256)
    HOLO_CMAP_INV = LinearSegmentedColormap('my_colormap', CDICT_INV, 256)

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
        assert len(phase.shape) == 2, 'Phase has to be 2-dimensional!'
        self._phase = phase.astype(dtype=np.float32)
        self._dim_uv = phase.shape

    @property
    def phase_vec(self):
        """Vector containing the phase shift."""
        return np.reshape(self.phase, -1)

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
        """Confidence array which determines the trust of specific regions of the phase_map."""
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        if confidence is not None:
            assert confidence.shape == self.phase.shape, \
                'Confidence and phase dimensions must match!'
        else:
            confidence = np.ones_like(self.phase)
        self._confidence = confidence.astype(dtype=np.float32)

    @property
    def unit(self):
        """The unit of the phase map. Default is `rad`."""
        return self._unit

    @unit.setter
    def unit(self, unit):
        assert unit in self.UNITDICT, 'Unit {} not supported!'.format(unit)
        self._unit = unit

    def __init__(self, a, phase, mask=None, confidence=None, unit='rad'):
        self._log.debug('Calling __init__')
        self.a = a
        self.phase = phase
        self.mask = mask
        self.confidence = confidence
        self.unit = unit
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, phase=%r, mask=%r, confidence=%r, unit=%r)' % \
               (self.__class__, self.a, self.phase, self.mask, self.confidence, self.unit)

    def __str__(self):
        self._log.debug('Calling __str__')
        return 'PhaseMap(a=%s, dim_uv=%s, mask=%s)' % (self.a, self.dim_uv, not np.all(self.mask))

    def __neg__(self):  # -self
        self._log.debug('Calling __neg__')
        return PhaseMap(self.a, -self.phase, self.mask, self.confidence, self.unit)

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
            conf_comb = (self.confidence + other.confidence) / 2  # confidence averaged!
            return PhaseMap(self.a, self.phase + other.phase, mask_comb, conf_comb, self.unit)
        else:  # other is a Number
            self._log.debug('Adding an offset')
            return PhaseMap(self.a, self.phase + other, self.mask, self.confidence, self.unit)

    def __sub__(self, other):  # self - other
        self._log.debug('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        assert (isinstance(other, Number) or
                (isinstance(other, np.ndarray) and other.shape == self.dim_uv)), \
            'PhaseMap objects can only be multiplied by scalar numbers or fitting arrays!'
        return PhaseMap(self.a, other * self.phase, self.mask, self.confidence, self.unit)

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

    def copy(self):
        """Returns a copy of the :class:`~.PhaseMap` object

        Returns
        -------
        phase_map: :class:`~.PhaseMap`
            A copy of the :class:`~.PhaseMap`.

        """
        self._log.debug('Calling copy')
        return PhaseMap(self.a, self.phase.copy(), self.mask.copy(),
                        self.confidence.copy(), self.unit)

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
        self.a *= 2 ** n
        for t in range(n):
            # Pad if necessary:
            pv, pu = self.dim_uv[0] % 2, self.dim_uv[1] % 2
            if pv != 0 or pu != 0:
                self.phase = np.pad(self.phase, ((0, pv), (0, pu)), mode='constant')
            # Create coarser grid for the magnetization:
            dim_uv = self.dim_uv
            self.phase = self.phase.reshape((dim_uv[0] / 2, 2, dim_uv[1] / 2, 2)).mean(axis=(3, 1))
            mask = self.mask.reshape(dim_uv[0] / 2, 2, dim_uv[1] / 2, 2)
            self.mask = mask[:, 0, :, 0] & mask[:, 1, :, 0] & mask[:, 0, :, 1] & mask[:, 1, :, 1]
            self.confidence = self.confidence.reshape(dim_uv[0] / 2, 2,
                                                      dim_uv[1] / 2, 2).mean(axis=(3, 1))

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
        self.a /= 2 ** n
        self.phase = zoom(self.phase, zoom=2 ** n, order=order)
        self.mask = zoom(self.mask, zoom=2 ** n, order=0)
        self.confidence = zoom(self.confidence, zoom=2 ** n, order=order)

    def pad(self, pad_values, masked=True):
        """Pad the current phase map with zeros for each individual axis.

        Parameters
        ----------
        pad_values : tuple of int
            Number of zeros which should be padded. Provided as a tuple where each entry
            corresponds to an axis. An entry can be one int (same padding for both sides) or again
            a tuple which specifies the pad values for both sides of the corresponding axis.
        masked: boolean
            Determines if the padded areas should be masked or not. Defaults to `True` and thus
            creates a 'buffer zone' for the magnetization distribution in the reconstruction.

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
        pv = np.zeros(4, dtype=np.int)
        for i, values in enumerate(pad_values):
            assert np.shape(values) in [(), (2,)], 'Only one or two values per axis can be given!'
            pv[2 * i:2 * (i + 1)] = values
        self.phase = np.pad(self.phase, ((pv[0], pv[1]), (pv[2], pv[3])), mode='constant')
        self.mask = np.pad(self.mask, ((pv[0], pv[1]), (pv[2], pv[3])), mode='constant',
                           constant_values=masked)
        self.confidence = np.pad(self.confidence, ((pv[0], pv[1]), (pv[2], pv[3])),
                                 mode='constant')

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
        self.phase = self.phase[cv[0]:cv[1], cv[2]:cv[3]]
        self.mask = self.mask[cv[0]:cv[1], cv[2]:cv[3]]
        self.confidence = self.confidence[cv[0]:cv[1], cv[2]:cv[3]]

    def to_signal(self):
        """Convert :class:`~.PhaseMap` data into a HyperSpy Image.

        Returns
        -------
        signal: :class:`~hyperspy.signals.Image`
            Representation of the :class:`~.PhaseMap` object as a HyperSpy Image.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        self._log.debug('Calling to_signal')
        if hs is None:
            self._log.error('This method recquires the hyperspy package!')
            return
        # Create signal:
        signal = hs.signals.Image(self.phase)
        # Set axes:
        signal.axes_manager[0].name = 'x-axis'
        signal.axes_manager[0].units = 'nm'
        signal.axes_manager[0].scale = self.a
        signal.axes_manager[1].name = 'y-axis'
        signal.axes_manager[1].units = 'nm'
        signal.axes_manager[1].scale = self.a
        # Set metadata:
        signal.metadata.Signal.title = 'PhaseMap'
        signal.metadata.Signal.unit = self.unit
        signal.metadata.Signal.mask = self.mask
        signal.metadata.Signal.confidence = self.confidence
        # Create and return EMD:
        return signal

    @classmethod
    def from_signal(cls, signal):
        """Convert a :class:`~hyperspy.signals.Image` object to a :class:`~.PhaseMap` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Image`
            The :class:`~hyperspy.signals.Image` object which should be converted to a PhaseMap.

        Returns
        -------
        phase_map: :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object containing the loaded data.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        cls._log.debug('Calling from_signal')
        # Extract phase:
        phase = signal.data
        # Extract properties:
        a = signal.axes_manager[0].scale
        try:
            unit = signal.metadata.Signal.unit
            mask = signal.metadata.Signal.mask
            confidence = signal.metadata.Signal.confidence
        except AttributeError:
            unit = 'rad'
            mask = None
            confidence = None
        return cls(a, phase, mask, confidence, unit)

    def save_to_hdf5(self, filename='phasemap.hdf5', *args, **kwargs):
        """Save magnetization data in a file with HyperSpys HDF5-format.

        Parameters
        ----------
        filename : string, optional
            The name of the HyperSpy-file in which to store the phase map.
            The default is 'phasemap.hdf5' in the phasemap folder.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_hdf5')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'phasemap')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        # Save data to file:
        self.to_signal().save(filename, *args, **kwargs)

    @classmethod
    def load_from_hdf5(cls, filename):
        """Construct :class:`~.DataMag` object from HyperSpys HDF5-file.

        Parameters
        ----------
        filename : string
            The name of the HDF5-file from which to load the data. Standard format is '\*.hdf5'.

        Returns
        -------
        mag_data: :class:`~.VectorData`
            A :class:`~.VectorData` object containing the loaded data.

        """
        cls._log.debug('Calling load_from_hdf5')
        if hs is None:
            cls._log.error('This method recquires the hyperspy package!')
            return
        # Use relative path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'phasemap')
            filename = os.path.join(directory, filename)
        # Load data from file:
        return PhaseMap.from_signal(hs.load(filename))

    def save_to_txt(self, filename='phasemap.txt', skip_header=False):
        """Save :class:`~.PhaseMap` data in a file with txt-format.

        Parameters
        ----------
        filename : string
            The name of the file in which to store the phase map data.
            The default is '..\output\phasemap.txt'.
        skip_header : boolean, optional
            Determines if the header, should be skipped (useful for some other programs).
            Default is False.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_txt')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'phasemap')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        # Save data to file:
        with open(filename, 'w') as phase_file:
            if not skip_header:
                phase_file.write('{}\n'.format(filename.replace('.txt', '')))
                phase_file.write('grid spacing = {} nm\n'.format(self.a))
            np.savetxt(phase_file, self.phase, fmt='%7.6e', delimiter='\t')

    @classmethod
    def load_from_txt(cls, filename):
        """Construct :class:`~.PhaseMap` object from a human readable txt-file.

        Parameters
        ----------
        filename : string
            The name of the file from which to load the data.

        Returns
        -------
        phase_map : :class:`~.PhaseMap`
            A :class:`~.PhaseMap` object containing the loaded data.

        Notes
        -----
        Does not recover the mask, confidence or unit of the original phase map, which default to
        `None`, `None` and `'rad'`, respectively.

        """
        cls._log.debug('Calling load_from_txt')
        # Use relative path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'phasemap')
            filename = os.path.join(directory, filename)
        # Load data from file:
        with open(filename, 'r') as phase_file:
            phase_file.readline()  # Headerline is not used
            a = float(phase_file.readline()[15:-4])
            phase = np.loadtxt(filename, delimiter='\t', skiprows=2)
        return cls(a, phase)

    def display_phase(self, title='Phase Map', cmap='RdBu', limit=None,
                      norm=None, axis=None, cbar=True, show_mask=True, show_conf=True):
        """Display the phasemap as a colormesh.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Phase Map'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.
        limit : float, optional
            Plotlimit for the phase in both negative and positive direction (symmetric around 0).
            If not specified, the maximum amplitude of the phase is used.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
            If not specified, :class:`~matplotlib.colors.Normalize` is automatically used.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        cbar : bool, optional
            A switch determining if the colorbar should be plotted or not. Default is True.
        show_mask : bool, optional
            A switch determining if the mask should be plotted or not. Default is True.
        show_conf : float, optional
            A switch determining if the confidence should be plotted or not. Default is True.
        Returns
        -------
        axis, cbar: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted and the colorbar.

        """
        self._log.debug('Calling display_phase')
        # Take units into consideration:
        phase = self.phase * self.UNITDICT[self.unit]
        if limit is None:
            limit = np.max(np.abs(phase))
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure(figsize=(7, 7))
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Plot the phasemap:
        im = axis.pcolormesh(phase, cmap=cmap, vmin=-limit, vmax=limit, norm=norm)
        if show_mask or show_conf:
            vv, uu = np.indices(self.dim_uv) + 0.5
            if show_mask and not np.all(self.mask):  # Plot mask if desired and not trivial!
                axis.contour(uu, vv, self.mask, levels=[0.5], colors='k', linestyles='dotted')
            if show_conf and not np.all(self.confidence == 1.0):
                colormap = TransparentColormap(0.2, 0.3, 0.2, [0.2, 0.])
                axis.pcolormesh(self.confidence, cmap=colormap)
        # Set the axes ticks and labels:
        if self.dim_uv[0] >= self.dim_uv[1]:
            u_bin, v_bin = np.max((2, np.floor(9 * self.dim_uv[1] / self.dim_uv[0]))), 9
        else:
            u_bin, v_bin = 9, np.max((2, np.floor(9 * self.dim_uv[0] / self.dim_uv[1])))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=u_bin, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=v_bin, integer=True))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x * self.a)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '{:g}'.format(x * self.a)))
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.set_title(title, fontsize=18)
        axis.set_xlim(0, self.dim_uv[1])
        axis.set_ylim(0, self.dim_uv[0])
        axis.set_xlabel('u-axis [nm]', fontsize=15)
        axis.set_ylabel('v-axis [nm]', fontsize=15)
        # Add colorbar:
        if cbar:
            fig = plt.gcf()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(u'phase shift [{}]'.format(self.unit), fontsize=15)
        # Return plotting axis:
        return axis

    def display_phase3d(self, title='Phase Map', cmap='RdBu'):
        """Display the phasemap as a 3-D surface with contourplots.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Phase Map'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        """
        self._log.debug('Calling display_phase3d')
        # Take units into consideration:
        phase = self.phase * self.UNITDICT[self.unit]
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
        axis.set_zlabel('phase shift [{}]'.format(self.unit))
        # Return plotting axis:
        return axis

    def display_holo(self, title=None, gain='auto', axis=None, grad_encode='bright',
                     interpolation='none'):
        """Display the color coded holography image.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Contour Map (gain: %g)' % gain.
        gain : float or 'auto', optional
            The gain factor for determining the number of contour lines. The default is 'auto',
            which means that the gain will be determined automatically to look pretty.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        grad_encode: {'bright', 'dark', 'color', 'none'}, optional
            Encoding mode of the phase gradient. 'none' produces a black-white image, 'color' just
            encodes the direction (without gradient strength), 'dark' modulates the gradient
            strength with a factor between 0 and 1 and 'bright' (which is the default) encodes
            the gradient strength with color saturation.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method. No interpolation is used in the default case.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        """
        self._log.debug('Calling display_holo')
        # Calculate gain if 'auto' is selected:
        if gain == 'auto':
            gain = 4 * 2 * np.pi / (np.abs(self.phase).max() + 1E-30)
        # Set title if not set:
        if title is None:
            title = 'Contour Map (gain: %.2g)' % gain
        # Calculate the holography image intensity:
        holo = np.cos(gain * self.phase)
        holo += 1  # Shift to positive values
        holo /= 2  # Rescale to [0, 1]
        # Calculate the phase gradients, expressed by amplitude and angle:
        phase_grad_x, phase_grad_y = np.gradient(self.phase, self.a, self.a)
        angles = (1 - np.arctan2(phase_grad_y, phase_grad_x) / np.pi) / 2
        phase_grad_amp = np.hypot(phase_grad_y, phase_grad_x)
        saturations = np.sin(
            phase_grad_amp / (phase_grad_amp.max() + 1E-30) * np.pi / 2)  # betw. 0 and 1
        # Calculate color encoding:
        if grad_encode == 'dark':
            pass
        elif grad_encode == 'bright':
            saturations = 2 - saturations
        elif grad_encode == 'color':
            saturations = np.ones_like(saturations)
        elif grad_encode == 'none':
            saturations = 2 * np.ones_like(saturations)
        else:
            raise AssertionError('Gradient encoding not recognized!')
        # Calculate colored holo image:
        rgb = DirectionalColormap.rgb_from_colorind_and_saturation(angles, saturations)
        rgb = (holo.T * rgb.T).T.astype(np.uint8)
        holo_image = Image.fromarray(rgb)
        # If no axis is specified, a new figure is created:
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Plot the image and set axes:
        axis.imshow(holo_image, origin='lower', interpolation=interpolation,
                    extent=(0, self.dim_uv[1], 0, self.dim_uv[0]))
        # Set the title and the axes labels:
        axis.set_title(title)
        axis.tick_params(axis='both', which='major', labelsize=14)
        axis.set_title(title, fontsize=18)
        axis.set_xlabel('u-axis [px]', fontsize=15)
        axis.set_ylabel('v-axis [px]', fontsize=15)
        if self.dim_uv[0] >= self.dim_uv[1]:
            u_bin, v_bin = np.max((2, np.floor(9 * self.dim_uv[1] / self.dim_uv[0]))), 9
        else:
            u_bin, v_bin = 9, np.max((2, np.floor(9 * self.dim_uv[0] / self.dim_uv[1])))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=u_bin, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=v_bin, integer=True))
        # Return plotting axis:
        return axis

    def display_combined(self, title='Combined Plot', cmap='RdBu', limit=None, norm=None,
                         gain='auto', interpolation='none', grad_encode='bright',
                         cbar=True, show_mask=True, show_conf=True):
        """Display the phase map and the resulting color coded holography image in one plot.

        Parameters
        ----------
        title : string, optional
            The title of the plot. The default is 'Combined Plot'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            The default is 'RdBu'.
        limit : float, optional
            Plotlimit for the phase in both negative and positive direction (symmetric around 0).
            If not specified, the maximum amplitude of the phase is used.
        norm : :class:`~matplotlib.colors.Normalize` or subclass, optional
            Norm, which is used to determine the colors to encode the phase information.
            If not specified, :class:`~matplotlib.colors.Normalize` is automatically used.
        gain : float or 'auto', optional
            The gain factor for determining the number of contour lines. The default is 'auto',
            which means that the gain will be determined automatically to look pretty.
        interpolation : {'none, 'bilinear', 'cubic', 'nearest'}, optional
            Defines the interpolation method for the holographic contour map.
            No interpolation is used in the default case.
        grad_encode: {'bright', 'dark', 'color', 'none'}, optional
            Encoding mode of the phase gradient. 'none' produces a black-white image, 'color' just
            encodes the direction (without gradient strength), 'dark' modulates the gradient
            strength with a factor between 0 and 1 and 'bright' (which is the default) encodes
            the gradient strength with color saturation.
        cbar : bool, optional
            A switch determining if the colorbar should be plotted or not. Default is True.
        show_mask : bool, optional
            A switch determining if the mask should be plotted or not. Default is True.
        show_conf : float, optional
            A switch determining if the confidence should be plotted or not. Default is True.

        Returns
        -------
        phase_axis, holo_axis: :class:`~matplotlib.axes.AxesSubplot`
            The axes on which the graphs are plotted.

        """
        self._log.debug('Calling display_combined')
        # Create combined plot and set title:
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle(title, fontsize=20)
        # Plot holography image:
        holo_axis = fig.add_subplot(1, 2, 1, aspect='equal')
        self.display_holo(gain=gain, axis=holo_axis, interpolation=interpolation,
                          grad_encode=grad_encode)
        # Plot phase map:
        phase_axis = fig.add_subplot(1, 2, 2, aspect='equal')
        fig.subplots_adjust(right=0.85)
        self.display_phase(cmap=cmap, limit=limit, norm=norm, axis=phase_axis,
                           cbar=cbar, show_mask=show_mask, show_conf=show_conf)
        # Return the plotting axes:
        return phase_axis, holo_axis

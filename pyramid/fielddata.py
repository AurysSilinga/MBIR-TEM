# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides classes for storing vector and scalar 3D-field."""

from __future__ import division

import logging
import os
from abc import ABCMeta, abstractmethod
from numbers import Number

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.interpolation import zoom

from pyramid import fft
from pyramid.colormap import DirectionalColormap

_log = logging.getLogger(__name__)
try:  # Try importing HyperSpy:
    import hyperspy.api as hs
except ImportError:
    hs = None
    _log.error('Could not load hyperspy package!')

__all__ = ['VectorData', 'ScalarData']


class FieldData(object):
    """Class for storing field data.

    Abstract base class for the representatio of magnetic or electric fields (see subclasses).
    Fields can be accessed as 3D numpy arrays via the `field` property or as a vector via
    `field_vec`. :class:`~.FieldData` objects support negation, arithmetic operators
    (``+``, ``-``, ``*``) and their augmented counterparts (``+=``, ``-=``, ``*=``), with numbers
    and other :class:`~.FieldData` objects of the same subclass, if their dimensions and grid
    spacings match. It is possible to load data from HDF5 or LLG (.txt) files or to save the data
    in these formats. Specialised plotting methods are also provided.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    field: :class:`~numpy.ndarray` (N=4)
        The field distribution for every 3D-gridpoint.

    """

    __metaclass__ = ABCMeta
    _log = logging.getLogger(__name__ + '.FieldData')

    @property
    def a(self):
        """The grid spacing in nm."""
        return self._a

    @a.setter
    def a(self, a):
        assert isinstance(a, Number), 'Grid spacing has to be a number!'
        assert a >= 0, 'Grid spacing has to be a positive number!'
        self._a = float(a)

    @property
    def shape(self):
        """The shape of the `field` (3D for scalar, 4D vor vector field)."""
        return self.field.shape

    @property
    def dim(self):
        """Dimensions (z, y, x) of the grid, only 3D coordinates, without components if present."""
        return self.shape[-3:]

    @property
    def field(self):
        """The field strength for every 3D-gridpoint (scalar: 3D, vector: 4D)."""
        return self._field

    @field.setter
    def field(self, field):
        assert isinstance(field, np.ndarray), 'Field has to be a numpy array!'
        assert 3 <= len(field.shape) <= 4, 'Field has to be 3- or 4-dimensional (scalar / vector)!'
        if len(field.shape) == 4:
            assert field.shape[0] == 3, 'A vector field has to have exactly 3 components!'
        self._field = np.asarray(field, dtype=fft.FLOAT)

    @property
    def field_amp(self):
        """The field amplitude (returns the field itself for scalar and the vector amplitude
        calculated via a square sum for a vector field."""
        if len(self.shape) == 4:
            return np.sqrt(np.sum(self.field ** 2, axis=0))
        else:
            return self.field

    @property
    def field_vec(self):
        """Vector containing the vector field distribution."""
        return np.reshape(self.field, -1)

    @field_vec.setter
    def field_vec(self, mag_vec):
        mag_vec = np.asarray(mag_vec, dtype=fft.FLOAT)
        assert np.size(mag_vec) == np.prod(self.shape), \
            'Vector has to match field shape! {} {}'.format(mag_vec.shape, np.prod(self.shape))
        self.field = mag_vec.reshape((3,) + self.dim)

    def __init__(self, a, field):
        self._log.debug('Calling __init__')
        self.a = a
        self.field = field
        self._log.debug('Created ' + str(self))

    def __repr__(self):
        self._log.debug('Calling __repr__')
        return '%s(a=%r, field=%r)' % (self.__class__, self.a, self.field)

    def __str__(self):
        self._log.debug('Calling __str__')
        return '%s(a=%s, dim=%s)' % (self.__class__, self.a, self.dim)

    def __neg__(self):  # -self
        self._log.debug('Calling __neg__')
        return VectorData(self.a, -self.field)

    def __add__(self, other):  # self + other
        self._log.debug('Calling __add__')
        assert isinstance(other, (FieldData, Number)), \
            'Only FieldData objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, FieldData):
            self._log.debug('Adding two VectorData objects')
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.shape == self.shape, 'Added field has to have the same dimensions!'
            return self.__init__(self.a, self.field + other.field)
        else:  # other is a Number
            self._log.debug('Adding an offset')
            return self.__init__(self.a, self.field + other)

    def __sub__(self, other):  # self - other
        self._log.debug('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        assert isinstance(other, Number), 'FieldData objects can only be multiplied by numbers!'
        return VectorData(self.a, other * self.field)

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
        """Returns a copy of the :class:`~.FieldData` object

        Returns
        -------
        field_data: :class:`~.FieldData`
            A copy of the :class:`~.FieldData`.

        """
        self._log.debug('Calling copy')
        return self.__init__(self.a, self.field.copy())

    def get_mask(self, threshold=0):
        """Mask all pixels where the amplitude of the field lies above `threshold`.

        Parameters
        ----------
        threshold : float, optional
            A pixel only gets masked, if it lies above this threshold . The default is 0.

        Returns
        -------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Mask of the pixels where the amplitude of the field lies above `threshold`.

        """
        self._log.debug('Calling get_mask')
        return np.where(self.field_amp > threshold, True, False)

    def contour_plot3d(self, title='Field Distribution', contours=10, opacity=0.25):
        """Plot the field as a 3D-contour plot.

        Parameters
        ----------
        title: string, optional
            The title for the plot.
        contours: int, optional
            Number of contours which should be plotted.
        opacity: float, optional
            Defines the opacity of the contours. Default is 0.25.

        Returns
        -------
        plot : :class:`mayavi.modules.vectors.Vectors`
            The plot object.


        """
        self._log.debug('Calling quiver_plot3D')
        from mayavi import mlab
        # Plot them as vectors:
        mlab.figure(size=(750, 700))
        plot = mlab.contour3d(self.field_amp, contours=contours, opacity=opacity)
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.title(title, height=0.95, size=0.35)
        mlab.orientation_axes()
        # mlab.show()
        return plot

    @abstractmethod
    def scale_down(self, n):
        """Scale down the field distribution by averaging over two pixels along each axis.

        Parameters
        ----------
        n : int, optional
            Number of times the field distribution is scaled down. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        Only possible, if each axis length is a power of 2!

        """
        pass

    @abstractmethod
    def scale_up(self, n, order):
        """Scale up the field distribution using spline interpolation of the requested order.

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
        pass

    @abstractmethod
    def get_vector(self, mask):
        """Returns the field as a vector, specified by a mask.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Masks the pixels from which the entries should be taken.

        Returns
        -------
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing the field of the specified pixels.

        """
        pass

    @abstractmethod
    def set_vector(self, vector, mask):
        """Set the field of the masked pixels to the values specified by `vector`.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean), optional
            Masks the pixels from which the field should be taken.
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing the field of the specified pixels.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def to_signal(self):
        """Convert :class:`~.FieldData` data into a HyperSpy signal.

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal`
            Representation of the :class:`~.FieldData` object as a HyperSpy Signal.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        pass

    @abstractmethod
    def save_to_hdf5(self, filename):
        """Save field data in a file with HyperSpys HDF5-format.

        Parameters
        ----------
        filename : string, optional
            The name of the HDF5-file in which to store the field data.

        Returns
        -------
        None

        """
        pass


class VectorData(FieldData):
    """Class for storing vector ield data.

    Represents 3-dimensional vector field distributions with 3 components which are stored as a
    3-dimensional numpy array in `field`, but which can also be accessed as a vector via
    `field_vec`. :class:`~.VectorData` objects support negation, arithmetic operators
    (``+``, ``-``, ``*``) and their augmented counterparts (``+=``, ``-=``, ``*=``), with numbers
    and other :class:`~.VectorData` objects, if their dimensions and grid spacings match. It is
    possible to load data from HDF5 or LLG (.txt) files or to save the data in these formats.
    Plotting methods are also provided.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    field: :class:`~numpy.ndarray` (N=4)
        The `x`-, `y`- and `z`-component of the vector field for every 3D-gridpoint
        as a 4-dimensional numpy array (first dimension has to be 3, because of the 3 components).

    """
    _log = logging.getLogger(__name__ + '.VectorData')

    def __init__(self, a, field):
        self._log.debug('Calling __init__')
        super(VectorData, self).__init__(a, field)
        self._log.debug('Created ' + str(self))

    def scale_down(self, n=1):
        """Scale down the field distribution by averaging over two pixels along each axis.

        Parameters
        ----------
        n : int, optional
            Number of times the field distribution is scaled down. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        Only possible, if each axis length is a power of 2!

        """
        self._log.debug('Calling scale_down')
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        self.a *= 2 ** n
        for t in range(n):
            # Pad if necessary:
            pz, py, px = self.dim[0] % 2, self.dim[1] % 2, self.dim[2] % 2
            if pz != 0 or py != 0 or px != 0:
                self.field = np.pad(self.field, ((0, 0), (0, pz), (0, py), (0, px)),
                                    mode='constant')
            # Create coarser grid for the vector field:
            shape_4d = (3, self.dim[0] / 2, 2, self.dim[1] / 2, 2, self.dim[2] / 2, 2)
            self.field = self.field.reshape(shape_4d).mean(axis=(6, 4, 2))

    def scale_up(self, n=1, order=0):
        """Scale up the field distribution using spline interpolation of the requested order.

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
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, (int, long)), \
            'order must be a positive integer between 0 and 5!'
        self.a /= 2 ** n
        self.field = np.array((zoom(self.field[0], zoom=2 ** n, order=order),
                               zoom(self.field[1], zoom=2 ** n, order=order),
                               zoom(self.field[2], zoom=2 ** n, order=order)))

    def pad(self, pad_values):
        """Pad the current field distribution with zeros for each individual axis.

        Parameters
        ----------
        pad_values : tuple of int
            Number of zeros which should be padded. Provided as a tuple where each entry
            corresponds to an axis. An entry can be one int (same padding for both sides) or again
            a tuple which specifies the pad values for both sides of the corresponding axis.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions accordingly.
        """
        self._log.debug('Calling pad')
        assert len(pad_values) == 3, 'Pad values for each dimension have to be provided!'
        pv = np.zeros(6, dtype=np.int)
        for i, values in enumerate(pad_values):
            assert np.shape(values) in [(), (2,)], 'Only one or two values per axis can be given!'
            pv[2 * i:2 * (i + 1)] = values
        self.field = np.pad(self.field,
                            ((0, 0), (pv[0], pv[1]), (pv[2], pv[3]), (pv[4], pv[5])),
                            mode='constant')

    def crop(self, crop_values):
        """Crop the current field distribution with zeros for each individual axis.

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
        assert len(crop_values) == 3, 'Crop values for each dimension have to be provided!'
        cv = np.zeros(6, dtype=np.int)
        for i, values in enumerate(crop_values):
            assert np.shape(values) in [(), (2,)], 'Only one or two values per axis can be given!'
            cv[2 * i:2 * (i + 1)] = values
        cv *= np.resize([1, -1], len(cv))
        cv = np.where(cv == 0, None, cv)
        self.field = self.field[:, cv[0]:cv[1], cv[2]:cv[3], cv[4]:cv[5]]

    def get_vector(self, mask):
        """Returns the vector field components arranged in a vector, specified by a mask.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Masks the pixels from which the components should be taken.

        Returns
        -------
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing vector field components of the specified pixels.
            Order is: first all `x`-, then all `y`-, then all `z`-components.

        """
        self._log.debug('Calling get_vector')
        if mask is not None:
            return np.reshape([self.field[0][mask],
                               self.field[1][mask],
                               self.field[2][mask]], -1)
        else:
            return self.field_vec

    def set_vector(self, vector, mask=None):
        """Set the field components of the masked pixels to the values specified by `vector`.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean), optional
            Masks the pixels from which the components should be taken.
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing vector field components of the specified pixels.
            Order is: first all `x`-, then all `y-, then all `z`-components.

        Returns
        -------
        None

        """
        self._log.debug('Calling set_vector')
        vector = np.asarray(vector, dtype=fft.FLOAT)
        assert np.size(vector) % 3 == 0, 'Vector has to contain all 3 components for every pixel!'
        count = np.size(vector) // 3
        if mask is not None:
            self.field[0][mask] = vector[:count]  # x-component
            self.field[1][mask] = vector[count:2 * count]  # y-component
            self.field[2][mask] = vector[2 * count:]  # z-component
        else:
            self.field_vec = vector

    def flip(self, axis='x'):
        """Flip/mirror the vector field around the specified axis.

        Parameters
        ----------
        axis: {'x', 'y', 'z'}, optional
            The axis around which the vector field is flipped.

        Returns
        -------
        mag_data_flip: :class:`~.VectorData`
           A flipped copy of the :class:`~.VectorData` object.

        """
        self._log.debug('Calling flip')
        if axis == 'x':
            mag_x, mag_y, mag_z = self.field[:, :, :, ::-1]
            field_flip = np.array((-mag_x, mag_y, mag_z))
        elif axis == 'y':
            mag_x, mag_y, mag_z = self.field[:, :, ::-1, :]
            field_flip = np.array((mag_x, -mag_y, mag_z))
        elif axis == 'z':
            mag_x, mag_y, mag_z = self.field[:, ::-1, :, :]
            field_flip = np.array((mag_x, mag_y, -mag_z))
        else:
            raise ValueError("Wrong input! 'x', 'y', 'z' allowed!")
        return VectorData(self.a, field_flip)

    def rot90(self, axis='x'):
        """Rotate the vector field 90° around the specified axis (right hand rotation).

        Parameters
        ----------
        axis: {'x', 'y', 'z'}, optional
            The axis around which the vector field is rotated.

        Returns
        -------
        mag_data_rot: :class:`~.VectorData`
           A rotated copy of the :class:`~.VectorData` object.

        """
        self._log.debug('Calling rot90')
        if axis == 'x':
            field_rot = np.zeros((3, self.dim[1], self.dim[0], self.dim[2]))
            for i in range(self.dim[2]):
                mag_x, mag_y, mag_z = self.field[:, :, :, i]
                mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x), np.rot90(mag_y), np.rot90(mag_z)
                field_rot[:, :, :, i] = np.array((mag_xrot, mag_zrot, -mag_yrot))
        elif axis == 'y':
            field_rot = np.zeros((3, self.dim[2], self.dim[1], self.dim[0]))
            for i in range(self.dim[1]):
                mag_x, mag_y, mag_z = self.field[:, :, i, :]
                mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x), np.rot90(mag_y), np.rot90(mag_z)
                field_rot[:, :, i, :] = np.array((mag_zrot, mag_yrot, -mag_xrot))
        elif axis == 'z':
            field_rot = np.zeros((3, self.dim[0], self.dim[2], self.dim[1]))
            for i in range(self.dim[0]):
                mag_x, mag_y, mag_z = self.field[:, i, :, :]
                mag_xrot, mag_yrot, mag_zrot = np.rot90(mag_x), np.rot90(mag_y), np.rot90(mag_z)
                field_rot[:, i, :, :] = np.array((mag_yrot, -mag_xrot, mag_zrot))
        else:
            raise ValueError("Wrong input! 'x', 'y', 'z' allowed!")
        return VectorData(self.a, field_rot)

    def get_slice(self, ax_slice=0, proj_axis='z', mode='complex'):
        """Extract a slice from the :class:`~.VectorData` object.

        Parameters
        ----------
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which the slice is taken. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Defaults to zero (first slice).
        mode : {'complex', 'amplitude'}, optional
            Determines if the 2D vector field is returned as complex values or if the amplitude
            of the two components is calculated.

        Returns
        -------
        mag_slice : :class:`~numpy.ndarray` (N=2)
            The extracted vector field slice.

        """
        self._log.debug('Calling get_slice')
        # Find slice:
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            u_mag = np.copy(self.field[0][ax_slice, ...])  # x-component
            v_mag = np.copy(self.field[1][ax_slice, ...])  # y-component
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            u_mag = np.copy(self.field[0][:, ax_slice, :])  # x-component
            v_mag = np.copy(self.field[2][:, ax_slice, :])  # z-component
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            u_mag = np.swapaxes(np.copy(self.field[2][..., ax_slice]), 0, 1)  # z-component
            v_mag = np.swapaxes(np.copy(self.field[1][..., ax_slice]), 0, 1)  # y-component
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        # Create data field:
        if mode == 'complex':
            return u_mag + 1j * v_mag
        elif mode == 'amplitude':
            return np.hypot(u_mag, v_mag)
        else:
            raise ValueError('Given mode not understood!')

    def to_signal(self):
        """Convert :class:`~.VectorData` data into a HyperSpy signal.

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal`
            Representation of the :class:`~.VectorData` object as a HyperSpy Signal.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        self._log.debug('Calling to_signal')
        # Try importing HyperSpy:
        if hs is None:
            self._log.error('This method recquires the hyperspy package!')
            return
        # Create signal:
        signal = hs.signals.Signal(np.rollaxis(self.field, 0, 4))
        # Set axes:
        signal.axes_manager[0].name = 'x-axis'
        signal.axes_manager[0].units = 'nm'
        signal.axes_manager[0].scale = self.a
        signal.axes_manager[1].name = 'y-axis'
        signal.axes_manager[1].units = 'nm'
        signal.axes_manager[1].scale = self.a
        signal.axes_manager[2].name = 'z-axis'
        signal.axes_manager[2].units = 'nm'
        signal.axes_manager[2].scale = self.a
        signal.axes_manager[3].name = 'components (x,y,z)'
        signal.axes_manager[3].units = ''
        # Set metadata:
        signal.metadata.Signal.title = 'VectorData'
        # Return signal:
        return signal

    @classmethod
    def from_signal(cls, signal):
        """Convert a :class:`~hyperspy.signals.Signal` object to a :class:`~.FieldData` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Signal`
            The :class:`~hyperspy.signals.Signal` object which should be converted to FieldData.

        Returns
        -------
        mag_data: :class:`~.FieldData`
            A :class:`~.VectorData` object containing the loaded data.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        cls._log.debug('Calling from_signal')
        # Extract field:
        field = np.rollaxis(signal.data, 3, 0)
        # Extract properties:
        a = signal.axes_manager[0].scale
        return cls(a, field)

    def save_to_hdf5(self, filename='vecdata.hdf5'):
        """Save vector field data in a file with HyperSpys HDF5-format.

        Parameters
        ----------
        filename : string, optional
            The name of the HDF5-file in which to store the vector field.
            The default is 'vecdata.hdf5'.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_hdf5')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'vecdata')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        # Save data to file:
        self.to_signal().save(filename)

    @classmethod
    def load_from_hdf5(cls, filename):
        """Construct :class:`~.VectorData` object from HyperSpys HDF5-file.

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
            directory = os.path.join(DIR_FILES, 'vecdata')
            filename = os.path.join(directory, filename)
        # Load data from file:
        return VectorData.from_signal(hs.load(filename))

    def save_to_llg(self, filename='vecdata.txt'):
        """Save vector field data in a file with LLG-format.

        Parameters
        ----------
        filename : string, optional
            The name of the LLG-file in which to store the vector field data.
            The default is 'vecdata.txt'.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_llg')
        SCALE = 1.0E-9 / 1.0E-2  # from nm to cm
        # Create 3D meshgrid and reshape it and the field into a list where x varies first:
        zz, yy, xx = self.a * SCALE * (np.indices(self.dim) + 0.5).reshape(3, -1)
        x_vec, y_vec, z_vec = self.field.reshape(3, -1)
        data = np.array([xx, yy, zz, x_vec, y_vec, z_vec]).T
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'vecdata')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        # Save data to file:
        with open(filename, 'w') as mag_file:
            mag_file.write('LLGFileCreator: %s\n' % filename)
            mag_file.write('    %d    %d    %d\n' % (self.dim[2], self.dim[1], self.dim[0]))
            mag_file.writelines('\n'.join('   '.join('{:7.6e}'.format(cell)
                                                     for cell in row) for row in data))

    @classmethod
    def load_from_llg(cls, filename):
        """Construct :class:`~.VectorData` object from LLG-file.

        Parameters
        ----------
        filename : string
            The name of the LLG-file from which to load the data.

        Returns
        -------
        mag_data: :class:`~.VectorData`
            A :class:`~.VectorData` object containing the loaded data.

        """
        cls._log.debug('Calling load_from_llg')
        SCALE = 1.0E-9 / 1.0E-2  # From cm to nm
        # Use relative path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'vecdata')
            filename = os.path.join(directory, filename)
        # Load data from file:
        data = np.genfromtxt(filename, skip_header=2)
        dim = tuple(np.genfromtxt(filename, dtype=int, skip_header=1, skip_footer=len(data[:, 0])))
        a = (data[1, 0] - data[0, 0]) / SCALE
        field = data[:, 3:6].T.reshape((3,) + dim)
        return cls(a, field)

    def quiver_plot(self, title='Vector Field', axis=None, proj_axis='z',
                    coloring='angle', ar_dens=1, ax_slice=None, log=False, scaled=True,
                    scale=1., show_mask=True):
        """Plot a slice of the vector field as a quiver plot.

        Parameters
        ----------
        title : string, optional
            The title for the plot.
        axis : :class:`~matplotlib.axes.AxesSubplot`, optional
            Axis on which the graph is plotted. Creates a new figure if none is specified.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        coloring : string
            Color coding mode of the arrows. Use 'angle' (default), 'amplitude' or 'uniform'.
        ar_dens: int (optional)
            Number defining the arrow density which is plotted. A higher ar_dens number skips more
            arrows (a number of 2 plots every second arrow). Default is 1.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        log : boolean, optional
            The loratihm of the arrow length is plotted instead. This is helpful if only the
             direction of the arrows is important and the amplitude varies a lot. Default is False.
        scaled : boolean, optional
            Normalizes the plotted arrows in respect to the highest one. Default is True.
        scale: float, optional
            Additional multiplicative factor scaling the arrow length. Default is 1
            (no further scaling).
        show_mask: boolean
            Default is True. Shows the outlines of the mask slice if available.

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        """
        self._log.debug('Calling quiver_plot')
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = self.dim[0] // 2
            u_mag = np.copy(self.field[0][ax_slice, ...])  # x-component
            v_mag = np.copy(self.field[1][ax_slice, ...])  # y-component
            u_label = 'x [px]'
            v_label = 'y [px]'
            submask = self.get_mask()[ax_slice, ...]
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = self.dim[1] // 2
            u_mag = np.copy(self.field[0][:, ax_slice, :])  # x-component
            v_mag = np.copy(self.field[2][:, ax_slice, :])  # z-component
            u_label = 'x [px]'
            v_label = 'z [px]'
            submask = self.get_mask()[:, ax_slice, :]
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            if ax_slice is None:
                self._log.debug('ax_slice is None')
                ax_slice = self.dim[2] // 2
            u_mag = np.swapaxes(np.copy(self.field[2][..., ax_slice]), 0, 1)  # z-component
            v_mag = np.swapaxes(np.copy(self.field[1][..., ax_slice]), 0, 1)  # y-component
            u_label = 'z [px]'
            v_label = 'y [px]'
            submask = self.get_mask()[..., ax_slice]
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        # Prepare quiver (select only used arrows if ar_dens is specified):
        dim_uv = u_mag.shape
        vv, uu = np.indices(dim_uv) + 0.5  # shift to center of pixel
        uu = uu[::ar_dens, ::ar_dens]
        vv = vv[::ar_dens, ::ar_dens]
        u_mag = u_mag[::ar_dens, ::ar_dens]
        v_mag = v_mag[::ar_dens, ::ar_dens]
        amplitudes = np.hypot(u_mag, v_mag)
        angles = np.angle(u_mag + 1j * v_mag, deg=True).tolist()
        # Calculate the arrow colors:
        if coloring == 'angle':
            self._log.debug('Encoding angles')
            colorinds = (1 + np.arctan2(v_mag, u_mag) / np.pi) / 2  # in-plane color index (0-1).
            cmap = DirectionalColormap()
        elif coloring == 'amplitude':
            self._log.debug('Encoding amplitude')
            colorinds = amplitudes / amplitudes.max()
            cmap = 'jet'
        elif coloring == 'uniform':
            self._log.debug('No color encoding')
            colorinds = np.zeros_like(u_mag)  # use black arrows!
            cmap = 'gray'
        else:
            raise AttributeError("Invalid coloring mode! Use 'angles', 'amplitude' or 'uniform'!")
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=(8.5, 7))
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Take the logarithm of the arrows to clearly show directions (if specified):
        if log and np.any(amplitudes):  # If the slice is empty, skip!
            cutoff = 10
            amp = np.round(amplitudes, decimals=cutoff)
            min_value = amp[np.nonzero(amp)].min()
            u_mag = np.round(u_mag, decimals=cutoff) / min_value
            u_mag = np.log10(np.abs(u_mag) + 1) * np.sign(u_mag)
            v_mag = np.round(v_mag, decimals=cutoff) / min_value
            v_mag = np.log10(np.abs(v_mag) + 1) * np.sign(v_mag)
            amplitudes = np.hypot(u_mag, v_mag)  # Recalculate (used if scaled)!
        # Scale the amplitude of the arrows to the highest one (if specified):
        if scaled:
            u_mag /= amplitudes.max() + 1E-30
            v_mag /= amplitudes.max() + 1E-30
        axis.quiver(uu, vv, u_mag, v_mag, colorinds, cmap=cmap, clim=(0, 1), angles=angles,
                    pivot='middle', units='xy', scale_units='xy', scale=scale / ar_dens,
                    minlength=0.25, headwidth=6, headlength=7)
        if show_mask and not np.all(submask):  # Plot mask if desired and not trivial!
            vv, uu = np.indices(dim_uv) + 0.5  # shift to center of pixel
            axis.contour(uu, vv, submask, levels=[0.5], colors='k', linestyles='dotted')
        axis.set_xlim(0, dim_uv[1])
        axis.set_ylim(0, dim_uv[0])
        axis.set_title(title, fontsize=18)
        axis.set_xlabel(u_label, fontsize=15)
        axis.set_ylabel(v_label, fontsize=15)
        axis.tick_params(axis='both', which='major', labelsize=14)
        if dim_uv[0] >= dim_uv[1]:
            u_bin, v_bin = np.max((2, np.floor(9 * dim_uv[1] / dim_uv[0]))), 9
        else:
            u_bin, v_bin = 9, np.max((2, np.floor(9 * dim_uv[0] / dim_uv[1])))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=u_bin, integer=True))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=v_bin, integer=True))
        # Return plotting axis:
        return axis

    def quiver_plot3d(self, title='Vector Field', limit=None, cmap='jet',
                      mode='2darrow', coloring='angle', ar_dens=1, opacity=1.0):
        """Plot the vector field as 3D-vectors in a quiverplot.

        Parameters
        ----------
        title : string, optional
            The title for the plot.
        limit : float, optional
            Plotlimit for the vector field arrow length used to scale the colormap.
        cmap : string, optional
            String describing the colormap which is used (default is 'jet').
        ar_dens: int (optional)
            Number defining the arrow density which is plotted. A higher ar_dens number skips more
            arrows (a number of 2 plots every second arrow). Default is 1.
        mode: string, optional
            Mode, determining the glyphs used in the 3D plot. Default is '2darrow', which
            corresponds to 2D arrows. For smaller amounts of arrows, 'arrow' (3D) is prettier.
        coloring : string
            Color coding mode of the arrows. Use 'angle' (default) or 'amplitude'.
        opacity: float, optional
            Defines the opacity of the arrows. Default is 1.0 (completely opaque).

        Returns
        -------
        plot : :class:`mayavi.modules.vectors.Vectors`
            The plot object.

        """
        self._log.debug('Calling quiver_plot3D')
        from mayavi import mlab
        a = self.a
        dim = self.dim
        if limit is None:
            limit = np.max(self.field_amp)
        ad = ar_dens
        # Create points and vector components as lists:
        zzz, yyy, xxx = (np.indices(dim) - a / 2).reshape((3,) + dim)
        zzz = zzz[::ad, ::ad, ::ad].flatten()
        yyy = yyy[::ad, ::ad, ::ad].flatten()
        xxx = xxx[::ad, ::ad, ::ad].flatten()
        x_mag = self.field[0][::ad, ::ad, ::ad].flatten()
        y_mag = self.field[1][::ad, ::ad, ::ad].flatten()
        z_mag = self.field[2][::ad, ::ad, ::ad].flatten()
        # Plot them as vectors:
        mlab.figure(size=(750, 700))
        plot = mlab.quiver3d(xxx, yyy, zzz, x_mag, y_mag, z_mag,
                             mode=mode, colormap=cmap, opacity=opacity)
        if coloring == 'angle':  # Encodes the full angle via colorwheel and saturation
            self._log.debug('Encoding full 3D angles')
            from tvtk.api import tvtk
            rgb = DirectionalColormap.rgb_from_direction(x_mag, y_mag, z_mag)
            colors = [tuple(c) for c in rgb]  # convert to list of tuples!
            sc = tvtk.UnsignedCharArray()  # Used to hold the colors
            sc.from_array(colors)
            plot.mlab_source.dataset.point_data.scalars = sc
            plot.mlab_source.dataset.modified()
            plot.glyph.color_mode = 'color_by_scalar'
        elif coloring == 'amplitude':  # Encodes the amplitude of the arrows with the jet colormap
            self._log.debug('Encoding amplitude')
            mlab.colorbar(label_fmt='%.2f')
            mlab.colorbar(orientation='vertical')
        else:
            raise AttributeError('Coloring mode not supported!')
        plot.glyph.glyph_source.glyph_position = 'center'
        plot.module_manager.vector_lut_manager.data_range = np.array([0, limit])
        mlab.outline(plot)
        mlab.axes(plot)
        mlab.title(title, height=0.95, size=0.35)
        mlab.orientation_axes()
        return plot


class ScalarData(FieldData):
    """Class for storing scalar field data.

    Represents 3-dimensional scalar field distributions which is stored as a 3-dimensional
    numpy array in `field`, but which can also be accessed as a vector via `field_vec`.
    :class:`~.ScalarData` objects support negation, arithmetic operators (``+``, ``-``, ``*``)
    and their augmented counterparts (``+=``, ``-=``, ``*=``), with numbers and other
    :class:`~.ScalarData` objects, if their dimensions and grid spacings match. It is possible
    to load data from HDF5 or LLG (.txt) files or to save the data in these formats.
    Plotting methods are also provided.

    Attributes
    ----------
    a: float
        The grid spacing in nm.
    field: :class:`~numpy.ndarray` (N=4)
        The scalar field.

    """
    _log = logging.getLogger(__name__ + '.ScalarData')

    def __init__(self, a, field):
        self._log.debug('Calling __init__')
        super(ScalarData, self).__init__(a, field)
        self._log.debug('Created ' + str(self))

    def scale_down(self, n=1):
        """Scale down the field distribution by averaging over two pixels along each axis.

        Parameters
        ----------
        n : int, optional
            Number of times the field distribution is scaled down. The default is 1.

        Returns
        -------
        None

        Notes
        -----
        Acts in place and changes dimensions and grid spacing accordingly.
        Only possible, if each axis length is a power of 2!

        """
        self._log.debug('Calling scale_down')
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        self.a *= 2 ** n
        for t in range(n):
            # Pad if necessary:
            pz, py, px = self.dim[0] % 2, self.dim[1] % 2, self.dim[2] % 2
            if pz != 0 or py != 0 or px != 0:
                self.field = np.pad(self.field, ((0, pz), (0, py), (0, px)), mode='constant')
            # Create coarser grid for the field:
            shape_4d = (self.dim[0] / 2, 2, self.dim[1] / 2, 2, self.dim[2] / 2, 2)
            self.field = self.field.reshape(shape_4d).mean(axis=(5, 3, 1))

    def scale_up(self, n=1, order=0):
        """Scale up the field distribution using spline interpolation of the requested order.

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
        assert n > 0 and isinstance(n, (int, long)), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, (int, long)), \
            'order must be a positive integer between 0 and 5!'
        self.a /= 2 ** n
        self.field = zoom(self.field, zoom=2 ** n, order=order)

    def get_vector(self, mask):
        """Returns the field as a vector, specified by a mask.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean)
            Masks the pixels from which the components should be taken.

        Returns
        -------
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing the field of the specified pixels.

        """
        self._log.debug('Calling get_vector')
        if mask is not None:
            return np.reshape(self.field[mask], -1)
        else:
            return self.field_vec

    def set_vector(self, vector, mask=None):
        """Set the field components of the masked pixels to the values specified by `vector`.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` (N=3, boolean), optional
            Masks the pixels from which the components should be taken.
        vector : :class:`~numpy.ndarray` (N=1)
            The vector containing the field of the specified pixels.

        Returns
        -------
        None

        """
        self._log.debug('Calling set_vector')
        vector = np.asarray(vector, dtype=fft.FLOAT)
        if mask is not None:
            self.field[mask] = vector
        else:
            self.field_vec = vector

    def to_signal(self):
        """Convert :class:`~.ScalarData` data into a HyperSpy signal.

        Returns
        -------
        signal: :class:`~hyperspy.signals.Signal`
            Representation of the :class:`~.ScalarData` object as a HyperSpy Signal.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        self._log.debug('Calling to_signal')
        # Try importing HyperSpy:
        if hs is None:
            self._log.error('This method recquires the hyperspy package!')
            return
        # Create signal:
        signal = hs.signals.Signal(self.field)
        # Set axes:
        signal.axes_manager[0].name = 'x-axis'
        signal.axes_manager[0].units = 'nm'
        signal.axes_manager[0].scale = self.a
        signal.axes_manager[1].name = 'y-axis'
        signal.axes_manager[1].units = 'nm'
        signal.axes_manager[1].scale = self.a
        signal.axes_manager[2].name = 'z-axis'
        signal.axes_manager[2].units = 'nm'
        signal.axes_manager[2].scale = self.a
        # Set metadata:
        signal.metadata.Signal.title = 'ScalarData'
        # Return signal:
        return signal

    @classmethod
    def from_signal(cls, signal):
        """Convert a :class:`~hyperspy.signals.Signal` object to a :class:`~.ScalarData` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Signal`
            The :class:`~hyperspy.signals.Signal` object which should be converted to ScalarData.

        Returns
        -------
        mag_data: :class:`~.ScalarData`
            A :class:`~.ScalarData` object containing the loaded data.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        cls._log.debug('Calling from_signal')
        # Extract field:
        field = signal.data
        # Extract properties:
        a = signal.axes_manager[0].scale
        return cls(a, field)

    def save_to_hdf5(self, filename='scaldata.hdf5'):
        """Save field data in a file with HyperSpys HDF5-format.

        Parameters
        ----------
        filename : string, optional
            The name of the HDF5-file in which to store the field data.
            The default is 'scaldata.hdf5'.

        Returns
        -------
        None

        """
        self._log.debug('Calling save_to_hdf5')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'scaldata')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        # Save data to file:
        self.to_signal().save(filename)

    @classmethod
    def load_from_hdf5(cls, filename):
        """Construct :class:`~.ScalarData` object from HyperSpys HDF5-file.

        Parameters
        ----------
        filename : string
            The name of the HDF5-file from which to load the data. Standard format is '\*.hdf5'.

        Returns
        -------
        mag_data: :class:`~.ScalarData`
            A :class:`~.ScalarData` object containing the loaded data.

        """
        cls._log.debug('Calling load_from_hdf5')
        if hs is None:
            cls._log.error('This method recquires the hyperspy package!')
            return
        # Use relative path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid.config import DIR_FILES
            directory = os.path.join(DIR_FILES, 'scaldata')
            filename = os.path.join(directory, filename)
        # Load data from file:
        return ScalarData.from_signal(hs.load(filename))

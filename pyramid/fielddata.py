# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides classes for storing vector and scalar 3D-field."""

import abc
import logging
from scipy.ndimage.interpolation import rotate
from numbers import Number

import numpy as np
from PIL import Image
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.ndimage.interpolation import zoom

import cmocean

from . import colors
from . import plottools
from .quaternion import Quaternion

__all__ = ['VectorData', 'ScalarData']


class FieldData(object, metaclass=abc.ABCMeta):
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

    _log = logging.getLogger(__name__ + '.FieldData')

    # TODO: Property mask: can be provided at construction and also set later. Default is None!
    # TODO: The getter function should use the current get_mask() function if mask is None (and
    # TODO: save it to _mask or something!

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
    @abc.abstractmethod
    def field(self):
        pass

    @property
    def field_amp(self):
        """The field amplitude (returns the field itself for scalar and the vector amplitude
        calculated via a square sum for a vector field."""
        if len(self.shape) == 4:
            return np.sqrt(np.sum(self.field ** 2, axis=0))
        else:
            return np.abs(self.field)

    @property
    def field_vec(self):
        """Vector containing the vector field distribution."""
        return np.reshape(self.field, -1)

    @field_vec.setter
    def field_vec(self, mag_vec):
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
        return self.__class__(self.a, -self.field)

    def __add__(self, other):  # self + other
        self._log.debug('Calling __add__')
        assert isinstance(other, (FieldData, Number)), \
            'Only FieldData objects and scalar numbers (as offsets) can be added/subtracted!'
        if isinstance(other, Number):  # other is a Number
            self._log.debug('Adding an offset')
            return self.__class__(self.a, self.field + other)
        elif isinstance(other, FieldData):
            self._log.debug('Adding two FieldData objects')
            assert other.a == self.a, 'Added phase has to have the same grid spacing!'
            assert other.shape == self.shape, 'Added field has to have the same dimensions!'
            return self.__class__(self.a, self.field + other.field)

    def __sub__(self, other):  # self - other
        self._log.debug('Calling __sub__')
        return self.__add__(-other)

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        assert isinstance(other, Number), 'FieldData objects can only be multiplied by numbers!'
        return self.__class__(self.a, self.field * other)

    def __truediv__(self, other):  # self / other
        self._log.debug('Calling __truediv__')
        assert isinstance(other, Number), 'FieldData objects can only be divided by numbers!'
        return self.__class__(self.a, self.field / other)

    def __floordiv__(self, other):  # self // other
        self._log.debug('Calling __floordiv__')
        assert isinstance(other, Number), 'FieldData objects can only be divided by numbers!'
        return self.__class__(self.a, self.field // other)

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
        # TODO: When indexing, the components should not need to be included:
        # TODO: e.g.: fielddata[1:7, 3:5, 3:4] instead of fielddata[:, 1:7, 3:5, 3:4]
        return self.__class__(self.a, self.field[item])

    def __array__(self, dtype=None):  # Used for numpy ufuncs, together with __array_wrap__!
        if dtype:
            return self.field.astype(dtype)
        else:
            return self.field

    def __array_wrap__(self, array, _=None):  # _ catches the context, which is not used.
        return type(self)(self.a, array)

    def copy(self):
        """Returns a copy of the :class:`~.FieldData` object

        Returns
        -------
        field_data: :class:`~.FieldData`
            A copy of the :class:`~.FieldData`.

        """
        self._log.debug('Calling copy')
        return self.__class__(self.a, self.field.copy())

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

    def plot_mask(self, title='Mask', threshold=0, grid=True, labels=True,
                  orientation=True, figsize=None, new_fig=True, **kwargs):
        """Plot the mask as a 3D-contour plot.

        Parameters
        ----------
        title: string, optional
            The title for the plot.
        threshold : float, optional
            A pixel only gets masked, if it lies above this threshold . The default is 0.

        Returns
        -------
        plot : :class:`mayavi.modules.vectors.Vectors`
            The plot object.

        """
        self._log.debug('Calling plot_mask')
        from mayavi import mlab
        if figsize is None:
            figsize = (750, 700)
        if new_fig:
            mlab.figure(size=figsize, bgcolor=(0.5, 0.5, 0.5), fgcolor=(0., 0., 0.))
        zzz, yyy, xxx = (np.indices(self.dim) + self.a / 2)
        zzz, yyy, xxx = zzz.T, yyy.T, xxx.T
        mask = self.get_mask(threshold=threshold).astype(int).T  # Transpose because of VTK order!
        extent = np.ravel(list(zip((0, 0, 0), mask.shape)))
        cont = mlab.contour3d(xxx, yyy, zzz, mask, contours=[1], **kwargs)
        if grid:
            mlab.outline(cont, extent=extent)
        if labels:
            mlab.axes(cont, extent=extent)
            mlab.title(title, height=0.95, size=0.35)
        if orientation:
            oa = mlab.orientation_axes()
            oa.marker.set_viewport(0, 0, 0.4, 0.4)
            mlab.draw()
        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.isometric_view()
        return cont

    def plot_contour3d(self, title='Field Distribution', contours=10, opacity=0.25,
                       new_fig=True, **kwargs):  # TODO: new_fig or hold in every mayavi plot!
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
        self._log.debug('Calling plot_contour3d')
        from mayavi import mlab
        if new_fig:
            mlab.figure(size=(750, 700))
        zzz, yyy, xxx = (np.indices(self.dim) + self.a / 2)
        zzz, yyy, xxx = zzz.T, yyy.T, xxx.T
        field_amp = self.field_amp.T  # Transpose because of VTK order!
        if not isinstance(contours, (list, tuple, np.ndarray)):  # Calculate the contours:
            contours = list(np.linspace(field_amp.min(), field_amp.max(), contours))
        extent = np.ravel(list(zip((0, 0, 0), field_amp.shape)))
        cont = mlab.contour3d(xxx, yyy, zzz, field_amp, contours=contours,
                              opacity=opacity, **kwargs)
        mlab.outline(cont, extent=extent)
        mlab.axes(cont, extent=extent)
        mlab.title(title, height=0.95, size=0.35)
        mlab.orientation_axes()
        cont.scene.isometric_view()
        return cont

    @abc.abstractmethod
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
        pass  # TODO: NotImplementedError instead? See that all classes have the same interface!
        # TODO: This means that all common functions of Scalar and Vector have to be abstract here!

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @classmethod
    def from_signal(cls, signal):
        """Convert a :class:`~hyperspy.signals.Signal` object to a :class:`~.FieldData` object.

        Parameters
        ----------
        signal: :class:`~hyperspy.signals.Signal`
            The :class:`~hyperspy.signals.Signal` object which should be converted to FieldData.

        Returns
        -------
        magdata: :class:`~.FieldData`
            A :class:`~.FieldData` object containing the loaded data.

        Notes
        -----
        This method recquires the hyperspy package!

        """
        cls._log.debug('Calling from_signal')
        return cls(signal.axes_manager[0].scale, signal.data)

    @abc.abstractmethod
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
        self._log.debug('Calling to_signal')
        try:  # Try importing HyperSpy:
            # noinspection PyUnresolvedReferences
            import hyperspy.api as hs
        except ImportError:
            self._log.error('This method recquires the hyperspy package!')
            return
        # Create signal:
        signal = hs.signals.BaseSignal(self.field)  # All axes are signal axes!
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
        return signal


class VectorData(FieldData):

    """Class for storing vector ield data.

    Represents 3-dimensional vector field distributions with 3 components which are stored as a
    3-dimensional numpy array in `field`, but which can also be accessed as a vector via
    `field_vec`. :class:`~.VectorData` objects support negation, arithmetic operators
    (``+``, ``-``, ``*``) and their augmented counterparts (``+=``, ``-=``, ``*=``), withnumbers
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

    @property
    def field(self):
        """The field strength for every 3D-gridpoint (scalar: 3D, vector: 4D)."""
        return self._field

    @field.setter
    def field(self, field):
        assert isinstance(field, np.ndarray), 'Field has to be a numpy array!'
        assert len(field.shape) == 4, 'Field has to be 4-dimensional (3D vector-field)!'
        assert field.shape[0] == 3, 'A vector field has to have exactly 3 components!'
        self._field = field

    def __getitem__(self, item):
        return self.__class__(self.a, self.field[item])

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
        assert np.size(vector) % 3 == 0, 'Vector has to contain all 3 components for every pixel!'
        count = np.size(vector) // 3
        if mask is not None:
            self.field[0][mask] = vector[:count]  # x-component
            self.field[1][mask] = vector[count:2 * count]  # y-component
            self.field[2][mask] = vector[2 * count:]  # z-component
        else:
            self.field_vec = vector

    # TODO: scale_down and scale_up should not work in place (also in ScalarData)!
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
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        a_new = self.a * 2 ** n
        field_new = self.field
        for t in range(n):
            dim = field_new.shape[1:]
            # Pad if necessary:
            pz, py, px = dim[0] % 2, dim[1] % 2, dim[2] % 2
            if pz != 0 or py != 0 or px != 0:
                field_new = np.pad(field_new, ((0, 0), (0, pz), (0, py), (0, px)), mode='edge')
                dim = field_new.shape[1:]  # Update dimensions!
            # Create coarser grid for the vector field:
            shape_4d = (3, dim[0] // 2, 2, dim[1] // 2, 2, dim[2] // 2, 2)
            field_new = field_new.reshape(shape_4d).mean(axis=(6, 4, 2))
        return VectorData(a_new, field_new)

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
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, int), \
            'order must be a positive integer between 0 and 5!'
        a_new = self.a / 2 ** n
        field_new = np.array((zoom(self.field[0], zoom=2 ** n, order=order),
                              zoom(self.field[1], zoom=2 ** n, order=order),
                              zoom(self.field[2], zoom=2 ** n, order=order)))
        return VectorData(a_new, field_new)

    def pad(self, pad_values, mode='constant'):
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
        field_pad = np.pad(self.field, ((0, 0), (pv[0], pv[1]), (pv[2], pv[3]), (pv[4], pv[5])),
                           mode=mode)
        return VectorData(self.a, field_pad)

    def crop(self, crop_values):  # TODO: bad copy&paste from pad? zeros cropped -> no sense!
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
        field_crop = self.field[:, cv[0]:cv[1], cv[2]:cv[3], cv[4]:cv[5]]
        return VectorData(self.a, field_crop)

    def flip(self, axis='x'):
        """Flip/mirror the vector field around the specified axis.

        Parameters
        ----------
        axis: {'x', 'y', 'z'}, optional
            The axis around which the vector field is flipped.

        Returns
        -------
        magdata_flip: :class:`~.VectorData`
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

    def rotate(self, angle, axis='z', reshape=False, order=1, **kwargs):
        # TODO: Docstring!
        # Define axes of rotation plane (axis 0 for vector component!) and rotate coord. system:
        axes = {'x': (1, 2), 'y': (1, 3), 'z': (2, 3)}[axis]
        field_coord_rot = rotate(self.field, angle, axes=axes, reshape=reshape,
                                 order=order, **kwargs)
        # Rotate vectors inside the voxels (- signs determined by scipy rotate axes in first step):
        vec_dict = {'x': (-1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, -1)}
        quat = Quaternion.from_axisangle(vec_dict[axis], np.deg2rad(angle))
        field_rot = quat.matrix.dot(field_coord_rot.reshape(3, -1)).reshape(field_coord_rot.shape)
        return VectorData(self.a, field_rot)

    def rot90(self, axis='x'):
        """Rotate the vector field 90° around the specified axis (right hand rotation).

        Parameters
        ----------
        axis: {'x', 'y', 'z'}, optional
            The axis around which the vector field is rotated.

        Returns
        -------
        magdata_rot: :class:`~.VectorData`
           A rotated copy of the :class:`~.VectorData` object.

        """
        self._log.debug('Calling rot90')
        if axis == 'x':  # RotMatrix for 90°: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            field_rot = np.zeros((3, self.dim[1], self.dim[0], self.dim[2]))
            for i in range(self.dim[2]):
                mag_x, mag_y, mag_z = self.field[:, :, :, i]
                mag_xrot = np.rot90(mag_x)
                mag_yrot = np.rot90(mag_z)
                mag_zrot = -np.rot90(mag_y)
                field_rot[:, :, :, i] = np.array((mag_xrot, mag_yrot, mag_zrot))
        elif axis == 'y':  # RotMatrix for 90°: [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
            field_rot = np.zeros((3, self.dim[2], self.dim[1], self.dim[0]))
            for i in range(self.dim[1]):
                mag_x, mag_y, mag_z = self.field[:, :, i, :]
                mag_xrot = np.rot90(mag_z)
                mag_yrot = np.rot90(mag_y)
                mag_zrot = -np.rot90(mag_x)
                field_rot[:, :, i, :] = np.array((mag_xrot, mag_yrot, mag_zrot))
        elif axis == 'z':  # RotMatrix for 90°: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            field_rot = np.zeros((3, self.dim[0], self.dim[2], self.dim[1]))
            for i in range(self.dim[0]):
                mag_x, mag_y, mag_z = self.field[:, i, :, :]
                mag_xrot = np.rot90(mag_y)
                mag_yrot = -np.rot90(mag_x)
                mag_zrot = np.rot90(mag_z)
                field_rot[:, i, :, :] = np.array((mag_xrot, mag_yrot, mag_zrot))
        else:
            raise ValueError("Wrong input! 'x', 'y', 'z' allowed!")
        return VectorData(self.a, field_rot)

    def roll(self, shift, axis='x'):  # TODO: Make sure both classes have all manipulator methods!
        """Rotate the scalar field 90° around the specified axis (right hand rotation).

                Parameters
                ----------
                axis: {'x', 'y', 'z'}, optional
                    The axis around which the vector field is rotated.

                Returns
                -------
                scalardata_rot: :class:`~.ScalarData`
                   A rotated copy of the :class:`~.ScalarData` object.

                """
        self._log.debug('Calling roll')
        ax = {'x': 2, 'y': 1, 'z': 0}[axis]
        mag_x_roll = np.roll(self.field[0, ...], shift, ax)
        mag_y_roll = np.roll(self.field[1, ...], shift, ax)
        mag_z_roll = np.roll(self.field[2, ...], shift, ax)
        return VectorData(self.a, np.asarray((mag_x_roll, mag_y_roll, mag_z_roll)))

    def clip_amp(self, vmin=None, vmax=None):
        # TODO: Docstring! vmin clips to zero, vmax clips to vmax!
        # TODO: 'mag' should be 'vec' everywhere!!!
        mag_x, mag_y, mag_z = self.field
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = self.field_amp.max()
        # scaling = np.zeros_like(self.field_amp)
        scaling = (vmin <= self.field_amp) & (self.field_amp <= vmax)
        mag_x = mag_x * scaling
        mag_y = mag_y * scaling
        mag_z = mag_z * scaling
        return VectorData(self.a, np.asarray((mag_x, mag_y, mag_z)))

    def get_slice(self, ax_slice=None, proj_axis='z'):
        # TODO: return x y z instead of u v w (to color fields consistent with xyz!)
        """Extract a slice from the :class:`~.VectorData` object.

        Parameters
        ----------
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which the slice is taken. The default is 'z'.
        ax_slice : None or int, optional
            The slice-index of the axis specified in `proj_axis`. Defaults to the center slice.

        Returns
        -------
        u_mag, v_mag, w_mag, submask : :class:`~numpy.ndarray` (N=2)
            The extracted vector field components in plane perpendicular to the `proj_axis` and
            the perpendicular component.

        """
        self._log.debug('Calling get_slice')
        # Find slice:
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            u_mag = np.copy(self.field[0][ax_slice, ...])  # x-component
            v_mag = np.copy(self.field[1][ax_slice, ...])  # y-component
            w_mag = np.copy(self.field[2][ax_slice, ...])  # z-component
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            u_mag = np.copy(self.field[0][:, ax_slice, :])  # x-component
            v_mag = np.copy(self.field[2][:, ax_slice, :])  # z-component
            w_mag = np.copy(self.field[1][:, ax_slice, :])  # y-component
        elif proj_axis == 'x':  # Slice of the zy-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            # TODO: Strange swapaxes, really necessary? Get rid EVERYWHERE if possible!
            # u_mag = np.swapaxes(np.copy(self.field[2][..., ax_slice]), 0, 1)  # z-component
            # v_mag = np.swapaxes(np.copy(self.field[1][..., ax_slice]), 0, 1)  # y-component
            # w_mag = np.swapaxes(np.copy(self.field[0][..., ax_slice]), 0, 1)  # x-component
            # TODO: z should be special and always along y in 2D if possible!!
            u_mag = np.copy(self.field[1][..., ax_slice])  # y-component
            v_mag = np.copy(self.field[2][..., ax_slice])  # z-component
            w_mag = np.copy(self.field[0][..., ax_slice])  # x-component
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        return u_mag, v_mag, w_mag

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
        signal = super().to_signal()
        # Set component axis:
        signal.axes_manager[3].name = 'x/y/z-component'
        signal.axes_manager[3].units = ''
        # Set metadata:
        signal.metadata.Signal.title = 'VectorData'
        # Return signal:
        return signal

    def save(self, filename, **kwargs):
        """Saves the VectorData in the specified format.

        The function gets the format from the extension:
            - hdf5 for HDF5.
            - EMD Electron Microscopy Dataset format (also HDF5).
            - llg format.
            - ovf format.
            - npy or npz for numpy formats.

        If no extension is provided, 'hdf5' is used. Most formats are
        saved with the HyperSpy package (internally the fielddata is first
        converted to a HyperSpy Signal.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str, optional
            Name of the file which the VectorData is saved into. The extension
            determines the saving procedure.

        """
        from .file_io.io_vectordata import save_vectordata
        save_vectordata(self, filename, **kwargs)

    # TODO: we have 2D and 3D plots, what about 1D line of vector arrows? Would be nice!
    # TODO: Make all a bit more flexible with squeeze (so dimensions are not as strict).
    # TODO: Implement clip value for amplitude!
    # TODO: ar_dens should work differently! do not show every n-th arrow! instead average:
    # mag_overlay_vort = mag_data_vort.copy()
    # mag_overlay_vort.scale_down(2)
    # mag_overlay_vort.plot_quiver(axis=newax, coloring='uniform', bgcolor='white', show_mask=False,
    #                              scalebar=False, hideaxes=True)
    # # plt.savefig(directory + '/ch5-0-magnetic_distributions_v.png', bbox_inches='tight')
    # plt.savefig(directory + '/ch5-0-magnetic_distributions_v.pdf', bbox_inches='tight')

    def plot_quiver(self, ar_dens='auto', log=False, scaled=True, scale=1., b_0=None, qkey_unit='T',
                    coloring='angle', cmap=None,  # Used here and plot_streamlines!
                    proj_axis='z', ax_slice=None, show_mask=True, bgcolor=None, axis=None,
                    figsize=None, stroke=None, fontsize=None, **kwargs):
        """Plot a slice of the vector field as a quiver plot.

        Parameters
        ----------
        ar_dens: int, optional
            Number defining the arrow density which is plotted. A higher ar_dens number skips more
            arrows (a number of 2 plots every second arrow). Default is 1.
        log : boolean, optional
            The loratihm of the arrow length is plotted instead. This is helpful if only the
             direction of the arrows is important and the amplitude varies a lot. Default is False.
        scaled : boolean, optional
            Normalizes the plotted arrows in respect to the highest one. Default is True.
        scale: float, optional
            Additional multiplicative factor scaling the arrow length. Default is 1
            (no further scaling).
        b_0 : float, optional
            Saturation induction (saturation magnetisation times the vacuum permeability).
            If this is specified, a quiverkey is used to indicate the length of the longest arrow.
        coloring : {'angle', 'amplitude', 'uniform', matplotlib color}
            Color coding mode of the arrows. Use 'full' (default), 'angle', 'amplitude', 'uniform'
            (black or white, depending on `bgcolor`), or a matplotlib color keyword.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            If not set, an appropriate one is used. Note that a subclass of
            :class:`~.colors.Colormap3D` should be used for angle encoding.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        show_mask: boolean
            Default is True. Shows the outlines of the mask slice if available.
        bgcolor: {'white', 'black'}, optional
            Determines the background color of the plot.
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
        self._log.debug('Calling plot_quiver')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        if fontsize is None:
            fontsize = plottools.FONTSIZE_DEFAULT
        if stroke is None:
            stroke = plottools.STROKE_DEFAULT
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            "Axis has to be 'x', 'y' or 'z'."
        # TODO: Deprecate ar_dens in favor of scale_it, call it "binsize" or something...
        # TODO: None or 'auto' as default for ar_dens/bin_size, set something sensible!!!
        if ar_dens == 'auto':
            ar_dens = np.max((1, np.max(self.dim) // 16))
            # TODO: Just for now, delete later, when power of 2 no longer needed:
            ar_dens = int(2**(np.log2(ar_dens)//1))
        if ar_dens > 1:
            scale_it = np.log2(ar_dens)
            assert scale_it % 1 == 0, 'ar_dens has to be power of 2 (for now)!'  # TODO: Delete!
            vecdata = self.scale_down(int(scale_it))
        else:
            vecdata = self
        # Extract slice and mask:
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        ax_slice //= ar_dens
        # TODO: Currently, the slice is taken AFTER the scale down. In EMPyRe, this plotting
        # TODO: function should be called on the slice (another 2D Vector Field class) instead!
        # TODO: The slicing is therefore done before the scale down and more accurate!(??)
        u_mag, v_mag = vecdata.get_slice(ax_slice, proj_axis)[:2]
        submask = np.where(np.hypot(u_mag, v_mag) > 0, True, False)
        # Prepare quiver (select only used arrows if ar_dens is specified):
        dim_uv = u_mag.shape
        vv, uu = (np.indices(dim_uv) + 0.5) * ar_dens  # 0.5: shift to center of pixel!
        amplitudes = np.hypot(u_mag, v_mag)
        # TODO: Delete if only used in log:
        # angles = np.angle(u_mag + 1j * v_mag, deg=True).tolist()
        # Calculate the arrow colors:
        if bgcolor is None:
            bgcolor = 'white'  # Default!
        cmap_overwrite = cmap
        if coloring == 'angle':
            self._log.debug('Encoding angles')
            hue = np.asarray(np.arctan2(v_mag, u_mag) / (2 * np.pi))
            hue[hue < 0] += 1
            cmap = colors.CMAP_CIRCULAR_DEFAULT
        elif coloring == 'amplitude':
            self._log.debug('Encoding amplitude')
            hue = amplitudes / amplitudes.max()
            if bgcolor == 'white':
                cmap = colors.cmaps['cubehelix_reverse']
            else:
                cmap = colors.cmaps['cubehelix_standard']
        elif coloring == 'uniform':
            self._log.debug('Automatic uniform color encoding')
            hue = amplitudes / amplitudes.max()
            cmap = colors.cmaps['transparent_white']
        else:
            self._log.debug('Specified uniform color encoding')
            hue = np.zeros_like(u_mag)
            cmap = ListedColormap([coloring])
        edgecolors = colors.cmaps['transparent_black'](amplitudes / amplitudes.max()).reshape(-1, 4)
        if cmap_overwrite is not None:
            cmap = cmap_overwrite
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Take the logarithm of the arrows to clearly show directions (if specified):
        # TODO: get rid of log!! (only problems...)
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
        # Plot quiver:
        # TODO: quiver does not work with matplotlib 2.0! FIX! DELETE ANGLES PARAMETER V
        quiv = axis.quiver(uu, vv, u_mag, v_mag, hue, cmap=cmap, clim=(0, 1),  # angles=angles,
                           pivot='middle', units='xy', scale_units='xy', scale=scale / ar_dens,
                           minlength=0.05, width=1*ar_dens, headlength=2, headaxislength=2,
                           headwidth=2, minshaft=2, edgecolor=edgecolors, linewidths=1)
        axis.set_xlim(0, dim_uv[1]*ar_dens)
        axis.set_ylim(0, dim_uv[0]*ar_dens)
        # Determine colormap if necessary:
        if coloring == 'amplitude':
            cbar_mappable, cbar_label = quiv, 'amplitude'
        else:
            cbar_mappable, cbar_label = None, None
        # Change background color:
        axis.set_facecolor(bgcolor)
        # Show mask:
        if show_mask and not np.all(submask):  # Plot mask if desired and not trivial!
            vv, uu = np.indices(dim_uv) + 0.5  # shift to center of pixel
            mask_color = 'white' if bgcolor == 'black' else 'black'
            axis.contour(uu, vv, submask, levels=[0.5], colors=mask_color,
                         linestyles='dotted', linewidths=2)
        # Plot quiverkey if B_0 is specified:
        if b_0 and not log:  # The angles needed for log would break the quiverkey!
            label = '{:.3g} {}'.format(amplitudes.max() * b_0, qkey_unit)
            quiv.angles = 'uv'  # With a list of angles, the quiverkey would break!
            txtcolor = 'w' if stroke == 'k' else 'k'
            edgecolor = stroke if stroke is not None else 'none'
            qk = plt.quiverkey(Q=quiv, X=0.88, Y=0.065, U=1, label=label, labelpos='W',
                               coordinates='axes', facecolor='w', edgecolor='k',
                               labelcolor=txtcolor, linewidth=0.5,
                               clip_box=axis.bbox, clip_on=True,
                               fontproperties={'size': kwargs.get('fontsize', fontsize)})
            if stroke is not None:
                qk.text.set_path_effects(
                    [patheffects.withStroke(linewidth=2, foreground=stroke)])
        # Return formatted axis:
        return plottools.format_axis(axis, sampling=a, cbar_mappable=cbar_mappable,
                                     cbar_label=cbar_label, **kwargs)

    def plot_field(self, proj_axis='z', ax_slice=None, show_mask=True, bgcolor=None, axis=None,
                   figsize=None, interpolation='none', **kwargs):
        """Plot a slice of the vector field as a color field imshow plot.

        Parameters
        ----------
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        show_mask: boolean
            Default is True. Shows the outlines of the mask slice if available.
        bgcolor: {'white', 'black'}, optional
            Determines the background color of the plot.
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
        self._log.debug('Calling plot_field')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            "Axis has to be 'x', 'y' or 'z'."
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        # Extract slice and mask:
        u_mag, v_mag, w_mag = self.get_slice(ax_slice, proj_axis)
        submask = np.where(np.hypot(u_mag, v_mag) > 0, True, False)
        # TODO: Before you fix this, fix get_slice!!! (should be easy...)
        # TODO: return x y z instead of u v w (to color fields consistent with xyz!)
        # TODO: maybe best to get colors of slice separately from u and v components!!!
        # TODO: vector_to_rgb does already exist!!
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            u_mag, v_mag, w_mag = u_mag, v_mag, w_mag
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            u_mag, v_mag, w_mag = u_mag, w_mag, v_mag
        elif proj_axis == 'x':  # Slice of the zy-plane with x = ax_slice
            u_mag, v_mag, w_mag = w_mag, u_mag, v_mag
        # TODO: END!
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)

        axis.set_aspect('equal')
        # Determine 'z'-component for luminance (keep as gray if None):
        z_mag = w_mag
        if bgcolor in ('white', 'w'):
            z_mag = np.where(submask, z_mag, np.max(np.hypot(u_mag, v_mag)))
        if bgcolor in ('black', 'k'):
            z_mag = np.where(submask, z_mag, -np.max(np.hypot(u_mag, v_mag)))
        # Plot the field:
        dim_uv = u_mag.shape
        rgb = colors.CMAP_CIRCULAR_DEFAULT.rgb_from_vector(np.asarray((u_mag, v_mag, z_mag)))
        axis.imshow(Image.fromarray(rgb), origin='lower', interpolation=interpolation,
                    extent=(0, dim_uv[1], 0, dim_uv[0]))
        # Change background color:
        if bgcolor is not None:
            axis.set_facecolor(bgcolor)
        # Show mask:
        if show_mask and not np.all(submask):  # Plot mask if desired and not trivial!
            vv, uu = np.indices(dim_uv) + 0.5  # shift to center of pixel
            mask_color = 'w' if bgcolor in ('black', 'k') else 'k'
            axis.contour(uu, vv, submask, levels=[0.5], colors=mask_color,
                         linestyles='dotted', linewidths=2)
        # Return formatted axis:
        return plottools.format_axis(axis, sampling=a, **kwargs)

    def plot_quiver_field(self, **kwargs):
        """Plot the vector field as a field plot with uniformly colored arrows overlayed.

        Parameters
        ----------
        See :func:`~.plot_quiver` and :func:`~.plot_quiver` for parameters!

        Returns
        -------
        axis: :class:`~matplotlib.axes.AxesSubplot`
            The axis on which the graph is plotted.

        """
        # with plt.style.context('../pyramid.mplstyle'):  # TODO: RIGHT USE LIKE THIS???
        # Extract parameters:
        show_mask = kwargs.pop('show_mask', True)  # Only needed once!
        axis = kwargs.pop('axis', None)
        # Plot field first (with mask and axis formatting), then quiver:
        axis = self.plot_field(axis=axis, show_mask=show_mask, **kwargs)
        self.plot_quiver(coloring='uniform', show_mask=False, axis=axis,
                         format_axis=False, **kwargs)
        # Return plotting axis:
        return axis

    def plot_streamline(self, density=2, linewidth=2, coloring='angle', cmap=None,
                        proj_axis='z', ax_slice=None, show_mask=True, bgcolor=None, axis=None,
                        figsize=None, **kwargs):
        """Plot a slice of the vector field as a quiver plot.

        Parameters
        ----------
        density : float or 2-tuple, optional
            Controls the closeness of streamlines. When density = 1, the domain is divided into a
            30x30 grid—density linearly scales this grid. Each cebll in the grid can have, at most,
            one traversing streamline. For different densities in each direction, use
            [density_x, density_y].
        linewidth : numeric or 2d array, optional
            Vary linewidth when given a 2d array with the same shape as velocities.
        coloring : {'angle', 'amplitude', 'uniform'}
            Color coding mode of the arrows. Use 'full' (default), 'angle', 'amplitude' or
            'uniform'.
        cmap : string, optional
            The :class:`~matplotlib.colors.Colormap` which is used for the plot as a string.
            If not set, an appropriate one is used. Note that a subclass of
            :class:`~.colors.Colormap3D` should be used for angle encoding.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        show_mask: boolean
            Default is True. Shows the outlines of the mask slice if available.
        bgcolor: {'white', 'black'}, optional
            Determines the background color of the plot.
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
        self._log.debug('Calling plot_quiver')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        u_mag, v_mag = self.get_slice(ax_slice, proj_axis)[:2]
        submask = np.where(np.hypot(u_mag, v_mag) > 0, True, False)
        # Prepare streamlines:
        dim_uv = u_mag.shape
        uu = np.arange(dim_uv[1]) + 0.5  # shift to center of pixel
        vv = np.arange(dim_uv[0]) + 0.5  # shift to center of pixel
        u_mag, v_mag = self.get_slice(ax_slice, proj_axis)[:2]
        # v_mag = np.ma.array(v_mag, mask=submask)
        amplitudes = np.hypot(u_mag, v_mag)
        # Calculate the arrow colors:
        if bgcolor is None:
            bgcolor = 'white'  # Default!
        cmap_overwrite = cmap
        if coloring == 'angle':
            self._log.debug('Encoding angles')
            hue = np.asarray(np.arctan2(v_mag, u_mag) / (2 * np.pi))
            hue[hue < 0] += 1
            cmap = colors.CMAP_CIRCULAR_DEFAULT
        elif coloring == 'amplitude':
            self._log.debug('Encoding amplitude')
            hue = amplitudes / amplitudes.max()
            if bgcolor == 'white':
                cmap = colors.cmaps['cubehelix_reverse']
            else:
                cmap = colors.cmaps['cubehelix_standard']
        elif coloring == 'uniform':
            self._log.debug('Automatic uniform color encoding')
            hue = amplitudes / amplitudes.max()
            if bgcolor == 'white':
                cmap = colors.cmaps['transparent_black']
            else:
                cmap = colors.cmaps['transparent_white']
        else:
            self._log.debug('Specified uniform color encoding')
            hue = np.zeros_like(u_mag)
            cmap = ListedColormap([coloring])
        if cmap_overwrite is not None:
            cmap = cmap_overwrite
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Plot the streamlines:
        im = plt.streamplot(uu, vv, u_mag, v_mag, density=density, linewidth=linewidth,
                            color=hue, cmap=cmap)
        # Determine colormap if necessary:
        if coloring == 'amplitude':
            cbar_mappable, cbar_label = im, 'amplitude'
        else:
            cbar_mappable, cbar_label = None, None
        # Change background color:
        axis.set_axis_bgcolor(bgcolor)
        # Show mask:
        if show_mask and not np.all(submask):  # Plot mask if desired and not trivial!
            vv, uu = np.indices(dim_uv) + 0.5  # shift to center of pixel
            mask_color = 'white' if bgcolor == 'black' else 'black'
            axis.contour(uu, vv, submask, levels=[0.5], colors=mask_color,
                         linestyles='dotted', linewidths=2)
        # Return formatted axis:
        return plottools.format_axis(axis, sampling=a, cbar_mappable=cbar_mappable,
                                     cbar_label=cbar_label, **kwargs)

    def plot_quiver3d(self, title='Vector Field', limit=None, cmap='jet', mode='2darrow',
                      coloring='angle', ar_dens=1, opacity=1.0, grid=True, labels=True,
                      orientation=True, size=(700, 750), new_fig=True, view='isometric',
                      position=None, bgcolor=(0.5, 0.5, 0.5)):
        """Plot the vector field as 3D-vectors in a quiverplot.

        Parameters
        ----------
        title : string, optional
            The title for the plot.
        limit : float, optional
            Plotlimit for the vector field arrow length used to scale the colormap.
        cmap : string, optional
            String describing the colormap which is used for amplitude encoding (default is 'jet').
        ar_dens: int, optional
            Number defining the arrow density which is plotted. A higher ar_dens number skips more
            arrows (a number of 2 plots every second arrow). Default is 1.
        mode: string, optional
            Mode, determining the glyphs used in the 3D plot. Default is '2darrow', which
            corresponds to 2D arrows. For smaller amounts of arrows, 'arrow' (3D) is prettier.
        coloring : {'angle', 'amplitude'}, optional
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
        if limit is None:
            limit = np.max(np.nan_to_num(self.field_amp))
        ad = ar_dens
        # Create points and vector components as lists:
        zzz, yyy, xxx = (np.indices(self.dim) + 1 / 2)
        zzz = zzz[::ad, ::ad, ::ad].ravel()
        yyy = yyy[::ad, ::ad, ::ad].ravel()
        xxx = xxx[::ad, ::ad, ::ad].ravel()
        x_mag = self.field[0][::ad, ::ad, ::ad].ravel()
        y_mag = self.field[1][::ad, ::ad, ::ad].ravel()
        z_mag = self.field[2][::ad, ::ad, ::ad].ravel()
        # Plot them as vectors:
        if new_fig:
            mlab.figure(size=size, bgcolor=bgcolor, fgcolor=(0., 0., 0.))
        extent = np.ravel(list(zip((0, 0, 0), (self.dim[2], self.dim[1], self.dim[0]))))
        if coloring == 'angle':  # Encodes the full angle via colorwheel and saturation:
            self._log.debug('Encoding full 3D angles')
            vecs = mlab.quiver3d(xxx, yyy, zzz, x_mag, y_mag, z_mag, mode=mode, opacity=opacity,
                                 scalars=np.arange(len(xxx)), line_width=2)
            vector = np.asarray((x_mag.ravel(), y_mag.ravel(), z_mag.ravel()))
            rgb = colors.CMAP_CIRCULAR_DEFAULT.rgb_from_vector(vector)
            rgba = np.hstack((rgb, 255 * np.ones((len(xxx), 1), dtype=np.uint8)))
            vecs.glyph.color_mode = 'color_by_scalar'
            vecs.module_manager.scalar_lut_manager.lut.table = rgba
            mlab.draw()
        elif coloring == 'amplitude':  # Encodes the amplitude of the arrows with the jet colormap:
            self._log.debug('Encoding amplitude')
            vecs = mlab.quiver3d(xxx, yyy, zzz, x_mag, y_mag, z_mag,
                                 mode=mode, colormap=cmap, opacity=opacity, line_width=2)
            mlab.colorbar(label_fmt='%.2f')
            mlab.colorbar(orientation='vertical')
        else:
            raise AttributeError('Coloring mode not supported!')
        vecs.glyph.glyph_source.glyph_position = 'center'
        vecs.module_manager.vector_lut_manager.data_range = np.array([0, limit])
        if grid:
            mlab.outline(vecs, extent=extent)
        if labels:
            mlab.axes(vecs, extent=extent)
            mlab.title(title, height=0.95, size=0.35)
        if orientation:
            oa = mlab.orientation_axes()
            oa.marker.set_viewport(0, 0, 0.4, 0.4)
            mlab.draw()
        engine = mlab.get_engine()
        scene = engine.scenes[0]
        if view == 'isometric':
            scene.scene.isometric_view()
        elif view == 'x_plus_view':
            scene.scene.x_plus_view()
        elif view == 'y_plus_view':
            scene.scene.y_plus_view()
        if position:
            scene.scene.camera.position = position
        return vecs


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

    @property
    def field(self):
        """The field strength for every 3D-gridpoint (scalar: 3D)."""
        return self._field

    @field.setter
    def field(self, field):
        assert isinstance(field, np.ndarray), 'Field has to be a numpy array!'
        assert len(field.shape) == 3, 'Field has to be 3-dimensional (3D scalar field)!'
        self._field = field

    @property
    def field_vec(self):
        """Vector containing the scalar field distribution."""
        return np.reshape(self.field, -1)

    @field_vec.setter
    def field_vec(self, c_vec):
        assert np.size(c_vec) == np.prod(self.shape), \
            'Vector has to match field shape! {} {}'.format(c_vec.shape, np.prod(self.shape))
        self.field = c_vec.reshape(self.dim)

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
        if mask is not None:
            self.field[mask] = vector
        else:
            self.field_vec = vector

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
        # TODO: Florian: make more general, not just power of 2, n should denote bin size, not
        # TODO: number of scale down operations! Same for scale up!
        # TODO: def rebin(a, shape):
        # TODO:     sh = (shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1])
        # TODO:     return a.reshape(sh).mean(-1).mean(1)
        self._log.debug('Calling scale_down')
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        a_new = self.a * 2 ** n
        field_new = self.field
        for t in range(n):
            # Pad if necessary:
            dim = field_new.shape
            pz, py, px = dim[0] % 2, dim[1] % 2, dim[2] % 2
            if pz != 0 or py != 0 or px != 0:
                field_new = np.pad(field_new, ((0, pz), (0, py), (0, px)), mode='constant')
            # Create coarser grid for the field:
            shape_3d = (dim[0] // 2, 2, dim[1] // 2, 2, dim[2] // 2, 2)
            field_new = field_new.reshape(shape_3d).mean(axis=(5, 3, 1))
        return ScalarData(a_new, field_new)

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
        assert n > 0 and isinstance(n, int), 'n must be a positive integer!'
        assert 5 > order >= 0 and isinstance(order, int), \
            'order must be a positive integer between 0 and 5!'
        a_new = self.a / 2 ** n
        field_new = zoom(self.field, zoom=2 ** n, order=order)
        return ScalarData(a_new, field_new)

    # TODO: flip!

    def crop(self, crop_values):  # TODO: it doesn't work now.
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
        field_crop = self.field[cv[0]:cv[1], cv[2]:cv[3], cv[4]:cv[5]]
        return ScalarData(self.a, field_crop)

    def rotate(self, angle, axis='z', reshape=False, **kwargs):
        # TODO: Docstring!
        axes = {'x': (0, 1), 'y': (0, 2), 'z': (1, 2)}[axis]  # Defines axes of plane of rotation!
        field_rot = rotate(self.field[...], angle, axes=axes, reshape=reshape, **kwargs)
        return ScalarData(self.a, field_rot)

    def rot90(self, axis='x'):
        """Rotate the scalar field 90° around the specified axis (right hand rotation).

        Parameters
        ----------
        axis: {'x', 'y', 'z'}, optional
            The axis around which the vector field is rotated.

        Returns
        -------
        scalardata_rot: :class:`~.ScalarData`
           A rotated copy of the :class:`~.ScalarData` object.

        """
        self._log.debug('Calling rot90')
        if axis == 'x':
            field_rot = np.zeros((self.dim[1], self.dim[0], self.dim[2]))
            for i in range(self.dim[2]):
                field_rot[:, :, i] = np.rot90(self.field[:, :, i])
        elif axis == 'y':
            field_rot = np.zeros((self.dim[2], self.dim[1], self.dim[0]))
            for i in range(self.dim[1]):
                field_rot[:, i, :] = np.rot90(self.field[:, i, :])
        elif axis == 'z':
            field_rot = np.zeros((self.dim[0], self.dim[2], self.dim[1]))
            for i in range(self.dim[0]):
                field_rot[i, :, :] = np.rot90(self.field[i, :, :])
        else:
            raise ValueError("Wrong input! 'x', 'y', 'z' allowed!")
        return ScalarData(self.a, field_rot)

    def roll(self, shift, axis='x'):  # TODO: Make sure both classes have all manipulator methods!
        """Rotate the scalar field 90° around the specified axis (right hand rotation).

                Parameters
                ----------
                axis: {'x', 'y', 'z'}, optional
                    The axis around which the vector field is rotated.

                Returns
                -------
                scalardata_rot: :class:`~.ScalarData`
                   A rotated copy of the :class:`~.ScalarData` object.

                """
        self._log.debug('Calling roll')
        ax = {'x': 2, 'y': 1, 'z': 0}[axis]
        return ScalarData(self.a, np.roll(self.field, shift, ax))

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
        signal = super().to_signal()
        # Set metadata:
        signal.metadata.Signal.title = 'ScalarData'
        # Return signal:
        return signal

    def save(self, filename, **kwargs):
        """Saves the ScalarData in the specified format.

        The function gets the format from the extension:
            - hdf5 for HDF5.
            - EMD Electron Microscopy Dataset format (also HDF5).
            - npy or npz for numpy formats.

        If no extension is provided, 'hdf5' is used. Most formats are
        saved with the HyperSpy package (internally the fielddata is first
        converted to a HyperSpy Signal.

        Each format accepts a different set of parameters. For details
        see the specific format documentation.

        Parameters
        ----------
        filename : str, optional
            Name of the file which the ScalarData is saved into. The extension
            determines the saving procedure.

        """
        from .file_io.io_scalardata import save_scalardata
        save_scalardata(self, filename, **kwargs)

    def get_slice(self, ax_slice=None, proj_axis='z'):
        # TODO: Docstring!
        """Extract a slice from the :class:`~.VectorData` object.

        Parameters
        ----------
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which the slice is taken. The default is 'z'.
        ax_slice : None or int, optional
            The slice-index of the axis specified in `proj_axis`. Defaults to the center slice.

        Returns
        -------
        u_mag, v_mag, w_mag, submask : :class:`~numpy.ndarray` (N=2)
            The extracted vector field components in plane perpendicular to the `proj_axis` and
            the perpendicular component.

        """
        self._log.debug('Calling get_slice')
        # Find slice:
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            scalar_slice = np.copy(self.field[ax_slice, ...])
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            scalar_slice = np.copy(self.field[:, ax_slice, :])
        elif proj_axis == 'x':  # Slice of the zy-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            scalar_slice = np.copy(self.field[..., ax_slice])
        else:
            raise ValueError('{} is not a valid argument (use x, y or z)'.format(proj_axis))
        return scalar_slice

    def plot_field(self, proj_axis='z', ax_slice=None, show_mask=True, bgcolor=None, axis=None,
                   figsize=None, vmin=None, vmax=None, symmetric=False, cmap=None, cbar=True,
                   **kwargs):
        # TODO: Docstring!
        """Plot a slice of the scalar field as an imshow plot.

        Parameters
        ----------
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which a slice is plotted. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Is set to the center of
            `proj_axis` if not specified.
        show_mask: boolean
            Default is True. Shows the outlines of the mask slice if available.
        bgcolor: {'white', 'black'}, optional
            Determines the background color of the plot.
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
        self._log.debug('Calling plot_field')
        a = self.a
        if figsize is None:
            figsize = plottools.FIGSIZE_DEFAULT
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            "Axis has to be 'x', 'y' or 'z'."
        if ax_slice is None:
            ax_slice = self.dim[{'z': 0, 'y': 1, 'x': 2}[proj_axis]] // 2
        # Extract slice and mask:
        field_slice = self.get_slice(ax_slice, proj_axis)
        submask = np.where(np.abs(field_slice) > 0, True, False)
        dim_uv = field_slice.shape
        # If no axis is specified, a new figure is created:
        if axis is None:
            self._log.debug('axis is None')
            fig = plt.figure(figsize=figsize)
            axis = fig.add_subplot(1, 1, 1)
        axis.set_aspect('equal')
        # Configure colormap and fix center to zero if colormap is symmetric:
        if cmap is None:
            cmap = cmocean.cm.thermal
        elif isinstance(cmap, str):  # Get colormap if given as string:
            cmap = plt.get_cmap(cmap)
        if vmin is None:
            vmin = np.min(field_slice)
        if vmax is None:
            vmax = np.max(field_slice)
        if symmetric:  # TODO: symmetric should be called divergent (see cmocean)!
            vmin, vmax = np.min([vmin, -0]), np.max([0, vmax])  # Ensure zero is present!
            limit = np.max(np.abs([vmin, vmax]))
            start = (vmin + limit) / (2 * limit)
            end = (vmax + limit) / (2 * limit)
            cmap_colors = cmap(np.linspace(start, end, 256))
            cmap = LinearSegmentedColormap.from_list('Symmetric', cmap_colors)
        # Plot the field:
        im = axis.imshow(field_slice, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                         interpolation='none', extent=(0, dim_uv[1], 0, dim_uv[0]))
        if show_mask and not np.all(submask):  # Plot mask if desired and not trivial!
            vv, uu = np.indices(dim_uv) + 0.5
            axis.contour(uu, vv, submask, levels=[0.5], colors='k',
                         linestyles='dotted', linewidths=2)
        if bgcolor is not None:
            axis.set_facecolor(bgcolor)
        # Determine colorbar title:
        cbar_mappable = None
        if cbar:
            cbar_mappable = im
        # Return formatted axis:
        return plottools.format_axis(axis, sampling=a, cbar_mappable=cbar_mappable, **kwargs)

# TODO: Histogram plots for magnetisation (see thesis!)

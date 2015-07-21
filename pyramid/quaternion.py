# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Quaternion` class which can be used for rotations."""


import numpy as np

import logging


__all__ = ['Quaternion']


class Quaternion(object):

    '''Class representing a rotation expressed by a quaternion.

    A quaternion is a four-dimensional description of a rotation which can also be described by
    a rotation vector (`v1`, `v2`, `v3`) and a rotation angle :math:`\theta`. The four components
    are calculated to:
    .. math::
       w = \cos(\theta/2)
       x = v_1 \cdot \sin(\theta/2)
       y = v_2 \cdot \sin(\theta/2)
       z = v_3 \cdot \sin(\theta/2)
    Use the :func:`~.from_axisangle` and :func:`~.to_axisangle` to convert to axis-angle
    representation and vice versa. Quaternions can be multiplied by other quaternions, which
    results in a new rotation or with a vector, which results in a rotated vector.

    Attributes
    ----------
    values : float
        The four quaternion values `w`, `x`, `y`, `z`.
    conj : :class:`~.Quaternion`
        The conjugate of the quaternion, representing a tilt in opposite direction.
    matrix : matrix
        The rotation matrix representation of the quaternion

    '''

    NORM_TOLERANCE = 1E-6

    _log = logging.getLogger(__name__+'.Quaternion')

    @property
    def conj(self):
        w, x, y, z = self.values
        return Quaternion((w, -x, -y, -z))

    @property
    def matrix(self):
        w, x, y, z = self.values
        return np.array([[1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
                         [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
                         [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]])

    def __init__(self, values):
        self._log.debug('Calling __init__')
        self.values = values
        self._normalize()
        self._log.debug('Created '+str(self))

    def __mul__(self, other):  # self * other
        self._log.debug('Calling __mul__')
        if isinstance(other, Quaternion):  # Quaternion multiplication
            return self.dot_quat(self, other)
        elif len(other) == 3:  # vector multiplication
            q_vec = Quaternion((0,)+tuple(other))
            q = self.dot_quat(self.dot_quat(self, q_vec), self.conj)
            return q.values[1:]

    def dot_quat(self, q1, q2):
        '''Multiply two :class:`~.Quaternion` objects to create a new one (always normalized).

        Parameters
        ----------
        q1, q2 : :class:`~.Quaternion`
            The quaternion which should be multiplied.

        Returns
        -------
        quaternion : :class:`~.Quaternion`
            The resulting quaternion.

        '''
        self._log.debug('Calling dot_quat')
        w1, x1, y1, z1 = q1.values
        w2, x2, y2, z2 = q2.values
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        return Quaternion((w, x, y, z))

    def _normalize(self):
        self._log.debug('Calling _normalize')
        mag2 = np.sum(n**2 for n in self.values)
        if abs(mag2 - 1.0) > self.NORM_TOLERANCE:
            mag = np.sqrt(mag2)
            self.values = tuple(n / mag for n in self.values)

    @classmethod
    def from_axisangle(cls, vector, theta):
        '''Create a quaternion from an axis-angle representation

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=3)
            Vector around which the rotation is executed.
        theta : float
            Rotation angle.

        Returns
        -------
        quaternion : :class:`~.Quaternion`
            The resulting quaternion.

        '''
        cls._log.debug('Calling from_axisangle')
        x, y, z = vector
        theta /= 2.
        w = np.cos(theta)
        x *= np.sin(theta)
        y *= np.sin(theta)
        z *= np.sin(theta)
        return Quaternion((w, x, y, z))

    def to_axisangle(self):
        '''Convert the quaternion to axis-angle-representation.

        Parameters
        ----------
        None

        Returns
        -------
        vector, theta : :class:`~numpy.ndarray` (N=3), float
            Vector around which the rotation is executed and rotation angle.

        '''
        self._log.debug('Calling to_axisangle')
        w, x, y, z = self.values
        theta = 2.0 * np.arccos(w)
        return np.array((x, y, z)), theta

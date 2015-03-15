# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:07:53 2015

@author: Jan
"""

# TODO: Cleanup

import numpy as np


__all__ = ['Quaternion']


class Quaternion(object):

    NORM_TOLERANCE = 1E-6

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
        self.values = values
        self._normalize()

    def __mul__(self, other):  # self * other
        if isinstance(other, Quaternion):  # Quaternion multiplication
            return self.dot_quat(self, other)
        elif len(other) == 3:  # vector multiplication
            q_vec = Quaternion((0,)+tuple(other))
            q = self.dot_quat(self.dot_quat(self, q_vec), self.conj)
            return q.values[1:]

    def dot_quat(self, q1, q2):
        w1, x1, y1, z1 = q1.values
        w2, x2, y2, z2 = q2.values
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        return Quaternion((w, x, y, z))

    def _normalize(self):
        mag2 = np.sum(n**2 for n in self.values)
        if abs(mag2 - 1.0) > self.NORM_TOLERANCE:
            mag = np.sqrt(mag2)
            self.values = tuple(n / mag for n in self.values)

    @classmethod
    def from_axisangle(cls, vector, theta):
        x, y, z = vector
        theta /= 2.
        w = np.cos(theta)
        x *= np.sin(theta)
        y *= np.sin(theta)
        z *= np.sin(theta)
        return Quaternion((w, x, y, z))

    def to_axisangle(self):
        w, x, y, z = self.values
        theta = 2.0 * np.arccos(w)
        return np.array((x, y, z)), theta

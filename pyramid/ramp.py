# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.Ramp` class which implements polynomial phase ramps."""


import numpy as np

from pyramid.phasemap import PhaseMap


__all__ = ['Ramp']


class Ramp(object):

    '''Class representing a polynomial phase ramp.

    Sometimes additional phase ramps occur in phase maps which do not stem from a magnetization
    distribution inside the FOV. This class allows the construction (and via the derivative
    functions also the reconstruction) of a polynomial ramp. This class is generally constructed
    within the ForwardModel and can be retrieved as its attribute if ramp information should be
    accessed.

    Attributes
    ----------
    data_set : :class:`~dataset.DataSet`
        :class:`~dataset.DataSet` object, which stores all required information calculation.
    order : int or None (default)
        Polynomial order of the additional phase ramp which will be added to the phase maps.
        All ramp parameters have to be at the end of the input vector and are split automatically.
        Default is None (no ramps are added).
    deg_of_freedom : int
        Number of degrees of freedom. This is calculated to ``1 + 2 * order``. There is just one
        degree of freedom for a ramp of order zero (offset), every higher order contributes two
        degrees of freedom.
    param_cache : :class:`numpy.ndarray` (N=2)
        Parameter cache which is used to store the polynomial coefficients. Higher coefficients
        (one for each degree of freedom) are saved along the first axis, values for different
        images along the second axis.
    n : int
        Size of the input space. Coincides with the numer of entries in `param_cache` and
        calculates to ``deg_of_freedom * data_set.count``.

    Notes
    -----
    After a reconstruction the relevant polynomial ramp information is stored in the
    `param_cache`. If a phasemap with index `i` in the DataSet should be corrected use:

    >>> phase_map -= ramp(i, dof_list)

    The optional parameter `dof_list` can be used to specify a list of degrees of freedom which
    should be used for the ramp (e.g. `[0]` will just apply the offset, `[0, 1, 2]` will apply
    the offset and linear ramps in both directions).

    Fitting polynoms of higher orders than `order = 1` is possible but not recommended, because
    features which stem from the magnetization could be covered by the polynom, decreasing the
    phase contribution of the magnetization distribution, leading to a false retrieval.

    '''

    def __init__(self, data_set, order=None):
        self.data_set = data_set
        assert order is None or (isinstance(order, int) and order >= 0), \
            'Order has to be None or a positive integer!'
        self.order = order
        self.deg_of_freedom = (1 + 2 * self.order) if self.order is not None else 0
        self.param_cache = np.zeros((self.deg_of_freedom, self.data_set.count))
        self.n = self.deg_of_freedom * self.data_set.count

    def __call__(self, index, dof_list=None):
        if self.order is None:  # Do nothing if order is None!
            return 0
        else:
            if dof_list is None:  # if no specific list is supplied!
                dof_list = range(self.deg_of_freedom)  # use all available degrees of freedom
            dim_uv = self.data_set.projectors[index].dim_uv
            phase_ramp = np.zeros(dim_uv)
            # Iterate over all degrees of freedom:
            for dof in dof_list:
                # Add the contribution of the current degree of freedom:
                phase_ramp += (self.param_cache[dof][index] *
                               self.create_poly_mesh(self.data_set.a, dof, dim_uv))
            return PhaseMap(self.data_set.a, phase_ramp, mask=np.zeros(dim_uv, dtype=np.bool))

    def jac_dot(self, index):
        '''Calculate the product of the Jacobi matrix with a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of the 3D magnetization distribution. First the `x`, then the `y` and
            lastly the `z` components are listed. Ramp parameters are also added at the end if
            necessary.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Product of the Jacobi matrix (which is not explicitely calculated) with the input
            `vector`. Just the ramp contribution is calculated!

        '''
        if self.order is None:  # Do nothing if order is None!
            return 0
        else:
            dim_uv = self.data_set.projectors[index].dim_uv
            phase_ramp = np.zeros(dim_uv)
            # Iterate over all degrees of freedom:
            for dof in range(self.deg_of_freedom):
                # Add the contribution of the current degree of freedom:
                phase_ramp += (self.param_cache[dof][index] *
                               self.create_poly_mesh(self.data_set.a, dof, dim_uv))
            return np.ravel(phase_ramp)

    def jac_T_dot(self, vector):
        ''''Calculate the transposed ramp parameters from a given `vector`.

        Parameters
        ----------
        vector : :class:`~numpy.ndarray` (N=1)
            Vectorized form of all 2D phase maps one after another in one vector.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Transposed ramp parameters.

        '''
        result = []
        hp = self.data_set.hook_points
        # Iterate over all degrees of freedom:
        for dof in range(self.deg_of_freedom):
            # Iterate over all projectors:
            for i, projector in enumerate(self.data_set.projectors):
                sub_vec = vector[hp[i]:hp[i+1]]
                poly_mesh = self.create_poly_mesh(self.data_set.a, dof, projector.dim_uv)
                # Transposed ramp parameters: summed product of the vector with the poly-mesh:
                result.append(np.sum(sub_vec * np.ravel(poly_mesh)))
        return result

    def extract_ramp_params(self, x):
        '''Extract the ramp parameters of an input vector and return the rest.

        Parameters
        ----------
        x : :class:`~numpy.ndarray` (N=1)
            Input vector which consists of the vectorised magnetization distribution and the ramp
            parameters at the end which will be extracted.

        Returns
        -------
        result_vector : :class:`~numpy.ndarray` (N=1)
            Inpput vector without the extracted ramp parameters.

        Notes
        -----
            This method should always be used before a vector `x` is processed if it is known that
            ramp parameters are present so that other functions do not have to bother with them
            and the :class:`.~ramp` already knows all important parameters for its own functions.

        '''
        if self.order is not None:  # Do nothing if order is None!
            # Split off ramp parameters and fill cache:
            x, ramp_params = np.split(x, [-self.n])
            self.param_cache = ramp_params.reshape((self.deg_of_freedom, self.data_set.count))
        return x

    @classmethod
    def create_poly_mesh(cls, a, deg_of_freedom, dim_uv):
        '''Create a polynomial mesh for the ramp calculation for a specific degree of freedom.

        Parameters
        ----------
        a : float
            Grid spacing which should be used for the ramp.
        deg_of_freedom : int
            Current degree of freedom for which the mesh should be created. 0 corresponds to a
            simple offset, 1 corresponds to a linear ramp in u-direction, 2 to a linear ramp in
            v-direction and so on.
        dim_uv : tuple (N=2)
            Dimensions of the 2D mesh that should be created.

        Returns
        -------
        result_mesh : :class:`~numpy.ndarray` (N=2)
            Polynomial mesh that was created and can be used for further calculations.

        '''
        # Determine if u-direction (u_or_v == 1) or v-direction (u_or_v == 0)!
        u_or_v = (deg_of_freedom - 1) % 2
        # Determine polynomial order:
        order = (deg_of_freedom + 1) // 2
        # Return polynomial mesh:
        return (np.indices(dim_uv)[u_or_v] * a) ** order

    @classmethod
    def create_ramp(cls, a, dim_uv, params):
        '''Class method to create an arbitrary polynomial ramp.

        Parameters
        ----------
        a : float
            Grid spacing which should be used for the ramp.
        dim_uv : tuple (N=2)
            Dimensions of the 2D mesh that should be created.
        params : list
            List of ramp parameters. The first entry corresponds to a simple offset, the second
            and third correspond to a linear ramp in u- and v-direction, respectively and so on.

        Returns
        -------
        phase_ramp : :class:`~pyramid.phasemap.PhaseMap`
            The phase ramp as a :class:`~pyramid.phasemap.PhaseMap` object.

        '''
        phase_ramp = np.zeros(dim_uv)
        dof_list = range(len(params))
        for dof in dof_list:
            phase_ramp += params[dof] * cls.create_poly_mesh(a, dof, dim_uv)
        # Return the phase ramp as a PhaseMap with empty (!) mask:
        return PhaseMap(a, phase_ramp, mask=np.zeros(dim_uv, dtype=np.bool))

# -*- coding: utf-8 -*-
"""Reconstruct magnetic distributions from given phasemaps.

This module reconstructs 3-dimensional magnetic distributions (as :class:`~pyramid.magdata.MagData`
objects) from a given set of phase maps (represented by :class:`~pyramid.phasemap.PhaseMap`
objects) by using several model based reconstruction algorithms which use the forward model
provided by :mod:`~pyramid.projector` and :mod:`~pyramid.phasemapper` and a priori knowledge of
the distribution.

"""


import numpy as np

from scipy.sparse.linalg import cg

from pyramid.kernel import Kernel
from pyramid.forwardmodel import ForwardModel
from pyramid.costfunction import Costfunction, CFAdapterScipyCG
from pyramid.magdata import MagData

import logging


LOG = logging.getLogger(__name__)


class PrintIterator(object):

    '''Iterator class which is responsible to give feedback during reconstruction iterations.

    Parameters
    ----------
    cost : :class:`~.Costfunction`
        :class:`~.Costfunction` class for outputting the `cost` of the current magnetization
        distribution. This should decrease per iteration if the algorithm converges and is only
        printed for a `verbosity` of 2.
    verbosity : {2, 1, 0}, optional
        Parameter defining the verposity of the output. `2` is the default and will show the
        current number of the iteration and the cost of the current distribution. `2` will just
        show the iteration number and `0` will prevent output all together.

    Notes
    -----
    Normally this class should not be used by the user and is instantiated whithin the
    :mod:`~.reconstruction` module itself.

    '''

    LOG = logging.getLogger(__name__+'.PrintIterator')

    def __init__(self, cost, verbosity=2):
        self.LOG.debug('Calling __init__')
        self.cost = cost
        self.verbosity = verbosity
        assert verbosity in {0, 1, 2}, 'verbosity has to be set to 0, 1 or 2!'
        self.iteration = 0
        self.LOG.debug('Created '+str(self))

    def __call__(self, xk):
        self.LOG.debug('Calling __call__')
        if self.verbosity == 0:
            return
        print 'iteration #', self.next(),
        if self.verbosity > 1:
            print 'cost =', self.cost(xk)
        else:
            print ''

    def __repr__(self):
        self.LOG.debug('Calling __repr__')
        return '%s(cost=%r, verbosity=%r)' % (self.__class__, self.cost, self.verbosity)

    def __str__(self):
        self.LOG.debug('Calling __str__')
        return 'PrintIterator(cost=%s, verbosity=%s)' % (self.cost, self.verbosity)

    def next(self):
        self.iteration += 1
        return self.iteration


def optimize_sparse_cg(data, verbosity=2):
    '''Reconstruct a three-dimensional magnetic distribution from given phase maps via the
    conjugate gradient optimizaion method :func:`~.scipy.sparse.linalg.cg`.

    Parameters
    ----------
    data : :class:`~.DataSet`
        :class:`~.DataSet` object containing all phase maps in :class:`~.PhaseMap` objects and all
        projection directions in :class:`~.Projector` objects. These provide the essential
        information for the reconstruction.
    verbosity : {2, 1, 0}, optional
        Parameter defining the verposity of the output. `2` is the default and will show the
        current number of the iteration and the cost of the current distribution. `2` will just
        show the iteration number and `0` will prevent output all together.

    Returns
    -------
    mag_data : :class:`~pyramid.magdata.MagData`
        The reconstructed magnetic distribution as a :class:`~.MagData` object.

    '''
    LOG.debug('Calling optimize_sparse_cg')
    # Set up necessary objects:
    y = data.phase_vec
    kernel = Kernel(data.a, data.dim_uv, data.b_0)
    fwd_model = ForwardModel(data.projectors, kernel)
    cost = Costfunction(y, fwd_model, lam=10**-10)
    # Optimize:
    A = CFAdapterScipyCG(cost)
    b = fwd_model.jac_T_dot(None, y)
    x_opt, info = cg(A, b, callback=PrintIterator(cost, verbosity))
    # Create and return fitting MagData object:
    mag_opt = MagData(fwd_model.a, np.zeros((3,)+fwd_model.dim))
    mag_opt.mag_vec = x_opt
    return mag_opt

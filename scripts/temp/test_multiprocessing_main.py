# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:02:00 2015

@author: Jan
"""


import multiprocessing as mp
import numpy as np

import pyramid as pr
from jutil.taketime import TakeTime


if __name__ == '__main__':

    mp.freeze_support()

    # Parameters:
    b_0 = 1
    order = None
    nprocs = 2
    count = 40

    # Load stuff:
    mag_data = pr.MagData.load_from_netcdf4('magdata.nc')
    with TakeTime('reference') as timer:
        phase_map = pr.pm(mag_data)
        print '--- Reference time:  ', timer.dt

    # Construct dataset:
    data = pr.DataSet(mag_data.a, mag_data.dim, b_0, mag_data.get_mask())
    [data.append(phase_map, pr.SimpleProjector(mag_data.dim)) for i in range(count)]

    # Reference:
    phase_map.display_combined('Reference')

    # Normal ForwardModel:
    fwd_model_norm = pr.ForwardModel(data, order)
    with TakeTime('reference') as timer:
        phase_norm = fwd_model_norm.jac_T_dot(None, data.phase_vec)#data.mag_data.get_vector(data.mask))
#        phase_map_norm = pr.PhaseMap(data.a, phase_norm.reshape(phase_map.dim_uv))
        print '--- Normal time:     ', timer.dt
#    phase_map_norm.display_combined('Normal')
#
#    # Normal ForwardModel:
#    fwd_model_norm = pr.ForwardModel(data, order)
#    with TakeTime('reference') as timer:
#        phase_norm = fwd_model_norm(mag_data.get_vector(data.mask))
##        phase_map_norm = pr.PhaseMap(data.a, phase_norm.reshape(phase_map.dim_uv))
#        print '--- Normal time:     ', timer.dt
##    phase_map_norm.display_combined('Normal')

    # Distributed ForwardModel:
    fwd_model_dist = pr.forwardmodel.DistributedForwardModel(data, ramp_order=order, nprocs=nprocs)
    with TakeTime('distributed') as timer:
        phase_dist = fwd_model_dist.jac_T_dot(None, data.phase_vec)#mag_data.get_vector(data.mask))
#        phase_map_dist = pr.PhaseMap(data.a, phase_dist.reshape(phase_map.dim_uv))
        print '--- Distributed time:', timer.dt
#    phase_map_dist.display_combined('Distributed')
#    fwd_model_dist.finalize()

    # Distributed ForwardModel:
    with TakeTime('distributed') as timer:
        phase_dist = fwd_model_dist.jac_T_dot(None, data.phase_vec)#mag_data.get_vector(data.mask))
#        phase_map_dist = pr.PhaseMap(data.a, phase_dist.reshape(phase_map.dim_uv))
        print '--- Distributed time:', timer.dt
#    phase_map_dist.display_combined('Distributed')
    fwd_model_dist.finalize()

    diff = phase_dist - phase_norm
    print diff.min(), diff.max()
#    assert np.testing.assert_almost_equal(phase_dist, phase_norm, decimal=5)

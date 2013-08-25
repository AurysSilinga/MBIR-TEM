#! python
# -*- coding: utf-8 -*-
"""Create magnetic distribution of alternating filaments"""


import pdb
import traceback
import sys
import os

from numpy import pi

import pyramid.magcreator as mc
from pyramid.magdata import MagData


def create_alternating_filaments():
    '''Calculate, display and save the magnetic distribution of alternating filaments to file.
    Arguments:
        None
    Returns:
        None

    '''
    directory = '../../output/magnetic distributions'
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Input parameters:
    filename = directory + '/mag_dist_alt_filaments.txt'
    dim = (1, 21, 21)  # in px (z, y, x)
    res = 10.0  # in nm
    phi = pi/2
    spacing = 5
    # Create empty MagData object:
    mag_data = MagData(res)
    count = int((dim[1]-1) / spacing) + 1
    for i in range(count):
        pos = i * spacing
        mag_shape = mc.Shapes.filament(dim, (0, pos))
        mag_data.add_magnitude(mc.create_mag_dist_homog(mag_shape, phi))
        phi *= -1  # Switch the angle
    # Plot magnetic distribution
    mag_data.quiver_plot()
    mag_data.save_to_llg(filename)
#    print np.max(mag_data.magnitude)


if __name__ == "__main__":
    try:
        create_alternating_filaments()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

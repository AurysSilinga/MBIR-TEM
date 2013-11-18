# -*- coding: utf-8 -*-
"""
Created on Tue Nov 05 14:52:48 2013

@author: Jan
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter

x = np.linspace(0, 2*pi, 1000)
y = np.where(x<pi, -1/np.tan(x+1E-30), 1/np.tan(x+1E-30))#20*np.arctan(np.exp(x))

def func(x, pos):
    return '{}'.format(x*10)

formatter = FuncFormatter(lambda x, pos: '{:g}'.format(x))

fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.plot(x, y)
axis.set_ylim(-10, 10)
axis.yaxis.set_major_formatter(formatter)

plt.show()
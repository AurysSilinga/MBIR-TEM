# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 14:11:23 2013

@author: Jan
"""

'''PLOT'''
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
plt.pcolormesh(Ar, cmap='Greys')

ticks = ax.get_xticks()*res
ax.set_xticklabels(ticks.astype(int))

ticks = ax.get_yticks()*res
ax.set_yticklabels(ticks.astype(int))

ax.set_title('Fourier Space Approach')
ax.set_xlabel('x-axis [nm]')
ax.set_ylabel('y-axis [nm]')
plt.colorbar()
plt.show()
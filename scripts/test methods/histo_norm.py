# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:30:14 2015

@author: Jan
"""


import matplotlib.pyplot as plt

from scipy import misc
import scipy as sp


im = misc.lena()

plt.figure()
plt.imshow(im, cmap=plt.cm.gray)
plt.show()


def histeq(im, nbr_bins=256):

    # get image histogram
    imhist, bins = sp.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = sp.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


im2, cdf = histeq(im)

plt.figure()
plt.imshow(im2, cmap=plt.cm.gray)
plt.show()

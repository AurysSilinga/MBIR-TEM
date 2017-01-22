# -*- coding: utf-8 -*-
# Copyright 2016 by Forschungszentrum Juelich GmbH
# Author: J. Caron
# Adapted from mpl_toolkits.axes_grid2


import numpy as np

from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, VPacker, HPacker, TextArea
from matplotlib.transforms import blended_transform_factory, IdentityTransform
from matplotlib.patches import Rectangle
from matplotlib import patheffects


class AnchoredScaleBar(AnchoredOffsetbox):
    """Class for a horizontal / vertical  bar with the size in data coordinate of the give axes.

    Attributes
    ----------
    transform : transformation object
        The coordinate frame (typically axes.transData)
    sizex, sizey : float
        Width of x,y bar, in data units. 0 to omit
    labelx, labely : float
        labels for x,y bars; None to omit
    loc : int or string
        position in containing axes
    pad, borderpad : float
        padding, in fraction of the legend font size (or prop)
    sep : float
        separation between labels and bars in points.
    **kwargs : dict
        additional arguments passed to base class constructor

    """
    def __init__(self, transform, width, height, color='w', label=None, txt_pos='above', loc=3,
                 pad=0.5, borderpad=0.1, sep=5, prop=None, axis=None, **kwargs):
        # Create scalebar rectangle:
        bars = AuxTransformBox(transform)
        bars.add_artist(Rectangle((0, 0), width, height, fc=color, linewidth=1,
                        clip_box=axis.bbox, clip_on=True))
        # Create text:
        txt = TextArea(label, minimumdescent=False, textprops={'color': 'w', 'fontsize': 18})
        txt.set_clip_box(axis.bbox)
        txt._text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])
        # Pack both together an create AnchoredOffsetBox:
        if txt_pos == 'above':
            bars = VPacker(children=[txt, bars], align="center", pad=pad, sep=sep)
        elif txt_pos == 'right':
            bars = HPacker(children=[bars, txt], align="center", pad=pad, sep=sep)
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)


def add_scalebar(axis, sampling, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes.

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    Parameters
    ----------
    width : float
        defines width of the bar. should be given as length(nm)/sampling(nm/px)
    height : float
        defines height of the bar.
    ax : matplotlib.axis
        the axis to attach ticks to
    matchx, matchy : bool
        If True, set size of scale bars to spacing between ticks if False, size should be set
        using sizex and sizey params.
    hidex, hidey : if True, hide x-axis and y-axis of parent
    **kwargs : dict
        additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    # Transform that scales the width along the data and leaves height constant at 8 pt (text):
    transform = blended_transform_factory(axis.transData, IdentityTransform())
    # Transform axis borders (1, 1) to data borders to get number of pixels in y and x:
    dim_uv = axis.transLimits.inverted().transform((1, 1))
    # Calculate scale
    scale = np.max((dim_uv[1] / 4, 1)) * sampling
    thresholds = [1, 5, 10, 50, 100, 500, 1000]
    for t in thresholds:  # For larger grids (real images), multiples of threshold look better!
        if scale > t:
            scale = (scale // t) * t
    # Set dimensions:
    width = scale / sampling  # In data coordinate system!
    height = 8  # In display coordinate system!
    # Set label:
    if scale >= 1000:  # Use higher order instead!
        label = '{:.3g} µm'.format(scale/1000)
    else:
        label = '{:.3g} nm'.format(scale)
    # Create and add scalebar:
    sb = AnchoredScaleBar(transform=transform, width=width, height=height,
                          label=label, axis=axis, **kwargs)
    axis.add_artist(sb)
    # Hide axes and return:
    if hidex:
        axis.xaxis.set_visible(False)
    if hidey:
        axis.yaxis.set_visible(False)
    return sb

# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid2
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)

import numpy as np

from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import blended_transform_factory, IdentityTransform


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, width, height, color='w', label=None, txt_pos='above', loc=3,
                 pad=0.5, borderpad=0.1, sep=5, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        from matplotlib import patheffects
        bars = AuxTransformBox(transform)

        bars.add_artist(Rectangle((0, 0), width, height, fc=color, linewidth=1))

        txt = TextArea(label, minimumdescent=False, textprops={'color': 'w', 'fontsize': 12})
        txt._text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])

        if txt_pos == 'above':
            bars = VPacker(children=[txt, bars], align="center", pad=pad, sep=sep)
        elif txt_pos == 'right':
            bars = HPacker(children=[bars, txt], align="center", pad=pad, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)


def add_scalebar(axis, sampling, unit='nm', hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - width: defines width of the bar. should be given as length(nm)/sampling(nm/px)
    - height: defines height of the bar.
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    # Transform that scales the width along the data and leaves height constant at 12 pt (text):
    transform = blended_transform_factory(axis.transData, IdentityTransform())
    # Transform axis borders (1, 1) to data borders to get number of pixels in y and x:
    dim_uv = axis.transLimits.inverted().transform((1, 1))
    # Create text:
    text = '{:.3g}'.format(np.max((dim_uv[1] / 8, 1)) * sampling)  # Nicely rounded number!
    label = '{} {}'.format(text, unit)  # Nicely formatted label with unit!
    # Set dimensions:
    width = float(text) / sampling  # In data coordinate system!
    height = 8  # In display coordinate system!
    # Create and add scalebar:
    sb = AnchoredScaleBar(transform=transform, width=width, height=height, label=label, **kwargs)
    axis.add_artist(sb)
    # Hide axes and return:
    if hidex:
        axis.xaxis.set_visible(False)
    if hidey:
        axis.yaxis.set_visible(False)
    return sb

def scalebar(axis):


    add_scalebar(axis, )

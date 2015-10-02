# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 13:28:39 2015

@author: Jan Caron
"""


from __future__ import division

import os
from PIL import Image
from lxml import etree
import matplotlib.pyplot as plt
import numpy as np


line = 500


def ebic_curr_from_tif(filename):
    print '\n------------------------------------------------------------------\n', filename
    # Read in data (this is a OME-tiff file if you want further information!):
    im = Image.open(filename)  # Open image file (OEM: Open Microscopy Environment)!
    try:
        im.seek(1)  # Go to second page!
    except EOFError:
        print 'Image does not contain a second page and will be skipped!'
        print '------------------------------------------------------------------'
        return
    data = np.asarray(im)  # Convert second page to array!
    print 'Grey values  -->  min: {}, max: {}'.format(data.min(), data.max())
    # Identify metadata node:
    xml = im.tag[700]  # This tag holds the XML-Tree! ('print im.tag' for the whole dictionary)
    root = etree.fromstring(xml)  # Get the root of the XML-Tree!
    ebic_node = root[0][4]  # Go to ebic! print ebic_node[i] for more info about i-th child!
    for i, child_node in enumerate(ebic_node.getchildren()):
        print i, '-->', child_node.tag.split('}')[1]
    # Identify metadata nodes:
    ooffset_node = ebic_node[21]
    contr_node = ebic_node[4]
    invioffset_node = ebic_node[18]
    ioffset_node = ebic_node[19]
    preampgain_node = ebic_node[24]
    # Extract metadata:
    ooffset = float(ooffset_node[0].text)
    contr = float(contr_node[0].text)
    invioffset = float(invioffset_node.text)
    ioffset = float(ioffset_node[0].text)
    preampgain = float(preampgain_node[0].text)
    # Print metadata:
    print 'Ooffset: {} {}'.format(ooffset, ooffset_node[1].text)
    print 'Contr: {} {}'.format(contr, contr_node[1].text)
    print 'InvIOffset: {}'.format(invioffset)  # Caution! Text is DIRECTLY in node (no subnode)!
    print 'IOffset: {} {}'.format(ioffset, ioffset_node[1].text)
    print 'PreampGain: {} {}'.format(preampgain, preampgain_node[1].text)
    # Calculate EBIC currents:
    diss = data/2**16  # Normalized to 2^16 = 65536 grey values!
    currents = (((diss-ooffset-(0.5-0.4993))/contr)-invioffset) / (10**preampgain) * 1E9 * 1.0047
    print 'EBIC current  [nA] -->  min: {}, max: {}'.format(currents.min(), currents.max())
    # Plots lines:
    plt.figure(figsize=(12, 7))
    plt.plot(currents[line, :])
    plt.xlim([0, currents.shape[1]])
    # Plot image:
    plt.figure(figsize=(12, 7))
    plt.imshow(currents, cmap='gray')
    plt.axhline(line)
    plt.colorbar()
    plt.savefig(filename.replace('.tif', '_ebic.png'))
    plt.close('all')  # Close the plot (better for large batches of files)!
    # Save currents:
    np.savetxt(filename.replace('.tif', '_ebic.txt'), currents)
    return currents
    print '------------------------------------------------------------------'


if __name__ == '__main__':
    # Look for tif-files in the directory of this script:
    filepaths = []
    for root, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.tif'):
                filepaths.append(os.path.join(root, filename))
    # Calculate EBIC current for each file:
    for filename in filepaths:
        currents = ebic_curr_from_tif(filename)

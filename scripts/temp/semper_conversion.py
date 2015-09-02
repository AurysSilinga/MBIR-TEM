# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:19:09 2015

@author: Jan
"""

import numpy as np
import struct


filename = 'ramp1.unf'


ICLASS = {1: 'image', 2: 'macro', 3: 'fourier', 4: 'spectrum',
          5: 'correlation', 6: 'undefined', 7: 'walsh', 8: 'position list',
          9: 'histogram', 10: 'display look-up table'}

IFORM = {0: np.byte, 1: np.int32, 2: np.float32, 3: np.complex64}


with open(filename, 'rb') as f:
    # Read header:
    nbytes = np.frombuffer(f.read(4), dtype=np.int32)[0]
    print '\n>>> HEADER'
    print 'nbytes:', nbytes
    header = np.frombuffer(f.read(nbytes), dtype=np.int16)
    print header
    ncol, nrow, nlay = header[:3]
    iclass = ICLASS[header[3]]
    iform = IFORM[header[4]]
    iflag = header[5]
    iversn, remain = divmod(iflag, 10000)
    ilabel, ntitle = divmod(remain, 1000)
    print 'ncol:', ncol
    print 'nrow:', nrow
    print 'nlay:', nlay
    print 'iclass:', iclass
    print 'iform:', iform
    print 'iflag:', iflag
    print 'iversn:', iversn
    print 'ilabel:', ilabel
    print 'ntitle:', ntitle
    if len(header) == 7:
        iformat = header[6]
        print 'iformat:', iformat
    print '>>> END HEADER', np.frombuffer(f.read(4), dtype=np.int32)[0]
    # Read title:
    if ntitle > 0:
        print '\n>>> TITLE', np.frombuffer(f.read(4), dtype=np.int32)[0]
        title_bytes = np.frombuffer(f.read(ntitle), dtype=np.byte)
        title = ''.join(map(chr, title_bytes))
        print title
        print '>>> END TITLE', np.frombuffer(f.read(4), dtype=np.int32)[0]
    # Read label:
    if ilabel:
        print '\n>>> LABEL', np.frombuffer(f.read(4), dtype=np.int32)[0]
        label = np.frombuffer(f.read(512), dtype=np.int16)
        print label
        for i, l in enumerate(label):
            print i+1, '-->', l
        print ''.join([chr(l) for l in label[:6]])
        print 'NCOL:', struct.unpack('>h', ''.join([chr(x) for x in label[6:8]]))[0]
        print 'NROW:', struct.unpack('>h', ''.join([chr(x) for x in label[8:10]]))[0]
        print 'NLAY:', struct.unpack('>h', ''.join([chr(x) for x in label[10:12]]))[0]
        print 'ICCOLN:', struct.unpack('>h', ''.join([chr(x) for x in label[12:14]]))[0]
        print 'ICROWN:', struct.unpack('>h', ''.join([chr(x) for x in label[14:16]]))[0]
        print 'ICLAYN:', struct.unpack('>h', ''.join([chr(x) for x in label[16:18]]))[0]
        print 'ICLASS:', label[18]
        print 'IFORM:', label[19]
        print 'IWP:', label[20]
        print 'Date: {}-{}-{} {}:{}:{}'.format(label[21]+1900, *label[22:27])
        ncrang = label[27]
        print 'NCRANG:', ncrang
        print ''.join([chr(l) for l in label[28:28+ncrang]])
        print 'unused:', label[ncrang+28:55]
        print 'IPLTYP:', label[55]
        print 'reserved:', label[56:99]
        print 'NCOL again:', label[56]
        print 'NROW again:', label[57]
        print 'NLAY again:', label[58]
        print 'ICCOLN again:', label[59]
        print 'ICROWN again:', label[60]
        print 'ICLAYN again:', label[61]
        print 'Real Coords?:', label[62]
        print '# blocks in this label (64b):', label[63]
        print 'unused:', label[64:67]
        print 'DATA cmd V7:', struct.unpack('<f', ''.join([chr(x) for x in label[67:71]]))[0]
        print 'DATA cmd V6:', struct.unpack('<f', ''.join([chr(x) for x in label[71:75]]))[0]
        print 'RealCoord V5 / DZ:', struct.unpack('<f', ''.join([chr(x) for x in label[75:79]]))[0]
        print 'RealCoord V4 / Z0:', struct.unpack('<f', ''.join([chr(x) for x in label[79:83]]))[0]
        print 'RealCoord V3 / DY:', struct.unpack('<f', ''.join([chr(x) for x in label[83:87]]))[0]
        print 'RealCoord V2 / Y0:', struct.unpack('<f', ''.join([chr(x) for x in label[87:91]]))[0]
        print 'RealCoord V1 / DX:', struct.unpack('<f', ''.join([chr(x) for x in label[91:95]]))[0]
        print 'RealCoord V0 / X0:', struct.unpack('<f', ''.join([chr(x) for x in label[95:99]]))[0]
        print 'NTITLE:', label[99]
        print ''.join([str(unichr(l)) for l in label[100:100+ntitle]])
        print 'unused:', label[100+ntitle:243]
        print 'reserved:', label[244:256]
        print 'RealCoord X Unit:', ''.join([chr(l) for l in label[244:248]])
        print 'RealCoord Y Unit:', ''.join([chr(l) for l in label[248:252]])
        print 'RealCoord Z Unit:', ''.join([chr(l) for l in label[252:256]])
        print '>>> END LABEL', np.frombuffer(f.read(4), dtype=np.int32)[0]
    print '\n>>> DATA'
    # Read picture data:
    data = np.empty((nrow, ncol), dtype=iform)
    for k in range(nlay):
        for j in range(nrow):
            rowbytes = np.frombuffer(f.read(4), dtype=np.int32)[0]
#            print 'rowbytes:', rowbytes
            row = np.frombuffer(f.read(rowbytes), dtype=iform)
#            print row
            data[j, :] = row
            rowbytes = np.frombuffer(f.read(4), dtype=np.int32)[0]
#            print 'rowbytes:', rowbytes
    print '>>> END DATA\n'
#    print data

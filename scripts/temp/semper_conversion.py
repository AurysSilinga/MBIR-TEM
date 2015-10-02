# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:19:09 2015

@author: Jan


Picture labels consist of a sequence of bytes
---------------------------------------------
v61:256B | v7:at least 256B, rounded up to multiple of block size 64,
              with max set by LNLAB in params.f
The versions have the same contents for the first 256B, as set out below,
and referred to as the label 'base'; beyond this, in the label 'extension',
the structure is deliberately undefined, allowing you to make your own use
of the additional storage.

From Aug14:
  B1-6    S e m p e r (ic chars)
   7,8    ncol msb,lsb (BIG-ended)
   9,10   nrow msb,lsb
  11,12   nlay msb,lsb
  13,14   ccol msb,lsb
  15,16   crow msb,lsb
  17,18   clay msb,lsb
    19    class: 1-20
            for image,macro,fou,spec,correln,undef,walsh,histog,plist,lut
    20    form: 0,1,2,3,4 = byte,i*2,fp,com,i*4 from Aug08
    21    wp: non-zero if prot
  22-27   creation year(-1900?),month,day,hour,min,sec
    28    v61|v7 # chars in range record | 255
  29-55   v61: min,max values present (ic chars for decimal repn)
           v7: min,max vals as two Fp values in B29-36 (LE ordered)
               followed by 19 unused bytes B37-55
    56    plist type: 1,2,3 = list,opencurve,closedcurve
            acc to EXAMPD - code appears to use explicit numbers
 57,58,59 ncol,nrow,nlay hsb
 60,61,62 ccol,crow,clay hsb
    63    RealCoords flag (non-zero -> DX,DY,DZ,X0,Y0,Z0,units held as below)
    64    v61:0 | v7: # blocks in (this) picture label (incl extn)
  65-67   0 (free at present)
  68-71   DATA cmd V7   4-byte Fp values, order LITTLE-ended
  72-75   DATA cmd V6
  76-79   RealCoord DZ / V5  RealCoord pars as 4-byte Fp values, LITTLE-ended
  80-83   ...       Z0 / V4
  84-87   ...       DY / V3
  88-91   ...       Y0 / V2
  92-95   ...       DX / V1
  96-99   ...       X0 / V0
   100    # chars in title
 101-244  title (ic chars)
 245-248  RealCoord X unit (4 ic chars)
 249-252  RealCoord Y unit (4 ic chars)
 253-256  RealCoord Z unit (4 ic chars)

Aug08-Aug14 labels held no RealCoord information, so:
   B63    'free' and zero - flagging absence of RealCoord information
 101-256  title (12 chars longer than now)

Before Aug08 labels held no hsb for pic sizes, so:
  57-99   all free/zero except for use by DATA cmd
 101-256  title (ic chars)



"""


import numpy as np
import struct
import os
from pyramid import DIR_FILES


os.chdir(os.path.join(DIR_FILES, 'semper'))
filename = 'zz72.unf'


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
        if ncrang == 255:  # Saved as two 4 byte floats!
            print struct.unpack('<f', ''.join([chr(x) for x in label[28:32]]))[0]
            print struct.unpack('<f', ''.join([chr(x) for x in label[32:36]]))[0]
            print 'unused:', label[36:55]
        else:  # Saved as string!
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

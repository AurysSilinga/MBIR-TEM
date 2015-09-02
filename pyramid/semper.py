# -*- coding: utf-8 -*-
# Copyright 2014 by Forschungszentrum Juelich GmbH
# Author: J. Caron
#
"""This module provides the :class:`~.SemperFormat` class which allows reading and writing `.unf`
files which are used in Semper."""


import os
from time import strftime
import numpy as np
import struct

from pyramid.phasemap import PhaseMap
from pyramid.magdata import MagData

import logging


__all__ = ['SemperFormat']


class SemperFormat(object):

    '''Class for importing and exporting Semper `.unf`-files.

    The :class:`~.SemperFormat` class represents a Semper binary file format with a header, which
    holds additional information. Floating point images can be converted to :class:`~.PhaseMaps',
    complex value images can be converted to :class:`~.MagData'-slices (2D magnetization along
    a simple axis) and vice versa with the corresponding methods and classmethods. `.unf`-files
    can also be saved and read from files.

    Attributes
    ----------
    dim : tuple (N=3)
        Dimensions of the data.
    a : float
        Grid spacing (nm per pixel).
    data : :class:`~numpy.ndarray` (N=3)
        The phase map or magnetization information in a 3D array (with one slice).
    iclass : int
        Defines the image class defined in `ICLASS_DICT`. Normally `image` (1) is chosen.
    iform : int
        Defines the data format defined in 'IFORM_DICT'.
    iversn :
        Current `.unf`-format version. Current: 2.
    ilabel : int
        Defines if a label is present (1) or not (0).
    iformat : int
        Defines if the file is formatted (1) or not (0).
    title : string
        Title of the file (not to be confused with the filename).
    iwp : int
        Write protect flag, determining if picture is (1) or is not (0) write-projtected.
    ipltyp : int
        Position type list. Standard seems to be 0 (picture not a position list).
    date : string
        The date of file construction.
    iccoln : int
        Column number of picture origin.
    icrown : int
        Row number of picture origin.
    iclayn : int
        Layer number of picture origin.

    '''

    _log = logging.getLogger(__name__)

    ICLASS_DICT = {1: 'image', 2: 'macro', 3: 'fourier', 4: 'spectrum',
                   5: 'correlation', 6: 'undefined', 7: 'walsh', 8: 'position list',
                   9: 'histogram', 10: 'display look-up table'}

    IFORM_DICT = {0: np.byte, 1: np.int32, 2: np.float32, 3: np.complex64}

    def __init__(self, arg_dict):
        self._log.debug('Calling __init__')
        self.dim = arg_dict['dim']
        self.a = arg_dict['a']
        self.data = arg_dict['data']
        self.iclass = arg_dict['ICLASS']
        self.iform = arg_dict['IFORM']
        self.iversn = arg_dict['IVERSN']
        self.ilabel = arg_dict['ILABEL']
        self.iformat = arg_dict['IFORMAT']
        self.title = arg_dict['title']
        self.iwp = arg_dict['IWP']
        self.ipltyp = arg_dict['IPLTYP']
        self.date = arg_dict['date']
        self.iccoln = arg_dict['ICCOLN']
        self.icrown = arg_dict['ICROWN']
        self.iclayn = arg_dict['ICLAYN']
        self._log.debug('Created '+str(self))

    @classmethod
    def from_file(self, filename):
        '''Load a `.unf`-file into a :class:`~.SemperFormat` object.

        Parameters
        ----------
        filename : string
            The name of the unf-file from which to load the data. Standard format is '\*.unf'.

        Returns
        -------
        semper : :class:`~.SemperFormat` (N=1)
            Semper file format object containing the loaded information.

        '''
        self._log.debug('Calling from_file')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid import DIR_FILES
            directory = os.path.join(DIR_FILES, 'semper')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        with open(filename, 'rb') as f:
            # Read header:
            rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of header
            header = np.frombuffer(f.read(rec_length), dtype=np.int16)
            ncol, nrow, nlay = header[:3]
            dim = (nlay, nrow, ncol)
            iclass = header[3]
            iform = header[4]
            data_format = self.IFORM_DICT[iform]
            iflag = header[5]
            iversn, remain = divmod(iflag, 10000)
            ilabel, ntitle = divmod(remain, 1000)
            iformat = header[6] if len(header) == 7 else None
            assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
            # Read title:
            title = ''
            if ntitle > 0:
                assert np.frombuffer(f.read(4), dtype=np.int32)[0] == ntitle  # length of title
                title_bytes = np.frombuffer(f.read(ntitle), dtype=np.byte)
                title = ''.join(map(chr, title_bytes))
                assert np.frombuffer(f.read(4), dtype=np.int32)[0] == ntitle
            # Read label:
            iwp, date, range_string, ipltype, a = [None] * 5  # Initialization!
            iccoln, icrown, iclayn = [None] * 3
            if ilabel:
                rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of label
                label = np.frombuffer(f.read(512), dtype=np.int16)
                assert ''.join([chr(l) for l in label[:6]]) == 'Semper'
                assert struct.unpack('>h', ''.join([chr(x) for x in label[6:8]]))[0] == ncol
                assert struct.unpack('>h', ''.join([chr(x) for x in label[8:10]]))[0] == nrow
                assert struct.unpack('>h', ''.join([chr(x) for x in label[10:12]]))[0] == nlay
                iccoln = struct.unpack('>h', ''.join([chr(x) for x in label[12:14]]))[0]
                icrown = struct.unpack('>h', ''.join([chr(x) for x in label[14:16]]))[0]
                iclayn = struct.unpack('>h', ''.join([chr(x) for x in label[16:18]]))[0]
                assert label[18] == iclass
                assert label[19] == iform
                iwp = label[20]
                date = '{}-{}-{} {}:{}:{}'.format(label[21]+1900, *label[22:27])
                ncrang = label[27]
                assert sum(label[ncrang+28:55]) == 0  # unused bytes
                ipltyp = label[55]  # position list type
                real_coords = label[62]
                if real_coords:
                    a = struct.unpack('<f', ''.join([chr(x) for x in label[83:87]]))[0]
                    assert a == struct.unpack('<f', ''.join([chr(x) for x in label[91:95]]))[0]
                else:
                    a = 1
                assert ''.join([str(unichr(l)) for l in label[100:100+ntitle]]) == title
                assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
            # Read picture data:
            data = np.empty((nlay, nrow, ncol), dtype=data_format)
            for k in range(nlay):
                for j in range(nrow):
                    rec_length = np.frombuffer(f.read(4), dtype=np.int32)[0]  # length of row
                    row = np.frombuffer(f.read(rec_length), dtype=data_format)
                    data[k, j, :] = row
                    assert np.frombuffer(f.read(4), dtype=np.int32)[0] == rec_length
        arg_dict = {}
        arg_dict['dim'] = dim
        arg_dict['a'] = a
        arg_dict['data'] = data
        arg_dict['ICLASS'] = iclass
        arg_dict['IFORM'] = iform
        arg_dict['IVERSN'] = iversn
        arg_dict['ILABEL'] = ilabel
        arg_dict['IFORMAT'] = iformat
        arg_dict['title'] = title
        arg_dict['IWP'] = iwp
        arg_dict['IPLTYP'] = ipltyp
        arg_dict['date'] = date
        arg_dict['ICCOLN'] = iccoln
        arg_dict['ICROWN'] = icrown
        arg_dict['ICLAYN'] = iclayn
        return SemperFormat(arg_dict)

    def to_file(self, filename='semper.unf', skip_header=False):
        '''Save a :class:`~.SemperFormat` to a file.

        Parameters
        ----------
        filename : string, optional
            The name of the unf-file to which the data should be written.
        skip_header : boolean, optional
            Determines if the header, title and label should be skipped (useful for some other
            programs). Default is False.

        Returns
        -------
        None

        '''
        self._log.debug('Calling to_file')
        # Construct path if filename isn't already absolute:
        if not os.path.isabs(filename):
            from pyramid import DIR_FILES
            directory = os.path.join(DIR_FILES, 'semper')
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = os.path.join(directory, filename)
        nlay, nrow, ncol = self.dim
        with open(filename, 'wb') as f:
            if not skip_header:
                # Create header:
                header = []
                header.extend(reversed(list(self.dim)))  # inverse order!
                header.append(self.iclass)
                header.append(self.iform)
                header.append(self.iversn*10000 + self.ilabel*1000 + len(self.title))
                if self.iformat is not None:
                    header.append(self.iformat)
                # Write header:
                f.write(struct.pack('I', 2*len(header)))  # record length, 4 byte format!
                for element in header:
                    f.write(struct.pack('h', element))  # 2 byte format!
                f.write(struct.pack('I', 2*len(header)))  # record length!
                # Write title:
                f.write(struct.pack('I', len(self.title)))  # record length, 4 byte format!
                f.write(self.title)
                f.write(struct.pack('I', len(self.title)))  # record length!
                # Create label:
                if self.ilabel:
                    label = np.zeros(256, dtype=np.int32)
                    label[:6] = [ord(c) for c in 'Semper']
                    label[6:8] = divmod(ncol, 256)
                    label[8:10] = divmod(nrow, 256)
                    label[10:12] = divmod(nlay, 256)
                    label[12:14] = divmod(self.iccoln, 256)
                    label[14:16] = divmod(self.icrown, 256)
                    label[16:18] = divmod(self.iclayn, 256)
                    label[18] = self.iclass
                    label[19] = self.iform
                    label[20] = self.iwp
                    year, time = self.date.split(' ')
                    label[21:24] = map(int, year.split('-'))
                    label[21] -= 1900
                    label[24:27] = map(int, time.split(':'))
                    range_string = '{:.4g},{:.4g}'.format(self.data.min(), self.data.max())
                    ncrang = len(range_string)
                    label[27] = ncrang
                    label[28:28+ncrang] = [ord(s) for s in range_string]
                    label[55] = self.ipltyp
                    label[62] = 1  # Use real coords!
                    label[75:79] = [ord(x) for x in struct.pack('<f', self.a)]  # RealCoord DZ
                    label[79:83] = [ord(x) for x in struct.pack('<f', self.a*self.iclayn)]  # Z0
                    label[83:87] = [ord(x) for x in struct.pack('<f', self.a)]  # RealCoord DY
                    label[87:91] = [ord(x) for x in struct.pack('<f', self.a*self.iccoln)]  # Y0
                    label[91:95] = [ord(x) for x in struct.pack('<f', self.a)]  # RealCoord DX
                    label[95:99] = [ord(x) for x in struct.pack('<f', self.a*self.icrown)]  # X0
                    label[100:100+len(self.title)] = [ord(s) for s in self.title]
                    label[244:248] = [ord(c) for c in 'nm'] + [0]*2  # x unit
                    label[248:252] = [ord(c) for c in 'nm'] + [0]*2  # y unit
                    label[252:256] = [ord(c) for c in 'nm'] + [0]*2  # z unit
                # Write label:
                if self.ilabel:
                    f.write(struct.pack('I', 2*256))  # record length, 4 byte format!
                    for element in label:
                        f.write(struct.pack('h', element))  # 2 byte format!
                    f.write(struct.pack('I', 2*256))  # record length!
            # Write picture data:
            for k in range(nlay):
                for j in range(nrow):
                    row = self.data[k, j, :]
                    factor = 8 if self.iform == 3 else 4  # complex numbers need more space!
                    f.write(struct.pack('I', factor*ncol))  # record length, 4 byte format!
                    if self.iform == 0:  # bytes:
                        raise Exception('Byte data is not supported! Use int, float or complex!')
                    elif self.iform == 1:  # int:
                        for element in row:
                            f.write(struct.pack('i', element))  # 4 bytes per data entry!
                    elif self.iform == 2:  # float:
                        for element in row:
                            f.write(struct.pack('f', element))  # 4 bytes per data entry!
                    elif self.iform == 3:  # complex:
                        for element in row:
                            f.write(struct.pack('f', element.real))  # 4 bytes per data entry!
                            f.write(struct.pack('f', element.imag))  # 4 bytes per data entry!
                    f.write(struct.pack('I', factor*ncol))  # record length, 4 byte format!

    @classmethod
    def from_phasemap(self, phase_map, title='PYRAMID-PhaseMap'):
        '''Export info from a :class:`~.PhaseMap` object to a :class:`~.SemperFormat` object.

        Parameters
        ----------
        phase_map : :class:`~.PhaseMap`
            Phase map object which should be converted.
        title : string, optional
            Title of the file (not to be confused with the filename).

        Returns
        -------
        semper : :class:`~.SemperFormat` (N=1)
            Semper file format object containing the loaded information.

        '''
        self._log.debug('Calling from_phasemap')
        arg_dict = {}
        arg_dict['dim'] = (1,) + phase_map.dim_uv
        arg_dict['a'] = phase_map.a
        arg_dict['data'] = np.expand_dims(phase_map.phase, axis=0)
        arg_dict['ICLASS'] = 1  # image
        arg_dict['IFORM'] = 2  # float
        arg_dict['IVERSN'] = 2  # current standard
        arg_dict['ILABEL'] = 1  # True
        arg_dict['IFORMAT'] = None  # not needed
        arg_dict['title'] = title
        arg_dict['IWP'] = 0  # seems standard
        arg_dict['IPLTYP'] = 248  # seems standard
        arg_dict['date'] = strftime('%y-%m-%d %H:%M:%S')
        arg_dict['ICCOLN'] = phase_map.dim_uv[1]//2 + 1
        arg_dict['ICROWN'] = phase_map.dim_uv[0]//2 + 1
        arg_dict['ICLAYN'] = 1
        return SemperFormat(arg_dict)

    def to_phasemap(self):
        '''Export info from a :class:`~.SemperFormat` object to a :class:`~.PhaseMap` object.

        Parameters
        ----------
        None

        Returns
        -------
        phase_map : :class:`~.PhaseMap`
            The created phase map object.

        Notes
        -----
        Only works if the `iform` parameter is `2` (floating point values).

        '''
        self._log.debug('Calling to_phasemap')
        assert self.dim[0] == 1  # only one layer!
        assert self.iform in (1, 2)  # only float or int! Phase has to be 2D!
        return PhaseMap(self.a, self.data[0, ...])

    @classmethod
    def from_magdata(self, mag_data, proj_axis='z', ax_slice=0, title='PYRAMID-MagData'):
        '''Export info from a :class:`~.MagData` slice to a :class:`~.SemperFormat` object.

        Parameters
        ----------
        mag_data : :class:`~.PhaseMap`
            Magnetic distribution object from which a slice should be converted.
        proj_axis : {'z', 'y', 'x'}, optional
            The axis, from which the slice is taken. The default is 'z'.
        ax_slice : int, optional
            The slice-index of the axis specified in `proj_axis`. Defaults to zero (first slice).
        title : string, optional
            Title of the file (not to be confused with the filename).

        Returns
        -------
        semper : :class:`~.SemperFormat` (N=1)
            Semper file format object containing the loaded information.

        '''
        self._log.debug('Calling from_magdata')
        # Find slice:
        assert proj_axis == 'z' or proj_axis == 'y' or proj_axis == 'x', \
            'Axis has to be x, y or z (as string).'
        if proj_axis == 'z':  # Slice of the xy-plane with z = ax_slice
            self._log.debug('proj_axis == z')
            u_mag = np.copy(mag_data.magnitude[0][ax_slice, ...])  # x-component
            v_mag = np.copy(mag_data.magnitude[1][ax_slice, ...])  # y-component
        elif proj_axis == 'y':  # Slice of the xz-plane with y = ax_slice
            self._log.debug('proj_axis == y')
            u_mag = np.copy(mag_data.magnitude[0][:, ax_slice, :])  # x-component
            v_mag = np.copy(mag_data.magnitude[2][:, ax_slice, :])  # z-component
        elif proj_axis == 'x':  # Slice of the yz-plane with x = ax_slice
            self._log.debug('proj_axis == x')
            u_mag = np.swapaxes(np.copy(mag_data.magnitude[2][..., ax_slice]), 0, 1)  # z-component
            v_mag = np.swapaxes(np.copy(mag_data.magnitude[1][..., ax_slice]), 0, 1)  # y-component
        # Create data field:
        data = np.expand_dims(u_mag, axis=0) + 1j*np.expand_dims(v_mag, axis=0)
#        np.flipud(data)  # for semper the origin must be in the first line! # TODO: ???
        arg_dict = {}
        arg_dict['dim'] = data.shape
        arg_dict['a'] = mag_data.a
        arg_dict['data'] = data
        arg_dict['ICLASS'] = 1  # image
        arg_dict['IFORM'] = 3  # complex
        arg_dict['IVERSN'] = 2  # current standard
        arg_dict['ILABEL'] = 1  # True
        arg_dict['IFORMAT'] = None  # not needed
        arg_dict['title'] = title
        arg_dict['IWP'] = 0  # seems standard
        arg_dict['IPLTYP'] = 248  # seems standard
        arg_dict['date'] = strftime('%y-%m-%d %H:%M:%S')
        arg_dict['ICCOLN'] = data.shape[2]//2 + 1
        arg_dict['ICROWN'] = data.shape[1]//2 + 1
        arg_dict['ICLAYN'] = 1
        return SemperFormat(arg_dict)

    def to_magdata(self):
        '''Export info from a :class:`~.SemperFormat` object to a :class:`~.MagData` slice.

        Parameters
        ----------
        None

        Returns
        -------
        mag_data : :class:`~.PhaseMap`
            The created magnetic distribution object.

        Notes
        -----
        Only works if the `iform` parameter is `3` (complex values).

        '''
        self._log.debug('Calling to_magdata')
        assert self.dim[0] == 1  # only one layer!
        assert self.iform == 3  # Magnetic slice is described by complex number!
        return MagData(self.a, np.array((self.data.real, self.data.imag, np.zeros(self.dim))))

    def convert_to_abs(self):
        '''Take the absolute of the data. Converts complex to float in the process.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Operates in place!

        '''
        self.data = np.abs(self.data)  # usable for float and complex values!
        self.iform = 2  # update the format (now it's definitely float)!

    def print_info(self):
        '''Print all flag information of the :class:`.~SemperFormat` object.

        Parameters
        ----------
        None

        Returns
        -------
        None

        '''
        self._log.debug('Calling print_info')
        print '\n------------------------------------------------------'
        print self.title
        print self.date, '\n'
        print 'dimensions:', self.dim
        print 'grid spacing:', self.a
        print 'data range:', (self.data.min(), self.data.max()), '\n'
        print 'ICLASS: ', self.ICLASS_DICT[self.iclass]
        print 'IFORM : ', self.IFORM_DICT[self.iform]
        print 'IVERSN: ', self.iversn
        print 'ILABEL: ', self.ilabel == 1
        if self.iformat is not None:
            print 'IFORMAT:', self.iformat
        if self.ilabel:
            print 'IWP   : ', self.iwp
            print 'ICCOLN: ', self.iccoln
            print 'ICROWN: ', self.icrown
            print 'ICLAYN: ', self.iclayn
        print '------------------------------------------------------\n'

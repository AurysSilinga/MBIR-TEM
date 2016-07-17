# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'mag_slicer.ui'
#
# Created: Sun Aug 31 20:39:52 2014
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!
"""GUI for slicing 3D magnetization distributions."""

import os
import sys

from PyQt4 import QtGui, QtCore
from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from ..projector import SimpleProjector
from ..kernel import Kernel
from ..phasemapper import PhaseMapperRDFC
from ..fielddata import VectorData


ui_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mag_slicer.ui')
UI_MainWindow, QMainWindow = loadUiType(ui_location)


class Main(QMainWindow, UI_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.connect(self.checkBoxLog, QtCore.SIGNAL('clicked()'),
                     self.update_slice)
        self.connect(self.checkBoxScale, QtCore.SIGNAL('clicked()'),
                     self.update_slice)
        self.connect(self.spinBoxSlice, QtCore.SIGNAL('valueChanged(int)'),
                     self.update_slice)
        self.connect(self.comboBoxSlice, QtCore.SIGNAL('currentIndexChanged(int)'),
                     self.update_phase)
        self.connect(self.spinBoxGain, QtCore.SIGNAL('valueChanged(double)'),
                     self.update_phase)
        self.connect(self.checkBoxAuto, QtCore.SIGNAL('toggled(bool)'),
                     self.update_phase)
        self.connect(self.checkBoxSmooth, QtCore.SIGNAL('toggled(bool)'),
                     self.update_phase)
        self.connect(self.pushButtonLoad, QtCore.SIGNAL('clicked()'),
                     self.load)
        self.is_mag_data_loaded = False
        self.mag_data = None

    def addmpl(self):
        fig = Figure()
        fig.add_subplot(111, aspect='equal')
        self.canvasMag = FigureCanvas(fig)
        self.layoutMag.addWidget(self.canvasMag)
        self.canvasMag.draw()
        self.toolbarMag = NavigationToolbar(self.canvasMag, self, coordinates=True)
        self.layoutMag.addWidget(self.toolbarMag)
        fig = Figure()
        fig.add_subplot(111, aspect='equal')
        self.canvasPhase = FigureCanvas(fig)
        self.layoutPhase.addWidget(self.canvasPhase)
        self.canvasPhase.draw()
        self.toolbarPhase = NavigationToolbar(self.canvasPhase, self, coordinates=True)
        self.layoutPhase.addWidget(self.toolbarPhase)
        fig = Figure()
        fig.add_subplot(111, aspect='equal')
        self.canvasHolo = FigureCanvas(fig)
        self.layoutHolo.addWidget(self.canvasHolo)
        self.canvasHolo.draw()
        self.toolbarHolo = NavigationToolbar(self.canvasHolo, self, coordinates=True)
        self.layoutHolo.addWidget(self.toolbarHolo)

    def update_phase(self):
        if self.is_mag_data_loaded:
            mode_ind = self.comboBoxSlice.currentIndex()
            if mode_ind == 0:
                self.mode = 'z'
                length = self.mag_data.dim[0] - 1
            elif mode_ind == 1:
                self.mode = 'y'
                length = self.mag_data.dim[1] - 1
            else:
                self.mode = 'x'
                length = self.mag_data.dim[2] - 1
            if self.checkBoxAuto.isChecked():
                gain = 'auto'
            else:
                gain = self.spinBoxGain.value()
            self.projector = SimpleProjector(self.mag_data.dim, axis=self.mode)
            self.spinBoxSlice.setMaximum(length)
            self.scrollBarSlice.setMaximum(length)
            self.spinBoxSlice.setValue(int(length / 2.))
            self.update_slice()
            kernel = Kernel(self.mag_data.a, self.projector.dim_uv)
            self.phase_mapper = PhaseMapperRDFC(kernel)
            self.phase_map = self.phase_mapper(self.projector(self.mag_data))
            self.canvasPhase.figure.axes[0].clear()
            self.phase_map.phase_plot(axis=self.canvasPhase.figure.axes[0], cbar=False)
            if self.checkBoxSmooth.isChecked():
                interpolation = 'bilinear'
            else:
                interpolation = 'none'
            self.canvasHolo.figure.axes[0].clear()
            self.phase_map.holo_plot(axis=self.canvasHolo.figure.axes[0], gain=gain,
                                     interpolation=interpolation)
            self.canvasPhase.draw()
            self.canvasHolo.draw()

    def update_slice(self):
        if self.is_mag_data_loaded:
            self.canvasMag.figure.axes[0].clear()
            self.mag_data.quiver_plot(axis=self.canvasMag.figure.axes[0], proj_axis=self.mode,
                                      ax_slice=self.spinBoxSlice.value(),
                                      log=self.checkBoxLog.isChecked(),
                                      scaled=self.checkBoxScale.isChecked())
            self.canvasMag.draw()

    def load(self):
        try:
            mag_file = QtGui.QFileDialog.getOpenFileName(self, 'Open Data File', '',
                                                         'HDF5 files (*.hdf5)')
        except ValueError:
            return  # Abort if no conf_path is selected!
        import hyperspy.api as hs
        print(hs.load(mag_file))
        self.mag_data = VectorData.load_from_hdf5(mag_file)
        if not self.is_mag_data_loaded:
            self.addmpl()
        self.is_mag_data_loaded = True
        self.comboBoxSlice.setCurrentIndex(0)
        self.update_phase()


def gui_mag_slicer():
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec()
    return main.mag_data

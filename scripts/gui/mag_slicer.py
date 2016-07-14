# -*- coding: utf-8 -*-
# Form implementation generated from reading ui file 'mag_slicer2.ui'
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

import pyramid as pr


UI_MainWindow, QMainWindow = loadUiType('mag_slicer.ui')


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
        self.mag_data_loaded = False

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
        if self.mag_data_loaded:
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
            self.projector = pr.SimpleProjector(self.mag_data.dim, axis=self.mode)
            self.spinBoxSlice.setMaximum(length)
            self.scrollBarSlice.setMaximum(length)
            self.spinBoxSlice.setValue(int(length / 2.))
            self.update_slice()
            kernel = pr.Kernel(self.mag_data.a, self.projector.dim_uv)
            self.phase_mapper = pr.PhaseMapperRDFC(kernel)
            self.phase_map = self.phase_mapper(self.projector(self.mag_data))
            self.canvasPhase.figure.axes[0].clear()
            self.phase_map.display_phase(axis=self.canvasPhase.figure.axes[0], cbar=False)
            if self.checkBoxSmooth.isChecked():
                interpolation = 'bilinear'
            else:
                interpolation = 'none'
            self.canvasHolo.figure.axes[0].clear()
            self.phase_map.display_holo(axis=self.canvasHolo.figure.axes[0], gain=gain,
                                        interpolation=interpolation)
            self.canvasPhase.draw()
            self.canvasHolo.draw()

    def update_slice(self):
        if self.mag_data_loaded:
            self.canvasMag.figure.axes[0].clear()
            self.mag_data.quiver_plot(axis=self.canvasMag.figure.axes[0], proj_axis=self.mode,
                                      ax_slice=self.spinBoxSlice.value(),
                                      log=self.checkBoxLog.isChecked(),
                                      scaled=self.checkBoxScale.isChecked())
            self.canvasMag.draw()

    def load(self):
        try:
            mag_file = QtGui.QFileDialog.getOpenFileName(self, str_caption='Open Data File',
                                                         str_filter='HDF5 files (*.hdf5)')
        except ValueError:
            return  # Abort if no conf_path is selected!
        self.mag_data = pr.VectorData.load_from_hdf5(mag_file)
        if not self.mag_data_loaded:
            self.addmpl()
        self.mag_data_loaded = True
        self.comboBoxSlice.setCurrentIndex(0)
        self.update_phase()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'phasemap_creator.ui'
#
# Created: Thu Sep 24 11:42:11 2015
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!
"""GUI for setting up PhasMaps from existing data in different formats."""

import os
import sys

from PyQt4 import QtGui, QtCore
from PyQt4.uic import loadUiType

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import pyramid as pr
import hyperspy.api as hs
from PIL import Image


UI_MainWindow, QMainWindow = loadUiType('phasemap_creator.ui')


class Main(QMainWindow, UI_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.connect(self.pushButton_phase, QtCore.SIGNAL('clicked()'),
                     self.load_phase)
        self.connect(self.pushButton_mask, QtCore.SIGNAL('clicked()'),
                     self.load_mask)
        self.connect(self.pushButton_conf, QtCore.SIGNAL('clicked()'),
                     self.load_conf)
        self.connect(self.pushButton_export, QtCore.SIGNAL('clicked()'),
                     self.export)
        self.connect(self.horizontalScrollBar, QtCore.SIGNAL('valueChanged(int)'),
                     self.doubleSpinBox_thres.setValue)
        self.connect(self.doubleSpinBox_thres, QtCore.SIGNAL('valueChanged(double)'),
                     self.horizontalScrollBar.setValue)
        self.connect(self.checkBox_mask, QtCore.SIGNAL('clicked()'),
                     self.update_phasemap)
        self.connect(self.checkBox_conf, QtCore.SIGNAL('clicked()'),
                     self.update_phasemap)
        self.connect(self.doubleSpinBox_a, QtCore.SIGNAL('editingFinished()'),
                     self.update_phasemap)
        self.connect(self.doubleSpinBox_thres, QtCore.SIGNAL('valueChanged(double)'),
                     self.update_mask)
        self.phase_loaded = False
        self.mask_loaded = False
        self.dir = ''

    def addmpl(self):
        fig = Figure()
        fig.add_subplot(111, aspect='equal')
        self.canvas = FigureCanvas(fig)
        self.mplLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self, coordinates=True)
        self.mplLayout.addWidget(self.toolbar)

    def update_phasemap(self):
        if self.phase_loaded:
            self.phase_map.a = self.doubleSpinBox_a.value()
            show_mask = self.checkBox_mask.isChecked()
            show_conf = self.checkBox_conf.isChecked()
            self.canvas.figure.axes[0].clear()
            self.canvas.figure.axes[0].hold(True)
            self.phase_map.display_phase('PhaseMap', axis=self.canvas.figure.axes[0],
                                         show_mask=show_mask, show_conf=show_conf, cbar=False)
            self.canvas.draw()

    def update_mask(self):
        if self.mask_loaded:
            threshold = self.doubleSpinBox_thres.value()
            mask_img = Image.fromarray(self.raw_mask)
            mask = np.asarray(mask_img.resize(list(reversed(self.phase_map.dim_uv))))
            self.phase_map.mask = np.where(mask >= threshold, True, False)
            self.update_phasemap()

    def load_phase(self):
        try:
            self.phase_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Phase', self.dir)
            if self.phase_path.endswith('.txt'):
                phase = np.genfromtxt(self.phase_path, delimiter=',')
                self.phase_map = pr.PhaseMap(1., phase)
            else:
                self.phase_map = pr.PhaseMap.from_signal(hs.load(self.phase_path))
        except ValueError:
            return  # Abort if no phase_path is selected!
        self.doubleSpinBox_a.setValue(self.phase_map.a)
        self.dir = os.path.join(os.path.dirname(self.phase_path))
        if not self.phase_loaded:
            self.addmpl()
        self.pushButton_mask.setEnabled(True)
        self.pushButton_conf.setEnabled(True)
        self.pushButton_export.setEnabled(True)
        self.phase_loaded = True
        self.horizontalScrollBar.setMinimum(0)
        self.horizontalScrollBar.setMaximum(0)
        self.horizontalScrollBar.setEnabled(False)
        self.doubleSpinBox_thres.setMinimum(0)
        self.doubleSpinBox_thres.setMaximum(0)
        self.doubleSpinBox_thres.setValue(0)
        self.doubleSpinBox_thres.setEnabled(False)
        self.mask_loaded = False
        self.update_phasemap()

    def load_mask(self):
        try:
            mask_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Mask', self.dir)
            if mask_path.endswith('.txt'):
                self.raw_mask = np.genfromtxt(mask_path, delimiter=',')
            else:
                self.raw_mask = hs.load(mask_path).data
        except ValueError:
            return  # Abort if no mask_path is selected!
        mask_min = self.raw_mask.min()
        mask_max = self.raw_mask.max()
        self.horizontalScrollBar.setEnabled(True)
        self.horizontalScrollBar.setMinimum(mask_min)
        self.horizontalScrollBar.setMaximum(mask_max)
        self.horizontalScrollBar.setSingleStep((mask_max - mask_min) / 255.)
        self.horizontalScrollBar.setValue((mask_max - mask_min) / 2.)
        self.doubleSpinBox_thres.setEnabled(True)
        self.doubleSpinBox_thres.setMinimum(mask_min)
        self.doubleSpinBox_thres.setMaximum(mask_max)
        self.doubleSpinBox_thres.setSingleStep((mask_max - mask_min) / 255.)
        self.doubleSpinBox_thres.setValue((mask_max - mask_min) / 2.)
        self.mask_loaded = True
        self.update_mask()

    def load_conf(self):
        try:
            conf_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Confidence', self.dir)
            if conf_path.endswith('.txt'):
                confidence = np.genfromtxt(conf_path, delimiter=',')
            else:
                confidence = hs.load(conf_path).data
        except ValueError:
            return  # Abort if no conf_path is selected!
        confidence = confidence / confidence.max()
        confidence = np.asarray(Image.fromarray(confidence).resize(self.phase_map.dim_uv))
        self.phase_map.confidence = confidence
        self.update_phasemap()

    def export(self):
        try:
            export_name = os.path.splitext(os.path.basename(self.phase_path))[0]
            export_default = os.path.join(self.dir, 'phasemap_gui_{}.hdf5'.format(export_name))
            export_path = QtGui.QFileDialog.getSaveFileName(self, 'Export PhaseMap',
                                                            export_default,
                                                            'HDF5 (*.hdf5)')
            self.phase_map.to_signal().save(export_path, overwrite=True)
        except (ValueError, AttributeError):
            return  # Abort if no export_path is selected or self.phase_map doesn't exist yet!


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

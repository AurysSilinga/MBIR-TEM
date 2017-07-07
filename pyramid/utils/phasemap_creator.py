# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'phasemap_creator.ui'
#
# Created: Thu Sep 24 11:42:11 2015
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!
"""GUI for setting up PhasMaps from existing data in different formats."""

import logging

import os
import sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

from PIL import Image

import numpy as np

#import hyperspy.api as hs  #  TODO: Necessary?

import pyramid as pr

try:
    from PyQt5 import QtGui, QtCore
    from PyQt5.uic import loadUiType
except ImportError:
    from PyQt4 import QtGui, QtCore
    from PyQt4.uic import loadUiType

__all__ = ['gui_phasemap_creator']
_log = logging.getLogger(__name__)


ui_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'phasemap_creator.ui')
UI_MainWindow, QMainWindow = loadUiType(ui_location)


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
        self.phasemap = None

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
            self.phasemap.a = self.doubleSpinBox_a.value()
            show_mask = self.checkBox_mask.isChecked()
            show_conf = self.checkBox_conf.isChecked()
            self.canvas.figure.axes[0].clear()
            self.canvas.figure.axes[0].hold(True)
            self.phasemap.plot_phase('PhaseMap', axis=self.canvas.figure.axes[0],
                                     show_mask=show_mask, show_conf=show_conf, cbar=False)
            self.canvas.draw()

    def update_mask(self):
        if self.mask_loaded:
            threshold = self.doubleSpinBox_thres.value()
            mask_img = Image.fromarray(self.raw_mask)
            mask = np.asarray(mask_img.resize(list(reversed(self.phasemap.dim_uv))))
            self.phasemap.mask = np.where(mask >= threshold, True, False)
            self.update_phasemap()

    def load_phase(self):
        try:
            self.phase_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Phase', self.dir)
            self.phasemap = pr.file_io.io_phasemap._load(self.phase_path, as_phasemap=True)
        except ValueError:
            return  # Abort if no phase_path is selected!
        self.doubleSpinBox_a.setValue(self.phasemap.a)
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
            self.raw_mask = pr.file_io.io_phasemap._load(mask_path)
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
            confidence = pr.file_io.io_phasemap._load(conf_path)
        except ValueError:
            return  # Abort if no conf_path is selected!
        confidence = confidence.astype(float) / confidence.max() + 1e-30
        self.phasemap.confidence = confidence
        self.update_phasemap()

    def export(self):
        try:
            export_name = os.path.splitext(os.path.basename(self.phase_path))[0]
            export_default = os.path.join(self.dir, 'phasemap_gui_{}.hdf5'.format(export_name))
            export_path = QtGui.QFileDialog.getSaveFileName(self, 'Export PhaseMap',
                                                            export_default, 'HDF5 (*.hdf5)')
            self.phasemap.to_signal().save(export_path, overwrite=True)
        except (ValueError, AttributeError):
            return  # Abort if no export_path is selected or self.phasemap doesn't exist yet!


def gui_phasemap_creator():
    """Call the GUI for phasemap creation."""
    _log.debug('Calling gui_phasemap_creator')
    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    app.exec()
    return main.phasemap

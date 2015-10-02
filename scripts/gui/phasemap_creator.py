# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'phasemap_creator.ui'
#
# Created: Thu Sep 24 11:42:11 2015
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!


import os
import sys

import numpy as np

from PyQt4 import QtCore, QtGui

from matplotlibwidget import MatplotlibWidget

from PIL import Image
from pyramid import PhaseMap
import ercpy


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class UI_PhaseMapCreatorMain(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        Form = self
        self.setGeometry(100, 100, 900, 900)
        self.setWindowTitle('PhaseMap Creator')
        Form.setObjectName(_fromUtf8('PhaseMap Creator'))
        ###########################################################################################
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.mplwidget = MatplotlibWidget(Form)
        self.mplwidget.setObjectName(_fromUtf8("mplwidget"))
        self.verticalLayout.addWidget(self.mplwidget)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton_phase = QtGui.QPushButton(Form)
        self.pushButton_phase.setObjectName(_fromUtf8("pushButton_phase"))
        self.horizontalLayout.addWidget(self.pushButton_phase)
        self.pushButton_mask = QtGui.QPushButton(Form)
        self.pushButton_mask.setObjectName(_fromUtf8("pushButton_mask"))
        self.horizontalLayout.addWidget(self.pushButton_mask)
        self.pushButton_conf = QtGui.QPushButton(Form)
        self.pushButton_conf.setObjectName(_fromUtf8("pushButton_conf"))
        self.horizontalLayout.addWidget(self.pushButton_conf)
        self.pushButton_export = QtGui.QPushButton(Form)
        self.pushButton_export.setObjectName(_fromUtf8("pushButton_export"))
        self.horizontalLayout.addWidget(self.pushButton_export)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_a = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_a.sizePolicy().hasHeightForWidth())
        self.label_a.setSizePolicy(sizePolicy)
        self.label_a.setObjectName(_fromUtf8("label_a"))
        self.horizontalLayout_2.addWidget(self.label_a)
        self.doubleSpinBox_a = QtGui.QDoubleSpinBox(Form)
        self.doubleSpinBox_a.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                          QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_a.setMaximum(1000)
        self.doubleSpinBox_a.setProperty("value", 1)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_a.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_a.setSizePolicy(sizePolicy)
        self.doubleSpinBox_a.setObjectName(_fromUtf8("doubleSpinBox_a"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_a)
        self.label_thres = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_thres.sizePolicy().hasHeightForWidth())
        self.label_thres.setSizePolicy(sizePolicy)
        self.label_thres.setObjectName(_fromUtf8("label_thres"))
        self.horizontalLayout_2.addWidget(self.label_thres)
        self.doubleSpinBox_thres = QtGui.QDoubleSpinBox(Form)
        self.doubleSpinBox_thres.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                              QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_thres.setMaximum(0)
        self.doubleSpinBox_thres.setProperty("value", 0)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doubleSpinBox_thres.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox_thres.setSizePolicy(sizePolicy)
        self.doubleSpinBox_thres.setObjectName(_fromUtf8("doubleSpinBox_thres"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBox_thres)
        self.horizontalScrollBar = QtGui.QScrollBar(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setMaximum(0)
        self.horizontalScrollBar.setObjectName(_fromUtf8("horizontalScrollBar"))
        self.horizontalLayout_2.addWidget(self.horizontalScrollBar)
        self.checkBox_mask = QtGui.QCheckBox(Form)
        self.checkBox_mask.setChecked(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_mask.sizePolicy().hasHeightForWidth())
        self.checkBox_mask.setSizePolicy(sizePolicy)
        self.checkBox_mask.setObjectName(_fromUtf8("checkBox_mask"))
        self.horizontalLayout_2.addWidget(self.checkBox_mask)
        self.checkBox_conf = QtGui.QCheckBox(Form)
        self.checkBox_conf.setChecked(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_conf.sizePolicy().hasHeightForWidth())
        self.checkBox_conf.setSizePolicy(sizePolicy)
        self.checkBox_conf.setObjectName(_fromUtf8("checkBox_conf"))
        self.horizontalLayout_2.addWidget(self.checkBox_conf)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        ###########################################################################################
        self.retranslateUi(Form)
        # CONNECTS:
        self.connect(self.pushButton_phase, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.load_phase)
        self.connect(self.pushButton_mask, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.load_mask)
        self.connect(self.pushButton_conf, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.load_conf)
        self.connect(self.pushButton_export, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.export)
        self.connect(self.horizontalScrollBar, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")),
                     self.doubleSpinBox_thres.setValue)
        self.connect(self.doubleSpinBox_thres, QtCore.SIGNAL(_fromUtf8("valueChanged(double)")),
                     self.horizontalScrollBar.setValue)
        self.connect(self.checkBox_mask, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_phasemap)
        self.connect(self.checkBox_conf, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_phasemap)
        self.connect(self.doubleSpinBox_a, QtCore.SIGNAL(_fromUtf8('editingFinished()')),
                     self.update_phasemap)
        self.connect(self.doubleSpinBox_thres, QtCore.SIGNAL(_fromUtf8('valueChanged(double)')),
                     self.update_mask)
        QtCore.QMetaObject.connectSlotsByName(Form)
        # OTHER STUFF:
        self.mplwidget.axes.set_visible(False)
        self.mplwidget.axes.set_visible(False)
        self.mplwidget.axes.set_visible(False)
        self.pushButton_mask.setEnabled(False)
        self.pushButton_conf.setEnabled(False)
        self.pushButton_export.setEnabled(False)
        self.doubleSpinBox_thres.setEnabled(False)
        self.horizontalScrollBar.setEnabled(False)
        self.phase_loaded = False
        self.mask_loaded = False
        self.dir = ''

    def retranslateUi(self, Form):
        ###########################################################################################
        Form.setWindowTitle(_translate("Form", "PhaseMap Creator", None))
        self.pushButton_phase.setText(_translate("Form", "Load Phase", None))
        self.pushButton_mask.setText(_translate("Form", "Load Mask", None))
        self.pushButton_conf.setText(_translate("Form", "Load Confidence", None))
        self.pushButton_export.setText(_translate("Form", "Export PhaseMap", None))
        self.label_a.setText(_translate("Form", "Grid Spacing [nm]:", None))
        self.label_thres.setText(_translate("Form", "Mask Threshold:", None))
        self.checkBox_mask.setText(_translate("Form", "show mask", None))
        self.checkBox_conf.setText(_translate("Form", "show confidence", None))
        ###########################################################################################

    def update_phasemap(self):
        if self.phase_loaded:
            self.phase_map.a = self.doubleSpinBox_a.value()
            show_mask = self.checkBox_mask.isChecked()
            show_conf = self.checkBox_conf.isChecked()
            self.mplwidget.axes.clear()
            self.mplwidget.axes.hold(True)
            self.phase_map.display_phase('PhaseMap', axis=self.mplwidget.axes,
                                         show_mask=show_mask, show_conf=show_conf, cbar=False)
            self.mplwidget.draw()

    def update_mask(self):
        if self.mask_loaded:
            threshold = self.doubleSpinBox_thres.value()
            mask = np.asarray(Image.fromarray(self.emd['mask']).resize(self.emd['phase'].shape))
            self.phase_map.mask = np.where(mask >= threshold, True, False)
            self.update_phasemap()

    def load_phase(self):
        try:
            self.phase_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Phase', self.dir)
            self.emd = ercpy.EMD()
            self.emd.add_signal_from_file(self.phase_path, name='phase')
        except ValueError:
            return  # Abort if no phase_path is selected!
        if self.emd.signals['phase'].axes_manager[0].scale != 1.:
            self.doubleSpinBox_a.setValue(self.emd.signals['phase'].axes_manager[0].scale)
        self.dir = os.path.join(os.path.dirname(self.phase_path))
        self.phase_map = PhaseMap(self.doubleSpinBox_a.value(), self.emd['phase'])
        self.mplwidget.axes.set_visible(True)
        self.mplwidget.axes.set_visible(True)
        self.mplwidget.axes.set_visible(True)
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
            self.emd.add_signal_from_file(mask_path, name='mask')
        except ValueError:
            return  # Abort if no mask_path is selected!
        mask_min = self.emd['mask'].min()
        mask_max = self.emd['mask'].max()
        self.horizontalScrollBar.setEnabled(True)
        self.horizontalScrollBar.setMinimum(mask_min)
        self.horizontalScrollBar.setMaximum(mask_max)
        self.horizontalScrollBar.setSingleStep((mask_max-mask_min)/255.)
        self.horizontalScrollBar.setValue((mask_max-mask_min)/2.)
        self.doubleSpinBox_thres.setEnabled(True)
        self.doubleSpinBox_thres.setMinimum(mask_min)
        self.doubleSpinBox_thres.setMaximum(mask_max)
        self.doubleSpinBox_thres.setSingleStep((mask_max-mask_min)/255.)
        self.doubleSpinBox_thres.setValue((mask_max-mask_min)/2.)
        self.mask_loaded = True
        self.update_mask()

    def load_conf(self):
        try:
            conf_path = QtGui.QFileDialog.getOpenFileName(self, 'Load Confidence', self.dir)
        except ValueError:
            return  # Abort if no conf_path is selected!
        self.emd.add_signal_from_file(conf_path, name='confidence')
        confidence = self.emd['confidence'] / self.emd['confidence'].max()
        confidence = np.asarray(Image.fromarray(confidence).resize(self.emd['phase'].shape))
        self.phase_map.confidence = confidence
        self.update_phasemap()

    def export(self):
        try:
            export_name = os.path.splitext(os.path.basename(self.phase_path))[0]
            export_default = os.path.join(self.dir, 'phasemap_gui_{}.nc'.format(export_name))
            export_path = QtGui.QFileDialog.getSaveFileName(self, 'Export PhaseMap',
                                                            export_default,
                                                            'EMD/NetCDF4 (*.emd *.nc)')
            if export_path.endswith('.nc'):
                self.phase_map.save_to_netcdf4(export_path)
            elif export_path.endswith('.emd'):
                self.emd.save_to_emd(export_path)
        except ValueError:
            return  # Abort if no export_path is selected!

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = UI_PhaseMapCreatorMain()
    widget.show()
    sys.exit(app.exec_())

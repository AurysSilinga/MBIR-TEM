# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mag_slicer2.ui'
#
# Created: Sun Aug 31 20:39:52 2014
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!


import os
import sys

import pyramid
from pyramid.kernel import Kernel
from pyramid.magdata import MagData
from pyramid.phasemapper import PhaseMapperRDFC
from pyramid.projector import SimpleProjector

from PyQt4 import QtCore, QtGui
from matplotlibwidget import MatplotlibWidget

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


class UI_MagSlicerMain(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        Form = self
        self.setGeometry(100, 100, 1405, 550)
        self.setWindowTitle('Mag Slicer')
        ###########################################################################################
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButtonLoad = QtGui.QPushButton(Form)
        self.pushButtonLoad.setObjectName(_fromUtf8("pushButtonLoad"))
        self.horizontalLayout.addWidget(self.pushButtonLoad)
        self.checkBoxScale = QtGui.QCheckBox(Form)
        self.checkBoxScale.setChecked(True)
        self.checkBoxScale.setTristate(False)
        self.checkBoxScale.setObjectName(_fromUtf8("checkBoxScale"))
        self.horizontalLayout.addWidget(self.checkBoxScale)
        self.checkBoxLog = QtGui.QCheckBox(Form)
        self.checkBoxLog.setObjectName(_fromUtf8("checkBoxLog"))
        self.horizontalLayout.addWidget(self.checkBoxLog)
        self.comboBoxSlice = QtGui.QComboBox(Form)
        self.comboBoxSlice.setObjectName(_fromUtf8("comboBoxSlice"))
        self.comboBoxSlice.addItem(_fromUtf8(""))
        self.comboBoxSlice.addItem(_fromUtf8(""))
        self.comboBoxSlice.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.comboBoxSlice)
        self.labelSlice = QtGui.QLabel(Form)
        self.labelSlice.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                     QtCore.Qt.AlignVCenter)
        self.labelSlice.setObjectName(_fromUtf8("labelSlice"))
        self.horizontalLayout.addWidget(self.labelSlice)
        self.spinBoxSlice = QtGui.QSpinBox(Form)
        self.spinBoxSlice.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                       QtCore.Qt.AlignVCenter)
        self.spinBoxSlice.setMaximum(0)
        self.spinBoxSlice.setProperty("value", 0)
        self.spinBoxSlice.setObjectName(_fromUtf8("spinBoxSlice"))
        self.horizontalLayout.addWidget(self.spinBoxSlice)
        self.scrollBarSlice = QtGui.QScrollBar(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollBarSlice.sizePolicy().hasHeightForWidth())
        self.scrollBarSlice.setSizePolicy(sizePolicy)
        self.scrollBarSlice.setMinimumSize(QtCore.QSize(400, 0))
        self.scrollBarSlice.setMaximum(0)
        self.scrollBarSlice.setOrientation(QtCore.Qt.Horizontal)
        self.scrollBarSlice.setObjectName(_fromUtf8("scrollBarSlice"))
        self.horizontalLayout.addWidget(self.scrollBarSlice)
        self.checkBoxSmooth = QtGui.QCheckBox(Form)
        self.checkBoxSmooth.setChecked(True)
        self.checkBoxSmooth.setObjectName(_fromUtf8("checkBoxSmooth"))
        self.horizontalLayout.addWidget(self.checkBoxSmooth)
        self.checkBoxAuto = QtGui.QCheckBox(Form)
        self.checkBoxAuto.setChecked(True)
        self.checkBoxAuto.setObjectName(_fromUtf8("checkBoxAuto"))
        self.horizontalLayout.addWidget(self.checkBoxAuto)
        self.labelGain = QtGui.QLabel(Form)
        self.labelGain.setObjectName(_fromUtf8("labelGain"))
        self.horizontalLayout.addWidget(self.labelGain)
        self.spinBoxGain = QtGui.QDoubleSpinBox(Form)
        self.spinBoxGain.setEnabled(False)
        self.spinBoxGain.setMaximum(1000000.0)
        self.spinBoxGain.setSingleStep(0.1)
        self.spinBoxGain.setProperty("value", 1.0)
        self.spinBoxGain.setObjectName(_fromUtf8("spinBoxGain"))
        self.horizontalLayout.addWidget(self.spinBoxGain)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayoutPlots = QtGui.QHBoxLayout()
        self.horizontalLayoutPlots.setSpacing(0)
        self.horizontalLayoutPlots.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.horizontalLayoutPlots.setObjectName(_fromUtf8("horizontalLayoutPlots"))
        self.mplWidgetMag = MatplotlibWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplWidgetMag.sizePolicy().hasHeightForWidth())
        self.mplWidgetMag.setSizePolicy(sizePolicy)
        self.mplWidgetMag.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.mplWidgetMag.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mplWidgetMag.setAutoFillBackground(False)
        self.mplWidgetMag.setObjectName(_fromUtf8("mplWidgetMag"))
        self.horizontalLayoutPlots.addWidget(self.mplWidgetMag)
        self.mplWidgetHolo = MatplotlibWidget(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplWidgetHolo.sizePolicy().hasHeightForWidth())
        self.mplWidgetHolo.setSizePolicy(sizePolicy)
        self.mplWidgetHolo.setObjectName(_fromUtf8("mplWidgetHolo"))
        self.horizontalLayoutPlots.addWidget(self.mplWidgetHolo)
        self.mplWidgetPhase = MatplotlibWidget(Form)
        self.mplWidgetPhase.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mplWidgetPhase.sizePolicy().hasHeightForWidth())
        self.mplWidgetPhase.setSizePolicy(sizePolicy)
        self.mplWidgetPhase.setObjectName(_fromUtf8("mplWidgetPhase"))
        self.horizontalLayoutPlots.addWidget(self.mplWidgetPhase)
        self.verticalLayout.addLayout(self.horizontalLayoutPlots)
        ###########################################################################################
        self.retranslateUi(Form)
        self.connect(self.spinBoxSlice, QtCore.SIGNAL(_fromUtf8("editingFinished()")),
                     self.scrollBarSlice.setValue)
        self.connect(self.scrollBarSlice, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")),
                     self.spinBoxSlice.setValue)
        self.connect(self.checkBoxAuto, QtCore.SIGNAL(_fromUtf8("toggled(bool)")),
                     self.spinBoxGain.setDisabled)
        self.connect(self.checkBoxLog, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_slice)
        self.connect(self.checkBoxScale, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_slice)
        self.connect(self.spinBoxSlice, QtCore.SIGNAL(_fromUtf8('valueChanged(int)')),
                     self.update_slice)
        self.connect(self.comboBoxSlice, QtCore.SIGNAL(_fromUtf8('currentIndexChanged(int)')),
                     self.update_phase)
        self.connect(self.spinBoxGain, QtCore.SIGNAL(_fromUtf8('editingFinished()')),
                     self.update_phase)
        self.connect(self.checkBoxAuto, QtCore.SIGNAL(_fromUtf8('toggled(bool)')),
                     self.update_phase)
        self.connect(self.checkBoxSmooth, QtCore.SIGNAL(_fromUtf8('toggled(bool)')),
                     self.update_phase)
        self.connect(self.pushButtonLoad, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.load)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.mplWidgetMag.axes.set_visible(False)
        self.mplWidgetHolo.axes.set_visible(False)
        self.mplWidgetPhase.axes.set_visible(False)
        self.mag_data_loaded = False

    def retranslateUi(self, Form):
        ###########################################################################################
        Form.setWindowTitle(_translate("Form", "Mag Slicer", None))
        self.pushButtonLoad.setText(_translate("Form", "Laden", None))
        self.checkBoxScale.setText(_translate("Form", "Scaled", None))
        self.checkBoxLog.setText(_translate("Form", "Log", None))
        self.comboBoxSlice.setItemText(0, _translate("Form", "xy-plane", None))
        self.comboBoxSlice.setItemText(1, _translate("Form", "xz-plane", None))
        self.comboBoxSlice.setItemText(2, _translate("Form", "zy-plane", None))
        self.labelSlice.setText(_translate("Form", "Slice:", None))
        self.checkBoxSmooth.setText(_translate("Form", "Smooth", None))
        self.checkBoxAuto.setText(_translate("Form", "Auto", None))
        self.labelGain.setText(_translate("Form", "Gain:", None))
        ###########################################################################################

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
            self.projector = SimpleProjector(self.mag_data.dim, axis=self.mode)
            self.spinBoxSlice.setMaximum(length)
            self.scrollBarSlice.setMaximum(length)
            self.spinBoxSlice.setValue(int(length/2.))
            self.update_slice()
            self.phase_mapper = PhaseMapperRDFC(Kernel(self.mag_data.a, self.projector.dim_uv))
            self.phase_map = self.phase_mapper(self.projector(self.mag_data))
            self.phase_map.display_phase(axis=self.mplWidgetPhase.axes, cbar=False)
            if self.checkBoxSmooth.isChecked():
                interpolation = 'bilinear'
            else:
                interpolation = 'none'
            self.phase_map.display_holo(axis=self.mplWidgetHolo.axes, gain=gain,
                                        interpolation=interpolation)
            self.mplWidgetPhase.draw()
            self.mplWidgetHolo.draw()

    def update_slice(self):
        if self.mag_data_loaded:
            self.mag_data.quiver_plot(axis=self.mplWidgetMag.axes, proj_axis=self.mode,
                                      ax_slice=self.spinBoxSlice.value(),
                                      log=self.checkBoxLog.isChecked(),
                                      scaled=self.checkBoxScale.isChecked())
            self.mplWidgetMag.draw()

    def load(self):
        directory = os.path.join(pyramid.DIR_FILES, 'magdata')
        mag_file = QtGui.QFileDialog.getOpenFileName(self, 'Open Data File', directory,
                                                     'HDF5 files (*.hdf5)')
        self.mag_data = MagData.load_from_hdf5(mag_file)
        self.mag_data_loaded = True
        self.mplWidgetMag.axes.set_visible(True)
        self.mplWidgetHolo.axes.set_visible(True)
        self.mplWidgetPhase.axes.set_visible(True)
        self.comboBoxSlice.setCurrentIndex(0)
        self.update_phase()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = UI_MagSlicerMain()
    widget.show()
    sys.exit(app.exec_())

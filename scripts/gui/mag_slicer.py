# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mag_slicer2.ui'
#
# Created: Sun Aug 31 20:39:52 2014
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!


import sys

from PyQt4 import QtCore, QtGui

from matplotlibwidget import MatplotlibWidget

from pyramid.magdata import MagData


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
        self.setGeometry(100, 100, 650, 650)
        self.setWindowTitle('Mag Slicer')
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.pushButton = QtGui.QPushButton(Form)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout.addWidget(self.pushButton)
        self.scaleCheckBox = QtGui.QCheckBox(Form)
        self.scaleCheckBox.setChecked(True)
        self.scaleCheckBox.setTristate(False)
        self.scaleCheckBox.setObjectName(_fromUtf8("scaleCheckBox"))
        self.horizontalLayout.addWidget(self.scaleCheckBox)
        self.logCheckBox = QtGui.QCheckBox(Form)
        self.logCheckBox.setObjectName(_fromUtf8("logCheckBox"))
        self.horizontalLayout.addWidget(self.logCheckBox)
        self.comboBox = QtGui.QComboBox(Form)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.horizontalLayout.addWidget(self.comboBox)
        self.label = QtGui.QLabel(Form)
        self.label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                QtCore.Qt.AlignVCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.spinBox = QtGui.QSpinBox(Form)
        self.spinBox.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing |
                                  QtCore.Qt.AlignVCenter)
        self.spinBox.setMaximum(0)
        self.spinBox.setProperty("value", 0)
        self.spinBox.setObjectName(_fromUtf8("spinBox"))
        self.horizontalLayout.addWidget(self.spinBox)
        self.scrollBar = QtGui.QScrollBar(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scrollBar.sizePolicy().hasHeightForWidth())
        self.scrollBar.setSizePolicy(sizePolicy)
        self.scrollBar.setMaximum(0)
        self.scrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.scrollBar.setObjectName(_fromUtf8("scrollBar"))
        self.horizontalLayout.addWidget(self.scrollBar)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.mplWidget = MatplotlibWidget(Form)
        self.mplWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.mplWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mplWidget.setAutoFillBackground(False)
        self.mplWidget.setObjectName(_fromUtf8("mplWidget"))
        self.gridLayout.addWidget(self.mplWidget, 1, 0, 1, 1)
        self.retranslateUi(Form)
        self.connect(self.scrollBar, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")),
                     self.spinBox.setValue)
        self.connect(self.spinBox, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")),
                     self.scrollBar.setValue)
        self.connect(self.logCheckBox, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_plot)
        self.connect(self.scaleCheckBox, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.update_plot)
        self.connect(self.spinBox, QtCore.SIGNAL(_fromUtf8('valueChanged(int)')),
                     self.update_plot)
        self.connect(self.comboBox, QtCore.SIGNAL(_fromUtf8('currentIndexChanged(int)')),
                     self.update_ui)
        self.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8('clicked()')),
                     self.load)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Mag Slicer", None))
        self.pushButton.setText(_translate("Form", "Laden", None))
        self.scaleCheckBox.setText(_translate("Form", "Scaled", None))
        self.logCheckBox.setText(_translate("Form", "Log", None))
        self.comboBox.setItemText(0, _translate("Form", "xy-plane", None))
        self.comboBox.setItemText(1, _translate("Form", "xz-plane", None))
        self.comboBox.setItemText(2, _translate("Form", "zy-plane", None))
        self.label.setText(_translate("Form", "Slice:", None))

    def update_ui(self):
        if self.mag_data_loaded:
            mode_ind = self.comboBox.currentIndex()
            if mode_ind == 0:
                self.mode = 'z'
                length = self.mag_data.dim[0]-1
            elif mode_ind == 1:
                self.mode = 'y'
                length = self.mag_data.dim[1]-1
            else:
                self.mode = 'x'
                length = self.mag_data.dim[2]-1
            self.spinBox.setMaximum(length)
            self.scrollBar.setMaximum(length)
            self.spinBox.setValue(int(length/2.))
            self.update_plot()

    def update_plot(self):
        if self.mag_data_loaded:
            self.mag_data.quiver_plot(axis=self.mplWidget.axes, proj_axis=self.mode,
                                      ax_slice=self.spinBox.value(),
                                      log=self.logCheckBox.isChecked(),
                                      scaled=self.scaleCheckBox.isChecked(),
                                      show=False)
            self.mplWidget.draw()

    def load(self):
        mag_file = QtGui.QFileDialog.getOpenFileName(self, 'Open Data File', '',
                                                     'NetCDF files (*.nc)')
        self.mag_data = MagData.load_from_netcdf4(mag_file)
        self.mag_data_loaded = True
        self.comboBox.setCurrentIndex(0)
        self.update_ui()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = UI_MagSlicerMain()
    widget.show()
    sys.exit(app.exec_())

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'create_logo.ui'
#
# Created: Tue May 21 14:29:03 2013
#      by: PyQt4 UI code generator 4.9.5
#
# WARNING! All changes made in this file will be lost!


from PyQt4 import QtCore, QtGui
from matplotlibwidget import MatplotlibWidget


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s


class Ui_CreateLogoWidget(object):
    def setupUi(self, CreateLogoWidget):
        CreateLogoWidget.setObjectName(_fromUtf8("CreateLogoWidget"))
        CreateLogoWidget.resize(520, 492)
        self.verticalLayout_2 = QtGui.QVBoxLayout(CreateLogoWidget)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.xLabel = QtGui.QLabel(CreateLogoWidget)
        self.xLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.xLabel.setObjectName(_fromUtf8("xLabel"))
        self.horizontalLayout.addWidget(self.xLabel)
        self.xSpinBox = QtGui.QSpinBox(CreateLogoWidget)
        self.xSpinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.xSpinBox.setMaximum(512)
        self.xSpinBox.setProperty("value", 128)
        self.xSpinBox.setObjectName(_fromUtf8("xSpinBox"))
        self.horizontalLayout.addWidget(self.xSpinBox)
        self.yLabel = QtGui.QLabel(CreateLogoWidget)
        self.yLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.yLabel.setObjectName(_fromUtf8("yLabel"))
        self.horizontalLayout.addWidget(self.yLabel)
        self.ySpinBox = QtGui.QSpinBox(CreateLogoWidget)
        self.ySpinBox.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ySpinBox.setMaximum(512)
        self.ySpinBox.setProperty("value", 128)
        self.ySpinBox.setObjectName(_fromUtf8("ySpinBox"))
        self.horizontalLayout.addWidget(self.ySpinBox)
        self.logoPushButton = QtGui.QPushButton(CreateLogoWidget)
        self.logoPushButton.setObjectName(_fromUtf8("logoPushButton"))
        self.horizontalLayout.addWidget(self.logoPushButton)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.mplWidget = MatplotlibWidget(CreateLogoWidget)
        self.mplWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.mplWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mplWidget.setAutoFillBackground(False)
        self.mplWidget.setObjectName(_fromUtf8("mplWidget"))
        self.verticalLayout_2.addWidget(self.mplWidget)

        self.retranslateUi(CreateLogoWidget)
        QtCore.QMetaObject.connectSlotsByName(CreateLogoWidget)

    def retranslateUi(self, CreateLogoWidget):
        CreateLogoWidget.setWindowTitle(QtGui.QApplication.translate("CreateLogoWidget", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.xLabel.setText(QtGui.QApplication.translate("CreateLogoWidget", "X [px] :", None, QtGui.QApplication.UnicodeUTF8))
        self.yLabel.setText(QtGui.QApplication.translate("CreateLogoWidget", "Y [px] :", None, QtGui.QApplication.UnicodeUTF8))
        self.logoPushButton.setText(QtGui.QApplication.translate("CreateLogoWidget", "Create Logo", None, QtGui.QApplication.UnicodeUTF8))

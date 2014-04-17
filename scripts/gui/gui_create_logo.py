# -*- coding: utf-8 -*-
"""Create the Pyramid-Logo via GUI-Input."""


import sys
import numpy as np
from numpy import pi
from PyQt4 import QtCore, QtGui, uic

import pyramid.magcreator as mc
from pyramid.projector import SimpleProjector
from pyramid.phasemapper import PMConvolve
from pyramid.magdata import MagData

from create_logo import Ui_CreateLogoWidget


class Overlay(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        palette = QtGui.QPalette(self.palette())
        palette.setColor(palette.Background, QtCore.Qt.transparent)
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(event.rect(), QtGui.QBrush(QtGui.QColor(255, 255, 255, 127)))
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        for i in range(6):
            if (self.counter / 5) % 6 == i:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127 + (self.counter % 5)*32, 127)))
            else:
                painter.setBrush(QtGui.QBrush(QtGui.QColor(127, 127, 127)))
            painter.drawEllipse(
                self.width()/2 + 30 * np.cos(2 * pi * i / 6.0) - 10,
                self.height()/2 + 30 * np.sin(2 * pi * i / 6.0) - 10,
                20, 20)
        painter.end()

    def showEvent(self, event):
        self.timer = self.startTimer(50)
        self.counter = 0

    def hideEvent(self, event):
        self.killTimer(self.timer)

    def timerEvent(self, event):
        self.counter += 1
        self.update()


class CreateLogoWidget(QtGui.QWidget, Ui_CreateLogoWidget):

    def __init__(self):
        # Call parent constructor
        QtGui.QWidget.__init__(self)
        self.setupUi(self)
        self.ui = uic.loadUi('create_logo.ui')
#        self.setCentralWidget(self.ui)
        # Connect Widgets with locally defined functions:
        self.connect(self.logoPushButton, QtCore.SIGNAL('clicked()'), self.buttonPushed)
        # Create overlay to indicate busy state:
        self.overlay = Overlay(self)
#        self.ui.overlay = Overlay(self.ui)
        self.overlay.hide()
        # Show Widget:
        self.show()

        self.workerThread = WorkerThread()

    def buttonPushed(self):
        self.overlay.show()
        x = self.xSpinBox.value()
        y = self.ySpinBox.value()
        create_logo((1, y, x), self.mplWidget.axes)
        self.mplWidget.draw()
#        self.workerThread.start()
        self.overlay.hide()

    def resizeEvent(self, event):
        self.overlay.resize(event.size())
        event.accept()


class WorkerThread(QtCore.QThread):

    def __init__(self, parent=None):
        QtCore.QThread.__init__(self)

    def run(self):
        x = self.xSpinBox.value()
        y = self.ySpinBox.value()
        create_logo((1, y, x), self.mplWidget.axes)
        self.mplWidget.draw()


def create_logo(dim, axis):
    '''Calculate and display the Pyramid-Logo.'''
    # Input parameters:
    a = 10.0  # in nm
    phi = -pi/2  # in rad
    density = 10
    # Create magnetic shape:
    mag_shape = np.zeros(dim)
    x = range(dim[2])
    y = range(dim[1])
    xx, yy = np.meshgrid(x, y)
    bottom = (yy >= 0.25*dim[1])
    left = (yy <= 0.75/0.5 * dim[1]/dim[2] * xx)
    right = np.fliplr(left)
    mag_shape[0, ...] = np.logical_and(np.logical_and(left, right), bottom)
    # Create magnetic data, project it, get the phase map and display the holography image:
    mag_data = MagData(a, mc.create_mag_dist_homog(mag_shape, phi))
    phase_map = PMConvolve(a, SimpleProjector(dim))(mag_data)
    phase_map.display_holo(density, 'PYRAMID - LOGO', interpolation='bilinear',
                           axis=axis, show=False)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    gui = CreateLogoWidget()
    sys.exit(app.exec_())

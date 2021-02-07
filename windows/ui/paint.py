# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'paint.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("PaintPredict")
        MainWindow.resize(400, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(5, 307, 54, 14))
        font = QtGui.QFont()
        font.setPointSize(10)
        font = QtGui.QFont()
        font.setPointSize(10)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 23))
        self.menubar.setObjectName("menubar")
        self.menuclear = QtWidgets.QMenu(self.menubar)
        self.menuclear.setObjectName("menuclear")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionclear = QtWidgets.QAction(MainWindow)
        self.actionclear.setObjectName("actionclear")
        self.actionsave = QtWidgets.QAction(MainWindow)
        self.actionsave.setObjectName("actionsave")
        self.menuclear.addAction(self.actionclear)
        self.menuclear.addAction(self.actionsave)
        self.menubar.addAction(self.menuclear.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("PaintPredict", "PaintPredict"))
        self.menuclear.setTitle(_translate("MainWindow", "file"))
        self.actionclear.setText(_translate("MainWindow", "clear"))
        self.actionsave.setText(_translate("MainWindow", "save"))

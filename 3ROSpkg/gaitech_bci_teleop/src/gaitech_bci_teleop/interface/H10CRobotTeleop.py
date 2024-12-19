# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './Ui/H10CRobotTeleop.ui'
#
# Created by: PyQt4 UI code generator 4.12.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_H10CRobotTeleop(object):
    def setupUi(self, H10CRobotTeleop):
        H10CRobotTeleop.setObjectName(_fromUtf8("H10CRobotTeleop"))
        H10CRobotTeleop.resize(708, 673)
        self.centralwidget = QtGui.QWidget(H10CRobotTeleop)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tabTeleop = QtGui.QWidget()
        self.tabTeleop.setStyleSheet(_fromUtf8("background: black"))
        self.tabTeleop.setObjectName(_fromUtf8("tabTeleop"))
        self.gridLayout_2 = QtGui.QGridLayout(self.tabTeleop)
        self.gridLayout_2.setMargin(0)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.widget_up = FlickeringImageWidget(self.tabTeleop)
        self.widget_up.setMinimumSize(QtCore.QSize(300, 120))
        self.widget_up.setMaximumSize(QtCore.QSize(16777215, 160))
        self.widget_up.setObjectName(_fromUtf8("widget_up"))
        self.horizontalLayout.addWidget(self.widget_up)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.widget_left = FlickeringImageWidget(self.tabTeleop)
        self.widget_left.setMinimumSize(QtCore.QSize(120, 200))
        self.widget_left.setMaximumSize(QtCore.QSize(160, 16777215))
        self.widget_left.setObjectName(_fromUtf8("widget_left"))
        self.horizontalLayout_3.addWidget(self.widget_left)
        self.camView = QtGui.QGraphicsView(self.tabTeleop)
        self.camView.setMinimumSize(QtCore.QSize(320, 240))
        self.camView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.camView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.camView.setObjectName(_fromUtf8("camView"))
        self.horizontalLayout_3.addWidget(self.camView)
        self.widget_right = FlickeringImageWidget(self.tabTeleop)
        self.widget_right.setMinimumSize(QtCore.QSize(120, 200))
        self.widget_right.setMaximumSize(QtCore.QSize(160, 16777215))
        self.widget_right.setObjectName(_fromUtf8("widget_right"))
        self.horizontalLayout_3.addWidget(self.widget_right)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.widget_down = FlickeringImageWidget(self.tabTeleop)
        self.widget_down.setMinimumSize(QtCore.QSize(300, 120))
        self.widget_down.setMaximumSize(QtCore.QSize(16777215, 160))
        self.widget_down.setObjectName(_fromUtf8("widget_down"))
        self.horizontalLayout_2.addWidget(self.widget_down)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem4)
        self.btnStop = QtGui.QPushButton(self.tabTeleop)
        self.btnStop.setMinimumSize(QtCore.QSize(0, 50))
        self.btnStop.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.btnStop.setFont(font)
        self.btnStop.setStyleSheet(_fromUtf8("color:red; background: gray;"))
        self.btnStop.setObjectName(_fromUtf8("btnStop"))
        self.horizontalLayout_4.addWidget(self.btnStop)
        self.btnStart = QtGui.QPushButton(self.tabTeleop)
        self.btnStart.setMinimumSize(QtCore.QSize(0, 50))
        self.btnStart.setMaximumSize(QtCore.QSize(16777215, 60))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setStyleSheet(_fromUtf8("color:green;\n"
"background: gray;"))
        self.btnStart.setObjectName(_fromUtf8("btnStart"))
        self.horizontalLayout_4.addWidget(self.btnStart)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem5)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 3, 0, 1, 1)
        self.tabWidget.addTab(self.tabTeleop, _fromUtf8(""))
        self.tabSettings = QtGui.QWidget()
        self.tabSettings.setObjectName(_fromUtf8("tabSettings"))
        self.gridLayout_7 = QtGui.QGridLayout(self.tabSettings)
        self.gridLayout_7.setMargin(0)
        self.gridLayout_7.setObjectName(_fromUtf8("gridLayout_7"))
        self.gbSSVEPSettings = QtGui.QGroupBox(self.tabSettings)
        self.gbSSVEPSettings.setObjectName(_fromUtf8("gbSSVEPSettings"))
        self.gridLayout_3 = QtGui.QGridLayout(self.gbSSVEPSettings)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_2 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_2.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_3.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_7 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_7.setMaximumSize(QtCore.QSize(25, 16777215))
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.gridLayout_3.addWidget(self.label_7, 3, 3, 1, 1)
        self.label_4 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_4.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 4, 0, 1, 1)
        self.label_15 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_15.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.gridLayout_3.addWidget(self.label_15, 3, 6, 1, 1)
        self.spfUp = QtGui.QDoubleSpinBox(self.gbSSVEPSettings)
        self.spfUp.setMinimum(5.0)
        self.spfUp.setMaximum(25.0)
        self.spfUp.setSingleStep(0.5)
        self.spfUp.setProperty("value", 5.0)
        self.spfUp.setObjectName(_fromUtf8("spfUp"))
        self.gridLayout_3.addWidget(self.spfUp, 1, 1, 1, 1)
        self.spfRight = QtGui.QDoubleSpinBox(self.gbSSVEPSettings)
        self.spfRight.setMinimum(5.0)
        self.spfRight.setMaximum(25.0)
        self.spfRight.setSingleStep(0.5)
        self.spfRight.setProperty("value", 11.0)
        self.spfRight.setObjectName(_fromUtf8("spfRight"))
        self.gridLayout_3.addWidget(self.spfRight, 3, 1, 1, 1)
        self.spfLeft = QtGui.QDoubleSpinBox(self.gbSSVEPSettings)
        self.spfLeft.setMinimum(5.0)
        self.spfLeft.setMaximum(25.0)
        self.spfLeft.setSingleStep(0.5)
        self.spfLeft.setProperty("value", 8.0)
        self.spfLeft.setObjectName(_fromUtf8("spfLeft"))
        self.gridLayout_3.addWidget(self.spfLeft, 2, 1, 1, 1)
        self.label_14 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_14.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.gridLayout_3.addWidget(self.label_14, 2, 6, 1, 1)
        self.label_8 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_8.setMaximumSize(QtCore.QSize(25, 16777215))
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.gridLayout_3.addWidget(self.label_8, 4, 3, 1, 1)
        self.label_3 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_3.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.cbBgStop = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbBgStop.setObjectName(_fromUtf8("cbBgStop"))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.cbBgStop.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbBgStop, 4, 5, 1, 1)
        self.label_5 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_5.setMaximumSize(QtCore.QSize(25, 16777215))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout_3.addWidget(self.label_5, 1, 3, 1, 1)
        self.label_6 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_6.setMaximumSize(QtCore.QSize(25, 16777215))
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout_3.addWidget(self.label_6, 2, 3, 1, 1)
        self.label_16 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_16.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.gridLayout_3.addWidget(self.label_16, 4, 6, 1, 1)
        self.label_11 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_11.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.gridLayout_3.addWidget(self.label_11, 3, 4, 1, 1)
        self.spfStop = QtGui.QDoubleSpinBox(self.gbSSVEPSettings)
        self.spfStop.setMinimum(5.0)
        self.spfStop.setMaximum(25.0)
        self.spfStop.setSingleStep(0.5)
        self.spfStop.setProperty("value", 13.0)
        self.spfStop.setObjectName(_fromUtf8("spfStop"))
        self.gridLayout_3.addWidget(self.spfStop, 4, 1, 1, 1)
        self.label_10 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_10.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.gridLayout_3.addWidget(self.label_10, 2, 4, 1, 1)
        self.label_9 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_9.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.gridLayout_3.addWidget(self.label_9, 1, 6, 1, 1)
        self.label_12 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_12.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.gridLayout_3.addWidget(self.label_12, 4, 4, 1, 1)
        self.label_13 = QtGui.QLabel(self.gbSSVEPSettings)
        self.label_13.setMaximumSize(QtCore.QSize(85, 16777215))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.gridLayout_3.addWidget(self.label_13, 1, 4, 1, 1)
        self.label = QtGui.QLabel(self.gbSSVEPSettings)
        self.label.setMaximumSize(QtCore.QSize(50, 16777215))
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_3.addWidget(self.label, 2, 0, 1, 1)
        self.cbBgUp = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbBgUp.setObjectName(_fromUtf8("cbBgUp"))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.cbBgUp.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbBgUp, 1, 5, 1, 1)
        self.cbBgLeft = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbBgLeft.setObjectName(_fromUtf8("cbBgLeft"))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.cbBgLeft.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbBgLeft, 2, 5, 1, 1)
        self.cbBgRight = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbBgRight.setObjectName(_fromUtf8("cbBgRight"))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.cbBgRight.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbBgRight, 3, 5, 1, 1)
        self.cbUp = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbUp.setObjectName(_fromUtf8("cbUp"))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.cbUp.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbUp, 1, 7, 1, 1)
        self.cbLeft = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbLeft.setObjectName(_fromUtf8("cbLeft"))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.cbLeft.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbLeft, 2, 7, 1, 1)
        self.cbRight = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbRight.setObjectName(_fromUtf8("cbRight"))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.cbRight.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbRight, 3, 7, 1, 1)
        self.cbStop = QtGui.QComboBox(self.gbSSVEPSettings)
        self.cbStop.setObjectName(_fromUtf8("cbStop"))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.cbStop.addItem(_fromUtf8(""))
        self.gridLayout_3.addWidget(self.cbStop, 4, 7, 1, 1)
        self.label_8.raise_()
        self.label_4.raise_()
        self.label_2.raise_()
        self.spfStop.raise_()
        self.spfRight.raise_()
        self.spfLeft.raise_()
        self.label.raise_()
        self.label_3.raise_()
        self.spfUp.raise_()
        self.label_6.raise_()
        self.label_5.raise_()
        self.label_7.raise_()
        self.cbBgUp.raise_()
        self.label_10.raise_()
        self.label_11.raise_()
        self.label_12.raise_()
        self.cbBgLeft.raise_()
        self.cbBgRight.raise_()
        self.cbBgStop.raise_()
        self.label_9.raise_()
        self.label_13.raise_()
        self.label_14.raise_()
        self.label_15.raise_()
        self.label_16.raise_()
        self.cbUp.raise_()
        self.cbLeft.raise_()
        self.cbRight.raise_()
        self.cbStop.raise_()
        self.gridLayout_7.addWidget(self.gbSSVEPSettings, 0, 0, 1, 1)
        self.gbROSSettings = QtGui.QGroupBox(self.tabSettings)
        self.gbROSSettings.setObjectName(_fromUtf8("gbROSSettings"))
        self.gridLayout_6 = QtGui.QGridLayout(self.gbROSSettings)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.label_18 = QtGui.QLabel(self.gbROSSettings)
        self.label_18.setMaximumSize(QtCore.QSize(200, 16777215))
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.gridLayout_6.addWidget(self.label_18, 0, 0, 1, 1)
        self.cbCamera = QtGui.QComboBox(self.gbROSSettings)
        self.cbCamera.setObjectName(_fromUtf8("cbCamera"))
        self.gridLayout_6.addWidget(self.cbCamera, 0, 1, 1, 1)
        self.tbReloadCamera = QtGui.QToolButton(self.gbROSSettings)
        self.tbReloadCamera.setObjectName(_fromUtf8("tbReloadCamera"))
        self.gridLayout_6.addWidget(self.tbReloadCamera, 0, 2, 1, 1)
        self.label_17 = QtGui.QLabel(self.gbROSSettings)
        self.label_17.setMaximumSize(QtCore.QSize(200, 16777215))
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.gridLayout_6.addWidget(self.label_17, 1, 0, 1, 1)
        self.cvCVel = QtGui.QComboBox(self.gbROSSettings)
        self.cvCVel.setObjectName(_fromUtf8("cvCVel"))
        self.gridLayout_6.addWidget(self.cvCVel, 1, 1, 1, 1)
        self.tbReloadCVel = QtGui.QToolButton(self.gbROSSettings)
        self.tbReloadCVel.setObjectName(_fromUtf8("tbReloadCVel"))
        self.gridLayout_6.addWidget(self.tbReloadCVel, 1, 2, 1, 1)
        self.gridLayout_7.addWidget(self.gbROSSettings, 1, 0, 1, 1)
        spacerItem6 = QtGui.QSpacerItem(20, 194, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout_7.addItem(spacerItem6, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tabSettings, _fromUtf8(""))
        self.tabH10CSettings = QtGui.QWidget()
        self.tabH10CSettings.setObjectName(_fromUtf8("tabH10CSettings"))
        self.tabWidget.addTab(self.tabH10CSettings, _fromUtf8(""))
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        H10CRobotTeleop.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(H10CRobotTeleop)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 708, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menu_File = QtGui.QMenu(self.menubar)
        self.menu_File.setObjectName(_fromUtf8("menu_File"))
        self.menu_Help = QtGui.QMenu(self.menubar)
        self.menu_Help.setObjectName(_fromUtf8("menu_Help"))
        H10CRobotTeleop.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(H10CRobotTeleop)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        H10CRobotTeleop.setStatusBar(self.statusbar)
        self.actionE_xit = QtGui.QAction(H10CRobotTeleop)
        self.actionE_xit.setObjectName(_fromUtf8("actionE_xit"))
        self.action_About = QtGui.QAction(H10CRobotTeleop)
        self.action_About.setObjectName(_fromUtf8("action_About"))
        self.action_How_to = QtGui.QAction(H10CRobotTeleop)
        self.action_How_to.setObjectName(_fromUtf8("action_How_to"))
        self.actionStart_Teleop = QtGui.QAction(H10CRobotTeleop)
        self.actionStart_Teleop.setObjectName(_fromUtf8("actionStart_Teleop"))
        self.actionView_H10_Data = QtGui.QAction(H10CRobotTeleop)
        self.actionView_H10_Data.setObjectName(_fromUtf8("actionView_H10_Data"))
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.actionE_xit)
        self.menu_Help.addAction(self.action_About)
        self.menu_Help.addAction(self.action_How_to)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menu_Help.menuAction())

        self.retranslateUi(H10CRobotTeleop)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(H10CRobotTeleop)

    def retranslateUi(self, H10CRobotTeleop):
        H10CRobotTeleop.setWindowTitle(_translate("H10CRobotTeleop", "BCI Teleop", None))
        self.btnStop.setToolTip(_translate("H10CRobotTeleop", "Emergency Stop", None))
        self.btnStop.setText(_translate("H10CRobotTeleop", "Stop Robot", None))
        self.btnStart.setToolTip(_translate("H10CRobotTeleop", "Send command velocities to robot", None))
        self.btnStart.setText(_translate("H10CRobotTeleop", "Operate Robot", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTeleop), _translate("H10CRobotTeleop", "Move Robot", None))
        self.gbSSVEPSettings.setTitle(_translate("H10CRobotTeleop", "SSVEP Settings", None))
        self.label_2.setText(_translate("H10CRobotTeleop", "Right", None))
        self.label_7.setText(_translate("H10CRobotTeleop", "Hz", None))
        self.label_4.setText(_translate("H10CRobotTeleop", "Stop", None))
        self.label_15.setText(_translate("H10CRobotTeleop", "Overlay", None))
        self.label_14.setText(_translate("H10CRobotTeleop", "Overlay", None))
        self.label_8.setText(_translate("H10CRobotTeleop", "Hz", None))
        self.label_3.setText(_translate("H10CRobotTeleop", "Up", None))
        self.cbBgStop.setItemText(0, _translate("H10CRobotTeleop", "White", None))
        self.cbBgStop.setItemText(1, _translate("H10CRobotTeleop", "Transparent", None))
        self.cbBgStop.setItemText(2, _translate("H10CRobotTeleop", "Black", None))
        self.cbBgStop.setItemText(3, _translate("H10CRobotTeleop", "Red", None))
        self.cbBgStop.setItemText(4, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbBgStop.setItemText(5, _translate("H10CRobotTeleop", "Green", None))
        self.cbBgStop.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.label_5.setText(_translate("H10CRobotTeleop", "Hz", None))
        self.label_6.setText(_translate("H10CRobotTeleop", "Hz", None))
        self.label_16.setText(_translate("H10CRobotTeleop", "Overlay", None))
        self.label_11.setText(_translate("H10CRobotTeleop", "Background", None))
        self.label_10.setText(_translate("H10CRobotTeleop", "Background", None))
        self.label_9.setText(_translate("H10CRobotTeleop", "Overlay", None))
        self.label_12.setText(_translate("H10CRobotTeleop", "Background", None))
        self.label_13.setText(_translate("H10CRobotTeleop", "Background", None))
        self.label.setText(_translate("H10CRobotTeleop", "Left", None))
        self.cbBgUp.setItemText(0, _translate("H10CRobotTeleop", "White", None))
        self.cbBgUp.setItemText(1, _translate("H10CRobotTeleop", "Transparent", None))
        self.cbBgUp.setItemText(2, _translate("H10CRobotTeleop", "Black", None))
        self.cbBgUp.setItemText(3, _translate("H10CRobotTeleop", "Red", None))
        self.cbBgUp.setItemText(4, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbBgUp.setItemText(5, _translate("H10CRobotTeleop", "Green", None))
        self.cbBgUp.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbBgLeft.setItemText(0, _translate("H10CRobotTeleop", "White", None))
        self.cbBgLeft.setItemText(1, _translate("H10CRobotTeleop", "Transparent", None))
        self.cbBgLeft.setItemText(2, _translate("H10CRobotTeleop", "Black", None))
        self.cbBgLeft.setItemText(3, _translate("H10CRobotTeleop", "Red", None))
        self.cbBgLeft.setItemText(4, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbBgLeft.setItemText(5, _translate("H10CRobotTeleop", "Green", None))
        self.cbBgLeft.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbBgRight.setItemText(0, _translate("H10CRobotTeleop", "White", None))
        self.cbBgRight.setItemText(1, _translate("H10CRobotTeleop", "Transparent", None))
        self.cbBgRight.setItemText(2, _translate("H10CRobotTeleop", "Black", None))
        self.cbBgRight.setItemText(3, _translate("H10CRobotTeleop", "Red", None))
        self.cbBgRight.setItemText(4, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbBgRight.setItemText(5, _translate("H10CRobotTeleop", "Green", None))
        self.cbBgRight.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbUp.setItemText(0, _translate("H10CRobotTeleop", "Black", None))
        self.cbUp.setItemText(1, _translate("H10CRobotTeleop", "White", None))
        self.cbUp.setItemText(2, _translate("H10CRobotTeleop", "Green", None))
        self.cbUp.setItemText(3, _translate("H10CRobotTeleop", "Dark Green", None))
        self.cbUp.setItemText(4, _translate("H10CRobotTeleop", "Red", None))
        self.cbUp.setItemText(5, _translate("H10CRobotTeleop", "Dark Red", None))
        self.cbUp.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbUp.setItemText(7, _translate("H10CRobotTeleop", "Dark Blue", None))
        self.cbUp.setItemText(8, _translate("H10CRobotTeleop", "Magenta", None))
        self.cbUp.setItemText(9, _translate("H10CRobotTeleop", "Dark Magenta", None))
        self.cbUp.setItemText(10, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbUp.setItemText(11, _translate("H10CRobotTeleop", "Dark Yellow", None))
        self.cbUp.setItemText(12, _translate("H10CRobotTeleop", "Cyan", None))
        self.cbUp.setItemText(13, _translate("H10CRobotTeleop", "Dark Cyan", None))
        self.cbLeft.setItemText(0, _translate("H10CRobotTeleop", "Black", None))
        self.cbLeft.setItemText(1, _translate("H10CRobotTeleop", "White", None))
        self.cbLeft.setItemText(2, _translate("H10CRobotTeleop", "Green", None))
        self.cbLeft.setItemText(3, _translate("H10CRobotTeleop", "Dark Green", None))
        self.cbLeft.setItemText(4, _translate("H10CRobotTeleop", "Red", None))
        self.cbLeft.setItemText(5, _translate("H10CRobotTeleop", "Dark Red", None))
        self.cbLeft.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbLeft.setItemText(7, _translate("H10CRobotTeleop", "Dark Blue", None))
        self.cbLeft.setItemText(8, _translate("H10CRobotTeleop", "Magenta", None))
        self.cbLeft.setItemText(9, _translate("H10CRobotTeleop", "Dark Magenta", None))
        self.cbLeft.setItemText(10, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbLeft.setItemText(11, _translate("H10CRobotTeleop", "Dark Yellow", None))
        self.cbLeft.setItemText(12, _translate("H10CRobotTeleop", "Cyan", None))
        self.cbLeft.setItemText(13, _translate("H10CRobotTeleop", "Dark Cyan", None))
        self.cbRight.setItemText(0, _translate("H10CRobotTeleop", "Black", None))
        self.cbRight.setItemText(1, _translate("H10CRobotTeleop", "White", None))
        self.cbRight.setItemText(2, _translate("H10CRobotTeleop", "Green", None))
        self.cbRight.setItemText(3, _translate("H10CRobotTeleop", "Dark Green", None))
        self.cbRight.setItemText(4, _translate("H10CRobotTeleop", "Red", None))
        self.cbRight.setItemText(5, _translate("H10CRobotTeleop", "Dark Red", None))
        self.cbRight.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbRight.setItemText(7, _translate("H10CRobotTeleop", "Dark Blue", None))
        self.cbRight.setItemText(8, _translate("H10CRobotTeleop", "Magenta", None))
        self.cbRight.setItemText(9, _translate("H10CRobotTeleop", "Dark Magenta", None))
        self.cbRight.setItemText(10, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbRight.setItemText(11, _translate("H10CRobotTeleop", "Dark Yellow", None))
        self.cbRight.setItemText(12, _translate("H10CRobotTeleop", "Cyan", None))
        self.cbRight.setItemText(13, _translate("H10CRobotTeleop", "Dark Cyan", None))
        self.cbStop.setItemText(0, _translate("H10CRobotTeleop", "Black", None))
        self.cbStop.setItemText(1, _translate("H10CRobotTeleop", "White", None))
        self.cbStop.setItemText(2, _translate("H10CRobotTeleop", "Green", None))
        self.cbStop.setItemText(3, _translate("H10CRobotTeleop", "Dark Green", None))
        self.cbStop.setItemText(4, _translate("H10CRobotTeleop", "Red", None))
        self.cbStop.setItemText(5, _translate("H10CRobotTeleop", "Dark Red", None))
        self.cbStop.setItemText(6, _translate("H10CRobotTeleop", "Blue", None))
        self.cbStop.setItemText(7, _translate("H10CRobotTeleop", "Dark Blue", None))
        self.cbStop.setItemText(8, _translate("H10CRobotTeleop", "Magenta", None))
        self.cbStop.setItemText(9, _translate("H10CRobotTeleop", "Dark Magenta", None))
        self.cbStop.setItemText(10, _translate("H10CRobotTeleop", "Yellow", None))
        self.cbStop.setItemText(11, _translate("H10CRobotTeleop", "Dark Yellow", None))
        self.cbStop.setItemText(12, _translate("H10CRobotTeleop", "Cyan", None))
        self.cbStop.setItemText(13, _translate("H10CRobotTeleop", "Dark Cyan", None))
        self.gbROSSettings.setTitle(_translate("H10CRobotTeleop", "ROS Settings", None))
        self.label_18.setText(_translate("H10CRobotTeleop", "Robot Camera Feed", None))
        self.cbCamera.setToolTip(_translate("H10CRobotTeleop", "<html><head/><body><p>Select Camera feed to display</p></body></html>", None))
        self.tbReloadCamera.setToolTip(_translate("H10CRobotTeleop", "<html><head/><body><p>Search for Image ROS Topics</p></body></html>", None))
        self.tbReloadCamera.setText(_translate("H10CRobotTeleop", "R", None))
        self.label_17.setText(_translate("H10CRobotTeleop", "Robot Command Velocity", None))
        self.cvCVel.setToolTip(_translate("H10CRobotTeleop", "<html><head/><body><p>Select cmd_vel to publish velocity commands to</p></body></html>", None))
        self.tbReloadCVel.setToolTip(_translate("H10CRobotTeleop", "<html><head/><body><p>Search for command velocity ROS Topics</p></body></html>", None))
        self.tbReloadCVel.setText(_translate("H10CRobotTeleop", "R", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabSettings), _translate("H10CRobotTeleop", "Configure", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabH10CSettings), _translate("H10CRobotTeleop", "H10C Device Settings", None))
        self.menu_File.setTitle(_translate("H10CRobotTeleop", "&File", None))
        self.menu_Help.setTitle(_translate("H10CRobotTeleop", "&Help", None))
        self.actionE_xit.setText(_translate("H10CRobotTeleop", "E&xit", None))
        self.action_About.setText(_translate("H10CRobotTeleop", "&About", None))
        self.action_How_to.setText(_translate("H10CRobotTeleop", "How To ... ?", None))
        self.actionStart_Teleop.setText(_translate("H10CRobotTeleop", "Start Teleop", None))
        self.actionView_H10_Data.setText(_translate("H10CRobotTeleop", "View H10 Data", None))

from gaitech_bci_teleop.pyqt.FlickeringImage import FlickeringImageWidget
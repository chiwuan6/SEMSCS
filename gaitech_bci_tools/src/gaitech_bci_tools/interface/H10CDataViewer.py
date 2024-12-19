# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file './Ui/H10CDataViewer.ui'
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

class Ui_H10CDataViewer(object):
    def setupUi(self, H10CDataViewer):
        H10CDataViewer.setObjectName(_fromUtf8("H10CDataViewer"))
        H10CDataViewer.resize(977, 643)
        H10CDataViewer.setMinimumSize(QtCore.QSize(0, 640))
        self.gridLayout_4 = QtGui.QGridLayout(H10CDataViewer)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gbDeviceStatus = QtGui.QGroupBox(H10CDataViewer)
        self.gbDeviceStatus.setMaximumSize(QtCore.QSize(350, 16777215))
        self.gbDeviceStatus.setObjectName(_fromUtf8("gbDeviceStatus"))
        self.gridLayout = QtGui.QGridLayout(self.gbDeviceStatus)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.lblDName = QtGui.QLabel(self.gbDeviceStatus)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lblDName.setFont(font)
        self.lblDName.setObjectName(_fromUtf8("lblDName"))
        self.gridLayout.addWidget(self.lblDName, 0, 1, 1, 1)
        self.lblDSig = QtGui.QLabel(self.gbDeviceStatus)
        self.lblDSig.setObjectName(_fromUtf8("lblDSig"))
        self.gridLayout.addWidget(self.lblDSig, 0, 2, 1, 1)
        self.label_4 = QtGui.QLabel(self.gbDeviceStatus)
        self.label_4.setMaximumSize(QtCore.QSize(60, 16777215))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.lbldconstant = QtGui.QLabel(self.gbDeviceStatus)
        self.lbldconstant.setMaximumSize(QtCore.QSize(60, 16777215))
        self.lbldconstant.setObjectName(_fromUtf8("lbldconstant"))
        self.gridLayout.addWidget(self.lbldconstant, 0, 0, 1, 1)
        self.lblDMode = QtGui.QLabel(self.gbDeviceStatus)
        font = QtGui.QFont()
        font.setItalic(True)
        self.lblDMode.setFont(font)
        self.lblDMode.setObjectName(_fromUtf8("lblDMode"))
        self.gridLayout.addWidget(self.lblDMode, 1, 1, 1, 2)
        self.verticalLayout.addWidget(self.gbDeviceStatus)
        self.gbChannels = QtGui.QGroupBox(H10CDataViewer)
        self.gbChannels.setMaximumSize(QtCore.QSize(350, 16777215))
        self.gbChannels.setObjectName(_fromUtf8("gbChannels"))
        self.gridLayout_2 = QtGui.QGridLayout(self.gbChannels)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.verticalLayout.addWidget(self.gbChannels)
        self.label_2 = QtGui.QLabel(H10CDataViewer)
        self.label_2.setMaximumSize(QtCore.QSize(350, 16777215))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout.addWidget(self.label_2)
        self.twMarkers = QtGui.QTableWidget(H10CDataViewer)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.twMarkers.sizePolicy().hasHeightForWidth())
        self.twMarkers.setSizePolicy(sizePolicy)
        self.twMarkers.setMaximumSize(QtCore.QSize(350, 16777215))
        self.twMarkers.setStyleSheet(_fromUtf8("background:\"transparent\";"))
        self.twMarkers.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.twMarkers.setAlternatingRowColors(True)
        self.twMarkers.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
        self.twMarkers.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.twMarkers.setShowGrid(False)
        self.twMarkers.setObjectName(_fromUtf8("twMarkers"))
        self.twMarkers.setColumnCount(3)
        self.twMarkers.setRowCount(0)
        item = QtGui.QTableWidgetItem()
        self.twMarkers.setHorizontalHeaderItem(0, item)
        item = QtGui.QTableWidgetItem()
        self.twMarkers.setHorizontalHeaderItem(1, item)
        item = QtGui.QTableWidgetItem()
        self.twMarkers.setHorizontalHeaderItem(2, item)
        self.twMarkers.horizontalHeader().setDefaultSectionSize(70)
        self.twMarkers.horizontalHeader().setMinimumSectionSize(30)
        self.twMarkers.verticalHeader().setVisible(False)
        self.verticalLayout.addWidget(self.twMarkers)
        self.gridLayout_4.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.tbtnStart = QtGui.QToolButton(H10CDataViewer)
        self.tbtnStart.setMinimumSize(QtCore.QSize(32, 32))
        self.tbtnStart.setObjectName(_fromUtf8("tbtnStart"))
        self.horizontalLayout.addWidget(self.tbtnStart)
        self.tbtnEnd = QtGui.QToolButton(H10CDataViewer)
        self.tbtnEnd.setMinimumSize(QtCore.QSize(32, 32))
        self.tbtnEnd.setObjectName(_fromUtf8("tbtnEnd"))
        self.horizontalLayout.addWidget(self.tbtnEnd)
        self.tbtnAddMarker = QtGui.QToolButton(H10CDataViewer)
        self.tbtnAddMarker.setMinimumSize(QtCore.QSize(32, 32))
        self.tbtnAddMarker.setCheckable(True)
        self.tbtnAddMarker.setObjectName(_fromUtf8("tbtnAddMarker"))
        self.horizontalLayout.addWidget(self.tbtnAddMarker)
        self.tbtnEdit = QtGui.QToolButton(H10CDataViewer)
        self.tbtnEdit.setMinimumSize(QtCore.QSize(32, 32))
        self.tbtnEdit.setCheckable(True)
        self.tbtnEdit.setObjectName(_fromUtf8("tbtnEdit"))
        self.horizontalLayout.addWidget(self.tbtnEdit)
        self.tbtnFullScreen = QtGui.QToolButton(H10CDataViewer)
        self.tbtnFullScreen.setMinimumSize(QtCore.QSize(32, 32))
        self.tbtnFullScreen.setCheckable(True)
        self.tbtnFullScreen.setObjectName(_fromUtf8("tbtnFullScreen"))
        self.horizontalLayout.addWidget(self.tbtnFullScreen)
        spacerItem = QtGui.QSpacerItem(267, 13, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.plotter = GraphicsLayoutWidget(H10CDataViewer)
        self.plotter.setObjectName(_fromUtf8("plotter"))
        self.verticalLayout_2.addWidget(self.plotter)
        self.gbDataRec = QtGui.QGroupBox(H10CDataViewer)
        self.gbDataRec.setObjectName(_fromUtf8("gbDataRec"))
        self.gridLayout_3 = QtGui.QGridLayout(self.gbDataRec)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.btnClear = QtGui.QPushButton(self.gbDataRec)
        self.btnClear.setMinimumSize(QtCore.QSize(0, 40))
        self.btnClear.setObjectName(_fromUtf8("btnClear"))
        self.gridLayout_3.addWidget(self.btnClear, 0, 1, 1, 1)
        self.btnRecord = QtGui.QPushButton(self.gbDataRec)
        self.btnRecord.setMinimumSize(QtCore.QSize(0, 40))
        self.btnRecord.setObjectName(_fromUtf8("btnRecord"))
        self.gridLayout_3.addWidget(self.btnRecord, 0, 2, 1, 1)
        self.btnSave = QtGui.QPushButton(self.gbDataRec)
        self.btnSave.setMinimumSize(QtCore.QSize(0, 40))
        self.btnSave.setObjectName(_fromUtf8("btnSave"))
        self.gridLayout_3.addWidget(self.btnSave, 0, 3, 1, 1)
        self.btnLoad = QtGui.QPushButton(self.gbDataRec)
        self.btnLoad.setMinimumSize(QtCore.QSize(0, 40))
        self.btnLoad.setObjectName(_fromUtf8("btnLoad"))
        self.gridLayout_3.addWidget(self.btnLoad, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.gbDataRec)
        self.gridLayout_4.addLayout(self.verticalLayout_2, 0, 1, 1, 1)

        self.retranslateUi(H10CDataViewer)
        QtCore.QMetaObject.connectSlotsByName(H10CDataViewer)

    def retranslateUi(self, H10CDataViewer):
        H10CDataViewer.setWindowTitle(_translate("H10CDataViewer", "Live Data Viewer", None))
        self.gbDeviceStatus.setTitle(_translate("H10CDataViewer", "Device Status", None))
        self.lblDName.setText(_translate("H10CDataViewer", "Test Device", None))
        self.lblDSig.setText(_translate("H10CDataViewer", "0.5", None))
        self.label_4.setText(_translate("H10CDataViewer", "Mode", None))
        self.lbldconstant.setText(_translate("H10CDataViewer", "Device", None))
        self.lblDMode.setText(_translate("H10CDataViewer", "Common Reference", None))
        self.gbChannels.setTitle(_translate("H10CDataViewer", "Display Channels", None))
        self.label_2.setText(_translate("H10CDataViewer", "User Markers In Data", None))
        item = self.twMarkers.horizontalHeaderItem(0)
        item.setText(_translate("H10CDataViewer", "Marker", None))
        item = self.twMarkers.horizontalHeaderItem(1)
        item.setText(_translate("H10CDataViewer", "Time", None))
        self.tbtnStart.setToolTip(_translate("H10CDataViewer", "Goto start of data", None))
        self.tbtnStart.setText(_translate("H10CDataViewer", "...", None))
        self.tbtnEnd.setToolTip(_translate("H10CDataViewer", "Goto end of data", None))
        self.tbtnEnd.setText(_translate("H10CDataViewer", "...", None))
        self.tbtnAddMarker.setToolTip(_translate("H10CDataViewer", "Add marker in data", None))
        self.tbtnAddMarker.setText(_translate("H10CDataViewer", "A", None))
        self.tbtnEdit.setToolTip(_translate("H10CDataViewer", "Edit marker in data", None))
        self.tbtnEdit.setText(_translate("H10CDataViewer", "S", None))
        self.tbtnFullScreen.setToolTip(_translate("H10CDataViewer", "Show fullscreen", None))
        self.tbtnFullScreen.setText(_translate("H10CDataViewer", "F", None))
        self.gbDataRec.setTitle(_translate("H10CDataViewer", "Data Recording Options", None))
        self.btnClear.setToolTip(_translate("H10CDataViewer", "Clear data from display and memory", None))
        self.btnClear.setText(_translate("H10CDataViewer", "Clear Data", None))
        self.btnRecord.setToolTip(_translate("H10CDataViewer", "Start streaming from device", None))
        self.btnRecord.setText(_translate("H10CDataViewer", "Start Recording", None))
        self.btnSave.setToolTip(_translate("H10CDataViewer", "Save data to file", None))
        self.btnSave.setText(_translate("H10CDataViewer", "Save Data", None))
        self.btnLoad.setToolTip(_translate("H10CDataViewer", "Load data from file", None))
        self.btnLoad.setText(_translate("H10CDataViewer", "Load Data", None))

from pyqtgraph import GraphicsLayoutWidget

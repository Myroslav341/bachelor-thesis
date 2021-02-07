import random
from string import ascii_lowercase
from config import Config
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from rows_manager import RowsManager
from windows.ui import PaintUI


class AppWindow(QtWidgets.QMainWindow, PaintUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.lastPoint = None

        self.click_point = None
        self.top = (0, 0)
        self.bottom = (100000, 100000)
        self.left = (0, 0)
        self.right = (100000, 100000)

        self.drawing = False
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.actionsave.triggered.connect(self.save)
        self.actionclear.triggered.connect(self.clear)

        # self.vectorise = Vectorise()
        self.rows_manager = RowsManager()

    def clear(self):
        self.image.fill(Qt.white)
        self.rows_manager.clear()
        self.update()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        return

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            self.click_point = event.pos()
            self.top = (0, 0)
            self.bottom = (100000, 100000)
            self.left = (100000, 100000)
            self.right = (0, 0)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

            self.rows_manager.add_new_hatch(
                self.left,
                self.right,
                self.bottom,
                self.top,
            )

            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
            vector = self.rows_manager.vectors[-1]

            painter.drawLine(*vector.paint_format())

            self.update()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 4, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()

            if self.lastPoint.y() > self.top[1]:
                self.top = (self.lastPoint.x(), self.lastPoint.y())
            if self.lastPoint.y() < self.bottom[1]:
                self.bottom = (self.lastPoint.x(), self.lastPoint.y())
            if self.lastPoint.x() < self.left[0]:
                self.left = (self.lastPoint.x(), self.lastPoint.y())
            if self.lastPoint.x() > self.right[0]:
                self.right = (self.lastPoint.x(), self.lastPoint.y())

            self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_A:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.blue, 4, Qt.SolidLine))

            self.rows_manager.process()
            for row in self.rows_manager.rows:
                if not row.vectors_list:
                    continue
                x, y, w, z = row.final_vector.paint_format()
                painter.drawLine(x, y, w, z)
                self.update()

                painter.setPen(QPen(Qt.green, 4, Qt.SolidLine))
                painter.drawLine(x, y + row.bottom, w, z + row.bottom)
                painter.drawLine(x, y - row.top, w, z - row.top)

            painter.setPen(QPen(Qt.yellow, 4, Qt.SolidLine))
            painter.drawLine(*self.top, self.top[0] + 1, self.top[1] + 1)
            painter.drawLine(*self.bottom, self.bottom[0] + 1, self.bottom[1] + 1)

            self.update()

    def save(self, path=None):
        if path is not None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image", "",
                "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) "
            )

            if path == "":
                return
            self.image.save(path)
        else:
            name = ''.join([random.choice(ascii_lowercase) for _ in range(8)]) + '.png'
            self.image.save(Config.BASE_DIR + '/tests/' + name)
            return Config.BASE_DIR + '/tests/' + name
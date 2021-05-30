import json
import random
from string import ascii_lowercase

from cnn import load_image_and_predict
from config import Config
from PyQt5.QtGui import QPainter, QImage, QPen, QStaticText
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog

from lib import get_dist_between_dots
from lib.project import save_rows_data
from rows_manager import RowsManager
from voronoi_tessellation import VoronoiTesselation, Cell
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
        self.voronoi_tesselation = None

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

            # vector = self.rows_manager.vectors[-1]
            # painter.drawLine(*vector.paint_format())
            # self.update()

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

            self.rows_manager.process()
            for row in self.rows_manager.rows:
                if not row.vectors_list:
                    continue
                x, y, w, z = row.final_vector.paint_format()
                painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
                painter.drawEllipse(x - 2, y - 2, 4, 4)
                painter.drawLine(x, y, w, z)
                self.update()

                painter.setPen(QPen(Qt.green, 4, Qt.SolidLine))
                painter.drawLine(x, y + row.bottom, w, z + row.bottom)
                painter.drawLine(x, y - row.top, w, z - row.top)

            painter.setPen(QPen(Qt.yellow, 4, Qt.SolidLine))

            # painter.drawLine(*self.top, self.top[0] + 1, self.top[1] + 1)
            # painter.drawLine(*self.bottom, self.bottom[0] + 1, self.bottom[1] + 1)

            self.update()

        if e.key() == Qt.Key_S:
            painter = QPainter(self.image)

            self.voronoi_tesselation = VoronoiTesselation(self.rows_manager.dots)
            self.voronoi_tesselation.process()

            for cell in Cell.objects_dict.values():
                for neighbor in cell.neighbors:
                    if not neighbor.is_finite:
                        continue
                    # painter.drawLine(
                    #     int(neighbor.dot_start[0]), int(neighbor.dot_start[1]),
                    #     int(neighbor.dot_end[0]), int(neighbor.dot_end[1])
                    # )
            # self.update()

        if e.key() == Qt.Key_D:
            painter = QPainter(self.image)

            self.voronoi_tesselation = VoronoiTesselation(self.rows_manager.dots)
            self.voronoi_tesselation.process()

            # for cell in Cell.objects_dict.values():
            #     for neighbor in cell.neighbors:
            #         try:
            #             painter.drawLine(
            #                 int(neighbor.dot_start[0]), int(neighbor.dot_start[1]),
            #                 int(neighbor.dot_end[0]), int(neighbor.dot_end[1])
            #             )
            #         except OverflowError:
            #             continue

            # self.update()

        if e.key() == Qt.Key_F:
            painter = QPainter(self.image)

            self.rows_manager.add_voronoi_cells()

            for row in self.rows_manager.rows:
                for cell in row.voronoi_cells:
                    st = QStaticText(f"{row.index}_{cell.id}")
                    st.setTextWidth(20)
                    st.setTextFormat(Qt.PlainText)
                    # painter.drawStaticText(int(cell.center[0]), int(cell.center[1]), st)

            # self.update()

        if e.key() == Qt.Key_G:
            self.rows_manager.predict_rows_neighbor_cells_2()

            for row in self.rows_manager.rows:
                neighbors = []
                for cell in row.voronoi_cells:
                    if cell.come_with:
                        neighbors.append(cell.id)
                    elif not cell.come_with and not neighbors:
                        print(f"({cell.id})", end=" ")
                    elif not cell.come_with and neighbors:
                        neighbors.append(cell.id)
                        print(f"({', '.join([str(n) for n in neighbors])})", end=" ")
                        neighbors = []
                    else:
                        print(f"({cell.id})", end=" ")
                        neighbors = []
                print()

        if e.key() == Qt.Key_H:
            self.image.save(Config.BASE_DIR + "/" + "image.png")
            load_image_and_predict()

            for row in self.rows_manager.rows:
                neighbors = []
                for cell in row.voronoi_cells:
                    if cell.come_with:
                        neighbors.append(cell.predicted_number)
                    elif not cell.come_with and not neighbors:
                        print(f"{cell.predicted_number}", end=" ")
                    elif not cell.come_with and neighbors:
                        neighbors.append(cell.predicted_number)
                        print(f"{''.join([str(n) for n in neighbors])}", end=" ")
                        neighbors = []
                    else:
                        print(f"{cell.predicted_number}", end=" ")
                        neighbors = []
                print()

        if e.key() == Qt.Key_Q:
            save_rows_data()

        if e.key() == Qt.Key_Z:
            self.save_as_test_data()

        if e.key() == Qt.Key_Space:
            self.save_as_a_dataset_digit()

    def save_as_test_data(self):
        row = self.rows_manager.rows[0]

        height = row.top + row.bottom
        print(f"row top = {row.top}, bottom = {row.bottom}, height = {height}")

        k = 1 / height

        dots = [row.vectors_list[0].begin] + [vector.end for vector in row.vectors_list]
        dots = [
            (
                (dot[0] - dots[0][0]) * k,
                (dot[1] - dots[0][1]) * k
            )
            for dot in dots
        ]
        average_cell_width = sum([cell.digit_width * k for cell in row.voronoi_cells]) / len(row.voronoi_cells)

        with open('dots_dist_data.json', 'r') as openfile:
            existing_info = json.load(openfile)

        record_id = len(existing_info["rows"])

        new_record = {"id": record_id, "dots": [], "k": k, "average_cell_width": average_cell_width}

        dots_record = []
        for i, dot in enumerate(dots):
            if i == len(dots) - 1:
                break
            dot_record = {
                "d_1": dot,
                "d_2": dots[i + 1],
                "dist": get_dist_between_dots(dot, dots[i + 1]),
                "relative": get_dist_between_dots(dot, dots[i + 1]) / average_cell_width,
                "is_come_together": None,
            }
            dots_record.append(dot_record)

        new_record["dots"] = dots_record

        existing_info["rows"].append(new_record)

        json_object = json.dumps(existing_info, indent=4)

        with open("dots_dist_data.json", "w") as outfile:
            outfile.write(json_object)

    def save_as_a_dataset_digit(self):
        digit = "9"
        path = Config.BASE_DIR + f"/dataset_2/{digit}/{random.randint(1000000, 9999999)}.png"

        self.image.save(path)

        self.image.fill(Qt.white)
        self.update()

    def save(self, path=None):
        if path is not None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Image",
                "",
                "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ",
            )

            if path == "":
                return
            self.image.save(path)
        else:
            name = "".join([random.choice(ascii_lowercase) for _ in range(8)]) + ".png"
            self.image.save(Config.BASE_DIR + "/tests/" + name)
            return Config.BASE_DIR + "/tests/" + name

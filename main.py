import sys

from PyQt5.QtWidgets import QApplication

from cnn import CNN
from windows import AppWindow
from lib.project import start_project


def start_window():
    cnn = CNN()
    cnn.load('4.h5')

    start_project()

    app = QApplication(sys.argv)

    window = AppWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    start_window()

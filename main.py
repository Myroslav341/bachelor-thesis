import sys

from PyQt5.QtWidgets import QApplication

from windows import AppWindow


def start_window():
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    start_window()

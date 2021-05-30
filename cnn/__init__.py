import cv2

from cnn.cnn_class import CNN
from config import Config
from rows_manager import RowsManager


def load_image_and_predict():
    cnn = CNN()
    rows_manager = RowsManager()

    img_data = cv2.imread(Config.BASE_DIR + "/" + "image.png", cv2.IMREAD_GRAYSCALE)

    for row in rows_manager.rows:
        for cell in row.voronoi_cells:
            h_top = cell.center[1] - row.top - 5
            h_bottom = cell.center[1] + row.bottom + 5
            w_left = cell.center[0] - cell.digit_width / 2 - 5
            w_right = cell.center[0] + cell.digit_width / 2 + 5

            digit = img_data[int(h_top):int(h_bottom), int(w_left):int(w_right)]

            cell.predicted_number = cnn.predict_concrete_number(digit)

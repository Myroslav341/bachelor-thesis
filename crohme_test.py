import os
from time import time
from xml.etree.ElementTree import ElementTree
import cv2
import numpy as np

from cnn import load_image_and_predict, CNN
from config import Config
from rows_manager import RowsManager
from voronoi_tessellation import VoronoiTesselation


def go_though_all():
    for f in os.listdir(f"{Config.BASE_DIR}/nnnn/"):
        if f == ".DS_Store":
            continue
        print(f)
        a = ElementTree(file=f"{Config.BASE_DIR}/nnnn/{f}")

        img = np.zeros((1000, 1000, 3), np.uint8)

        for x in a.getroot():
            if not x.tag.endswith('trace'):
                continue
            content: str = x.text[1:-1]

            dots = content.split(', ')

            for i in range(len(dots) - 1):
                img = cv2.line(
                    img,
                    (int(dots[i].split(' ')[0]), int(dots[i].split(' ')[1])),
                    (int(dots[i + 1].split(' ')[0]), int(dots[i + 1].split(' ')[1])),
                    (255, 0, 0), 5
                )

        cv2.imwrite('color_img.jpg', img)
        cv2.imshow("image", img)
        cv2.waitKey()


def go_though_original():
    for _ in range(3, 123):

        a = ElementTree(
            file=f"{Config.BASE_DIR}/CROHME2014_data/MatricesTest/RIT_MatrixTest_2014_{_}.inkml"
        )

        try:
            img = np.zeros((1000, 1000, 3), np.uint8)

            for x in a.getroot():
                if not x.tag.endswith('trace'):
                    continue
                content: str = x.text[1:-1]

                dots = content.split(', ')

                for i in range(len(dots) - 1):
                    img = cv2.line(
                        img,
                        (int(dots[i].split(' ')[0]), int(dots[i].split(' ')[1])),
                        (int(dots[i + 1].split(' ')[0]), int(dots[i + 1].split(' ')[1])),
                        (255, 0, 0), 5
                    )

            cv2.imwrite('color_img.jpg', img)
            cv2.imshow("image", img)
            cv2.waitKey()

            q = input()
            if q == "s":
                os.system(f"cp {Config.BASE_DIR}/CROHME2014_data/MatricesTest/RIT_MatrixTest_2014_{_}.inkml {Config.BASE_DIR}/nnnn/")

        except Exception as e:
            print(e)
            continue


def show_concrete():
    f = "RIT_MatrixTest_2014_58.inkml"

    a = ElementTree(file=f"{Config.BASE_DIR}/nnnn/{f}")

    img = np.zeros((1000, 1000, 3), np.uint8)

    for x in a.getroot():
        if not x.tag.endswith('trace'):
            continue
        content: str = x.text[1:-1]

        dots = content.split(', ')

        for i in range(len(dots) - 1):
            img = cv2.line(
                img,
                (int(dots[i].split(' ')[0]), int(dots[i].split(' ')[1])),
                (int(dots[i + 1].split(' ')[0]), int(dots[i + 1].split(' ')[1])),
                (255, 0, 0), 5
            )

        cv2.imwrite('color_img.jpg', img)
        cv2.imshow("image", img)
        cv2.waitKey()


rez = {
    "18": [[1, 1, 3], [0, 0, 4]],
    "46": [[1, 1], [1, 3]],
    "59": [[3, 0, 0], [1, 3, 0], [0, 0, 3]],
    "87": [[3, 1], [1, 3]],
    "26": [[1, 2], [1, 0]],
    "45": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "58": [[0, 1, 0], [0, 0, 1]],
    "96": [[1, 2, 3], [3, 0, 3], [1, 4, 5]],
    "56": [[6, 0, 0, 0], [1, 6, 0, 0], [0, 1, 6, 0], [0, 0, 0, 6]],
    "89": [[1, 3, 7], [2, 3, 8], [0, 1, 2], [4, 0, 4]],
    "4": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "15": [[1, 3], [4, 12]],
    "50": [[6, 2, 4, 4], [2, 2, 2, 2], [4, 2, 5, 1], [4, 2, 1, 5]],
    "90": [[2, 1, 0, 0], ],
    "52": [[1], [0]],
    "3": [[1, 4, 3], [1, 0, 3], [1, 8, 9]],
    "12": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "16": [[1, 2], [10, 4]],
    "32": [[2, 4], [9, 4]],
}


def test():
    n = 16
    f = f"RIT_MatrixTest_2014_{n}.inkml"

    cnn = CNN()
    cnn.load('4.h5')

    rows_manager = RowsManager()

    a = ElementTree(file=f"{Config.BASE_DIR}/nnnn/{f}")

    img = np.zeros((1000, 1000, 3), np.uint8)

    for x in a.getroot():
        if not x.tag.endswith('trace'):
            continue

        left, top, right, bottom = (10000000, 0), (0, 10000000), (0, 0), (0, 0)

        content: str = x.text[1:-1]

        dots = content.split(', ')

        for dot in dots:
            x, y = int(dot.split(' ')[0]), int(dot.split(' ')[1])
            if x < left[0]:
                left = (x, y)
            elif x > right[0]:
                right = (x, y)
            if y < top[1]:
                top = (x, y)
            elif y > bottom[1]:
                bottom = (x, y)

        for i in range(len(dots) - 1):
            img = cv2.line(
                img,
                (int(dots[i].split(' ')[0]), int(dots[i].split(' ')[1])),
                (int(dots[i + 1].split(' ')[0]), int(dots[i + 1].split(' ')[1])),
                (255, 255, 255), 5
            )

        rows_manager.add_new_hatch(left, right, top, bottom)

    img = np.invert(img)
    cv2.imwrite('image.png', img)

    t = time()

    rows_manager.process()

    voronoi_tesselation = VoronoiTesselation(rows_manager.dots)
    voronoi_tesselation.process()

    rows_manager.add_voronoi_cells()

    rows_manager.predict_rows_neighbor_cells_2()

    load_image_and_predict()

    print(time() - t)

    for row in rows_manager.rows:
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


if __name__ == '__main__':
    # show_concrete()
    test()

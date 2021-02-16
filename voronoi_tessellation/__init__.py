from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from scipy.spatial import Voronoi

from lib import Dot
from lib.decorators import singleton


@dataclass
class CellNeighbor:
    cell_id: int
    neighbor_id: int

    dot_start: Dot
    dot_end: Optional[Dot] = None

    is_finite: bool = field(default=None)

    def __post_init__(self):
        self.is_finite = bool(self.dot_end)

        if not self.is_finite:
            points = [Cell.objects_dict[self.cell_id].center, Cell.objects_dict[self.neighbor_id].center]

            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords)[0]

            a, b, c = m, -1, -c
            a, b = b, -a
            c = -(a * self.dot_start[0] + b * self.dot_start[1])

            if b < 0:
                a_ = a * -1
            else:
                a_ = a

            if a_ >= 0:
                x = max(points[0][0], points[1][0])
            else:
                x = min(points[0][0], points[1][0])

            y = -(a*x + c) / b
            self.dot_end = (x, y)


@dataclass
class Cell:
    id: int

    center: Dot
    is_closed: bool = False
    neighbors: List[CellNeighbor] = field(default_factory=list)

    def __post_init__(self):
        if not hasattr(Cell, "objects_dict"):
            Cell.objects_dict = {}

        if self.id not in Cell.objects_dict:
            Cell.objects_dict[self.id] = self


@singleton
class VoronoiTesselation:
    def __init__(self, dots: List[Dot]):
        self.dots: List[Dot] = dots
        self.dots_np = np.array([list(dot) for dot in dots])
        self.voronoi_result = None
        self.voronoi_vertices = None

    def process(self):
        # delete all cells created before
        Cell.objects_dict = {}

        # create new cells around the dots
        for i, dot in enumerate(self.dots):
            Cell(i, center=dot)

        # processing voronoi tesselation
        self.voronoi_result = Voronoi(self.dots_np)
        self.voronoi_vertices = [tuple(vertex) for vertex in self.voronoi_result.vertices]

        for cell_dots, divider_dots in self.voronoi_result.ridge_dict.items():
            # cell dots: dots that are the middle of cells
            # divider_dots: dots that form the divider between these dots

            cell_1 = Cell.objects_dict[cell_dots[0]]
            cell_2 = Cell.objects_dict[cell_dots[1]]

            if divider_dots[0] == -1 or divider_dots[1] == -1:
                # if the divider between these dots is infinite
                dot_id = divider_dots[1] if divider_dots[0] == -1 else divider_dots[0]
                cell_1.neighbors.append(CellNeighbor(cell_1.id, cell_dots[1], self.voronoi_vertices[dot_id]))
                cell_2.neighbors.append(CellNeighbor(cell_2.id, cell_dots[0], self.voronoi_vertices[dot_id]))
            else:
                # if the divider between these dots is finite
                cell_1.neighbors.append(
                    CellNeighbor(
                        cell_1.id,
                        cell_dots[1],
                        self.voronoi_vertices[divider_dots[0]],
                        self.voronoi_vertices[divider_dots[1]]
                    )
                )
                cell_2.neighbors.append(
                    CellNeighbor(
                        cell_2.id,
                        cell_dots[0],
                        self.voronoi_vertices[divider_dots[0]],
                        self.voronoi_vertices[divider_dots[1]]
                    )
                )


if __name__ == '__main__':
    tesselation = VoronoiTesselation([(0, 0), (0, 1), (1, 1), (1, 0)])
    tesselation.process()
    assert True

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from lib import Dot, Vector, get_middle_dot, get_angle_between_vectors, get_dist_to_straight, get_triangle_square
from lib.decorators import singleton

logging.basicConfig(level=logging.INFO)


@dataclass
class MatrixRow:
    index: int
    vectors_list: List[Vector] = field(default_factory=list)
    cos_list: List[float] = field(default_factory=list)
    top: float = field(default=0)
    bottom: float = field(default=0)
    _top_dot: Optional[Dot] = field(default=None)
    _bottom_dot: Optional[Dot] = field(default=None)

    @property
    def final_vector(self) -> Vector:
        return Vector(self.vectors_list[0].begin, self.vectors_list[-1].end)

    def add_hatch(self, vector: Vector, top: Dot, bottom: Dot):
        self.vectors_list.append(vector)
        self._update_top(top)
        self._update_bottom(bottom)

    def is_dot_inside_row(self, dot: Dot) -> bool:
        x1 = self.final_vector.begin[0], self.final_vector.begin[1] + self.bottom
        x2 = self.final_vector.end[0], self.final_vector.end[1] + self.bottom
        x3 = self.final_vector.end[0], self.final_vector.end[1] - self.top
        x4 = self.final_vector.begin[0], self.final_vector.begin[1] - self.top
        middle = (x1[0] + x2[0]) / 2, (x2[1] + x3[1]) / 2
        x = [x1, x2, x3, x4]
        s1, s2 = 0, 0
        for i in range(4):
            dot1 = x[i]
            dot2 = x[(i + 1) % 4]
            s1 += get_triangle_square(dot1, dot2, dot)
            s2 += get_triangle_square(dot1, dot2, middle)
        return abs(s1 - s2) < 0.001

    def _update_top(self, new_top: Dot):
        if not self.vectors_list:
            return
        d = get_dist_to_straight(new_top, self.final_vector.begin, self.final_vector.end)
        if self.top < d:
            self._top_dot = new_top
            self.top = d
        else:
            d = get_dist_to_straight(self._top_dot, self.final_vector.begin, self.final_vector.end)
            self.top = d

    def _update_bottom(self, new_bottom: Dot):
        if not self.vectors_list:
            return
        d = get_dist_to_straight(new_bottom, self.final_vector.begin, self.final_vector.end)
        if self.bottom < d:
            self._bottom_dot = new_bottom
            self.bottom = d
        else:
            d = get_dist_to_straight(self._bottom_dot, self.final_vector.begin, self.final_vector.end)
            self.bottom = d


@singleton
class RowsManager:
    def __init__(self):
        self.vectors: List[Vector] = []
        self.cos_list: List[float] = []
        self.top_list: List[Dot] = []
        self.bottom_list: List[Dot] = []

        self.rows: List[MatrixRow] = []

    def add_new_hatch(self, left: Dot, right: Dot, top: Dot, bottom: Dot):
        middle_dot = ((left[0] + right[0]) / 2, (top[1] + bottom[1]) / 2)

        if len(self.vectors) == 0:
            self.vectors.append(Vector(middle_dot, middle_dot))
            self.top_list.append(top)
            self.bottom_list.append(bottom)
            return

        prev_vector_end = self.vectors[-1].end
        self.vectors.append(Vector(prev_vector_end, middle_dot))
        self.cos_list.append(get_angle_between_vectors(self.vectors[-1], Vector((0, 0), (1, 0))))
        self.top_list.append(top)
        self.bottom_list.append(bottom)

    def process(self) -> List[MatrixRow]:
        self.rows = []
        row_id = 0

        i = -1
        added_to_existing_row = False
        for cos, vector, top, bottom in zip(self.cos_list, self.vectors[1:], self.top_list, self.bottom_list):
            i += 1
            added_to_existing_row = False
            logging.info(f"processing hatch {i}")
            if cos <= 70:
                if not self.rows:
                    row_id += 1
                    self.rows.append(MatrixRow(row_id))
                self.rows[-1].add_hatch(vector, top, bottom)
            else:
                # checking existing rows
                for row in self.rows:
                    is_inside_row = row.is_dot_inside_row(vector.end)
                    if is_inside_row:
                        added_to_existing_row = True
                        break
                if not added_to_existing_row:
                    if not self.rows:
                        # creating two rows, one for row 1 and second for row 2
                        # example: row1: 1, row2: 2
                        row_id += 1
                        self.rows.append(MatrixRow(row_id))
                        self.rows[-1]._top_dot = top  # noqa
                        self.rows[-1]._bottom_dot = bottom  # noqa
                        self.rows[-1].vectors_list.append(Vector(vector.begin, vector.begin))
                        row_id += 1
                        self.rows.append(MatrixRow(row_id))
                        self.rows[-1].vectors_list.append(Vector(vector.end, vector.end))
                    else:
                        row_id += 1
                        self.rows.append(MatrixRow(row_id))
                        self.rows[-1]._top_dot = top  # noqa
                        self.rows[-1]._bottom_dot = bottom  # noqa
        if not added_to_existing_row:
            self.rows[-1]._update_top(self.top_list[-1])  # noqa
            self.rows[-1]._update_bottom(self.bottom_list[-1])  # noqa

        return self.rows

    def clear(self):
        self.rows = []
        self.vectors = []
        self.top_list = []
        self.bottom_list = []
        self.cos_list = []

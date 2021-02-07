import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import config
from lib import (
    Dot,
    Vector,
    get_angle_between_vectors,
    get_dist_to_straight,
    get_triangle_square,
)
from lib.decorators import singleton


@dataclass
class MatrixRow:
    """Instance for the processed row of the matrix."""

    index: int
    vectors_list: List[Vector] = field(default_factory=list)
    cos_list: List[float] = field(default_factory=list)
    top: float = field(default=0)
    bottom: float = field(default=0)
    _top_dot: Optional[Dot] = field(default=None)
    _bottom_dot: Optional[Dot] = field(default=None)

    def __post_init__(self):
        self._is_vector_extended = False

    @property
    def final_vector(self) -> Vector:
        """Beginning of the first vector -> end of the last vector. Basically the vector of the row."""
        return Vector(self.vectors_list[0].begin, self.vectors_list[-1].end)

    def add_hatch(self, vector: Vector, top: Dot, bottom: Dot):
        """Save new hatch to the row."""
        if len(self.vectors_list) == 1 and not self._is_vector_extended:
            self.vectors_list[0].end = vector.end
            self._is_vector_extended = True
        else:
            self.vectors_list.append(vector)
        self._update_top(top)
        self._update_bottom(bottom)

    def is_dot_inside_row(self, dot: Dot) -> bool:
        """Used for detecting is some random hatch belongs to this row."""
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
        """Process new top dot and update _top_dot and top params if needed."""
        if not self.vectors_list:
            return
        d = get_dist_to_straight(
            new_top, self.final_vector.begin, self.final_vector.end
        )
        if self.top < d:
            self._top_dot = new_top
            self.top = d
        else:
            # new dot may change the angle of the final vector, so need to double-check top distance
            d = get_dist_to_straight(
                self._top_dot, self.final_vector.begin, self.final_vector.end
            )
            self.top = d

    def _update_bottom(self, new_bottom: Dot):
        """Process new top dot and update _bottom_dot and bottom params if needed."""
        if not self.vectors_list:
            return
        d = get_dist_to_straight(
            new_bottom, self.final_vector.begin, self.final_vector.end
        )
        if self.bottom < d:
            self._bottom_dot = new_bottom
            self.bottom = d
        else:
            # new dot may change the angle of the final vector, so need to double-check bottom distance
            d = get_dist_to_straight(
                self._bottom_dot, self.final_vector.begin, self.final_vector.end
            )
            self.bottom = d


@singleton
class RowsManager:
    """Class for managing rows of the the matrix based on live data."""

    # todo need to ignore horizontal lines, they are braking the row construction because angle might be ~90˚

    def __init__(self):
        # list of all vectors that will be processed
        self.vectors: List[Vector] = []
        # list of cos between (1,0) and each vector
        self.cos_list: List[float] = []
        # list of top dots
        self.top_list: List[Dot] = []
        # list of bottom dots
        self.bottom_list: List[Dot] = []

        # list of processed rows
        self.rows: List[MatrixRow] = []

    def add_new_hatch(self, left: Dot, right: Dot, top: Dot, bottom: Dot):
        """Save new user hatch. Create a vector for it, save top and bottom."""

        # the middle of the hatch, based on this dots vectors will be created
        middle_dot = ((left[0] + right[0]) / 2, (top[1] + bottom[1]) / 2)

        if len(self.vectors) == 0:
            self.vectors.append(Vector(middle_dot, middle_dot))
            self.top_list.append(top)
            self.bottom_list.append(bottom)
            return

        # previous vector end + new middle dot = new vector to save
        prev_vector_end = self.vectors[-1].end
        self.vectors.append(Vector(prev_vector_end, middle_dot))
        self.cos_list.append(
            get_angle_between_vectors(self.vectors[-1], Vector((0, 0), (1, 0)))
        )
        self.top_list.append(top)
        self.bottom_list.append(bottom)

    def process(self) -> List[MatrixRow]:
        """Processing saved data to distinguish matrix rows."""
        self.rows: List[MatrixRow] = [MatrixRow(0)]
        self.rows[0].vectors_list = [self.vectors[0]]

        row_id = 0

        # identifier for the number of current hatch, used for logs
        i = -1

        # bool flag is the last hatch was ignored because found a row for it
        is_added_to_existing_row = False

        for cos, vector, top, bottom in zip(
            self.cos_list, self.vectors[1:], self.top_list, self.bottom_list
        ):
            i += 1
            is_added_to_existing_row = False
            logging.info(f"processing hatch {i}")

            if cos <= config.NEW_ROW_ANGLE:
                # add this hatch to current row because angle is smaller than config
                self._add_hatch_to_current_row(vector=vector, top=top, bottom=bottom)
            else:
                # ignore this hatch if row will be found or create a new row
                is_added_to_existing_row, row_id = self._process_hatch_not_current_row(
                    row_id,
                    vector=vector,
                    top=top,
                    bottom=bottom
                )

        # updating the last row with top and bottom of the last digit
        if not is_added_to_existing_row:
            self.rows[-1]._update_top(self.top_list[-1])  # noqa
            self.rows[-1]._update_bottom(self.bottom_list[-1])  # noqa

        logging.info(f"was processed {i + 1} hatches, and was distinguished {len(self.rows)} rows.")

        return self.rows

    def clear(self):
        """Clear all saved data."""
        self.rows = []
        self.vectors = []
        self.top_list = []
        self.bottom_list = []
        self.cos_list = []

    def _add_hatch_to_current_row(self, *, vector: Vector, top: Dot, bottom: Dot):
        """Save hutch to the current row."""
        self.rows[-1].add_hatch(vector, top, bottom)

    def _process_hatch_not_current_row(
        self,
        row_id: int,
        *,
        vector: Vector,
        top: Dot,
        bottom: Dot,
    ) -> Tuple[bool, int]:
        is_added_to_existing_row = False

        # checking existing rows, if match will be found hatch will be ignored
        for row in self.rows:
            if row.top and row.bottom and row.is_dot_inside_row(vector.end):
                is_added_to_existing_row = True
                break

        # if no row was found for this hatch, create new row
        if not is_added_to_existing_row:
            row_id += 1
            # updating top and bottom of the previous row before creating a new one
            self.rows[-1]._update_top(top)  # noqa
            self.rows[-1]._update_bottom(bottom)  # noqa
            # creating a new row and init vector with a dot
            self.rows.append(MatrixRow(row_id))
            self.rows[-1].vectors_list = [Vector(vector.end, vector.end)]

        return is_added_to_existing_row, row_id

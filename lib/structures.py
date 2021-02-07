import math
from typing import Tuple

Dot = Tuple[float, float]
DotInt = Tuple[int, int]


class Vector:
    """Representation of math vector."""

    def __init__(self, begin: Dot, end: Dot):
        self.begin = begin
        self.end = end

    def __str__(self) -> str:
        return f"[{self.begin}, {self.end}]"

    @property
    def len(self) -> float:
        return math.sqrt(
            (self.end[0] - self.begin[0]) ** 2 + (self.end[1] - self.begin[1]) ** 2
        )

    def paint_format(self) -> Tuple[int, int, int, int]:
        from lib import dot_to_int

        begin = dot_to_int(self.begin)
        end = dot_to_int(self.end)

        return begin[0], begin[1], end[0], end[1]

    def __mul__(self, other: "Vector") -> float:
        v1 = (self.end[0] - self.begin[0], self.end[1] - self.begin[1])
        v2 = (other.end[0] - other.begin[0], other.end[1] - other.begin[1])

        return v1[0] * v2[0] + v1[1] * v2[1]

    def __rmul__(self, other: "Vector"):
        return other.__mul__(self)

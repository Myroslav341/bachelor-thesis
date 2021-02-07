import math
from typing import Tuple

from lib.structures import Dot, DotInt, Vector


def get_middle_dot(*dots: Dot) -> Dot:
    middle_dot = (
        sum([dot[0] for dot in dots]) / len(dots),
        sum([dot[1] for dot in dots]) / len(dots),
    )

    return middle_dot


def dot_to_int(dot: Dot) -> DotInt:
    return int(dot[0]), int(dot[1])


def get_angle_between_vectors(vector_1: Vector, vector_2: Vector) -> float:
    cos = get_cos_between_vectors(vector_1, vector_2)
    acos = math.acos(cos) * 180 / math.pi

    return acos


def get_cos_between_vectors(vector_1: Vector, vector_2: Vector) -> float:
    a = vector_1 * vector_2
    b = vector_1.len * vector_2.len

    return a / b


def get_dist_to_straight(d: Dot, a: Dot, b: Dot):
    a_c, b_c, c_c = get_straight_coefficients(a, b)

    return math.fabs(d[0] * a_c + d[1] * b_c + c_c) / math.sqrt(a_c ** 2 + b_c ** 2)


def get_dist_between_dots(a: Dot, b: Dot):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def get_straight_coefficients(a: Dot, b: Dot) -> Tuple[float, float, float]:
    if a[0] == b[0]:
        a_ = 1
        b_ = 0
        c_ = a[0]
        return a_, b_, c_

    if a[1] == b[1]:
        a_ = 0
        b_ = 1
        c_ = a[1]
        return a_, b_, c_

    a_ = 1 / (a[0] - b[0])
    b_ = -1 / (a[1] - b[1])
    c_ = -b[0] / (a[0] - b[0]) + b[1] / (a[1] - b[1])
    return a_, b_, c_


def get_triangle_square(a, b, c):
    d1, d2, d3 = (
        get_dist_between_dots(a, b),
        get_dist_between_dots(b, c),
        get_dist_between_dots(c, a),
    )
    s = (d1 + d2 + d3) / 2
    return math.sqrt(s * (s - d1) * (s - d2) * (s - d3))

import math

from lib import get_angle_between_vectors, Vector, get_straight_coefficients

# a1, a2 = (1, 1), (4, 1)
# b1, b2 = (1, 3), (4, 4)
# x = (2, 2)
#
# alpha = get_angle_between_vectors(Vector(a1, a2), Vector(b1, b2)) // 2
# print(f"alpha = {alpha}")
#
#
# a = math.tan(alpha * math.pi / 180)
# b = x[1] - a * x[0]
#
# print(f"y = {a:.2f} * x + {b:.2f}, {x[1] == a * x[0] + b}")
#
# v1 = (-a, 1)
# v2 = (1, a)
#
# print(f"{v1} ~> {v2}")
#
# b = -(v2[0] * x[0] + v2[1] * x[1])
#
# print(f"final: a = {v2[0]}, b = {v2[1]}, c = {b}")
#
# aa, ab, ad = get_straight_coefficients(a1, a2)
# ba, bb, bd = get_straight_coefficients(b1, b2)
#
# y1 = (aa * b - v2[0] * ad) / (aa * v2[1] - v2[0] * ab)
# x1 = (-v2[1] * y1 - b) / v2[0]
#
# print(x1, y1)


if __name__ == '__main__':
    print(get_angle_between_vectors(Vector((0, 0), (-1, 0)), Vector((0, 0), (1, 0))))

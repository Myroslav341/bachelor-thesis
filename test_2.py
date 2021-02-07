from lib import get_triangle_square

x1 = (1, 1)
x2 = (4, 1)
x3 = (1, 3)
x4 = (4, 4)
dot = (2, 3)
middle = (x1[0] + x2[0]) / 2, (x2[1] + x3[1]) / 2
x = [x1, x2, x3, x4]
s1, s2 = 0, 0
for i in range(4):
    dot1 = x[i]
    dot2 = x[(i + 1) % 4]
    s1 += get_triangle_square(dot1, dot2, dot)
    s2 += get_triangle_square(dot1, dot2, middle)

print(s1, s2)
print(abs(s1 - s2) < 0.001)

from math import acos, copysign, cos, pi, sqrt
from typing import Tuple


def cubic_roots(a: float, b: float,
                c: float, d: float) -> tuple[complex, complex, complex]:
    tmp = a
    a = b/tmp
    b = c/tmp
    c = d/tmp
    Q = (a**2 - 3*b)/9
    R = (2*a**3 - 9*a*b + 27*c)/54
    Q3 = Q**3
    R2 = R**2
    if R2 < Q3:
        th = acos(R/sqrt(Q3))
        x1 = -2*sqrt(Q)*cos(th/3) - a/3
        x2 = -2*sqrt(Q)*cos((th + 2*pi)/3) - a/3
        x3 = -2*sqrt(Q)*cos((th - 2*pi)/3) - a/3
    else:
        A = -copysign((abs(R) + sqrt(R2 - Q3))**(1/3), R)
        if A == 0.0:
            B = 0.0
        else:
            B = Q/A
        x1 = (A + B) - a/3
        x2 = complex(-0.5*(A + B) - a/3, sqrt(3)/2*(A - B))
        x3 = complex(x2.real, 0.0 - x2.imag)
    return x1, x2, x3

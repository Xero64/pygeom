from math import acos, copysign, cos, pi, sqrt

TWOPIO3 = 2*pi/3


def quadratic_roots(a: float, b: float, c: float) -> tuple[float | complex, float | complex]:
    dx = b**2 - 4*a*c
    if dx < 0.0:
        q = -0.5*complex(b, copysign(sqrt(-dx), b))
    else:
        q = -0.5*float(b + copysign(sqrt(dx), b))
    return q/a, c/q


def cubic_roots(a: float, b: float, c: float, d: float) -> tuple[float | complex, float | complex, float | complex]:
    tmp = a
    a = b/tmp
    b = c/tmp
    c = d/tmp
    q = (a**2 - 3*b)/9
    r = (2*a**3 - 9*a*b + 27*c)/54
    q3 = q**3
    r2 = r**2
    ao3 = a/3
    if r2 < q3:
        nsqrtqx2 = -2*sqrt(q)
        tho3 = acos(r/sqrt(q3))/3
        x1 = nsqrtqx2*cos(tho3) - ao3
        x2 = nsqrtqx2*cos(tho3 + TWOPIO3) - ao3
        x3 = nsqrtqx2*cos(tho3 - TWOPIO3) - ao3
    else:
        A = -copysign((abs(r) + sqrt(r2 - q3))**(1/3), r)
        if A == 0.0:
            B = 0.0
        else:
            B = q/A
        x1 = (A + B) - a/3
        x2 = complex(-0.5*(A + B) - ao3, sqrt(3)/2*(A - B))
        x3 = complex(x2.real, -x2.imag)
    return x1, x2, x3

#%%
# Import Dependencies
from math import comb
from typing import TYPE_CHECKING, List, Union, Tuple

from matplotlib.pyplot import figure
from numpy import arctan2, asarray, cos, float64, linspace, sin, zeros
from pygeom.array2d import zero_arrayvector2d
from pygeom.geom2d import Vector2D
from sympy import Symbol, Add

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]

#%%
# Bernstien Polynomial
def bernstein_poly(n: int, i: int, t: 'Numeric') -> 'Numeric':
    omt = 1 - t
    nmi = n - i
    return comb(n, i)*t**i*omt**nmi

def bernstein_polys(n: int, t: 'Numeric') -> List['Numeric']:
    return [bernstein_poly(n, i, t) for i in range(n + 1)]

def bernstein_poly_derivative(n: int, i: int, t: 'Numeric') -> 'Numeric':
    omt = 1 - t
    nmi = n - i
    im1 = i - 1
    nmim1 = nmi - 1
    return comb(n, i)*(i*t**im1*omt**nmi - t**i*nmi*omt**nmim1)

def bernstein_poly_derivatives(n: int, t: 'Numeric') -> List['Numeric']:
    return [bernstein_poly_derivative(n, i, t) for i in range(n + 1)]

#%%
# Define the NurbsCurve class
class NurbsCurve2D():
    points: 'ArrayVector2D' = None
    weights: 'NDArray[float64]' = None
    degree: int = None

    def __init__(self, points: 'ArrayVector2D', weights: 'NDArray[float64]') -> None:
        self.points = points
        self.weights = weights
        self.degree = points.size - 1

    def evaluate_at_t(self, t: 'Numeric') -> 'VectorLike':
        polys = bernstein_polys(self.degree, t)
        if isinstance(t, float64):
            t = asarray([t], dtype=float64)
        denom = zeros(t.shape, dtype=float64)
        points = zero_arrayvector2d(t.shape, dtype=float64)
        for weight, point, poly in zip(self.weights, self.points, polys):
            denom += weight * poly
            points += point * weight * poly
        points = points / denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_points(self, num: int) -> 'ArrayVector2D':
        t = linspace(0.0, 1.0, num, dtype=float64)
        return self.evaluate_at_t(t)

num = 101

#%%
# Define the control points and weights
a = 2.0
b = 1.0

ctlpts = zero_arrayvector2d(3)
ctlpts[0] = Vector2D(a, 0.0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0.0, b)

weights = [1.0, 1.0, 2.0]

nurbscurve = NurbsCurve2D(ctlpts, weights)

pnts = nurbscurve.evaluate_points(num)

th = arctan2(pnts.y, pnts.x)
r = (pnts.x**2 + pnts.y**2)**0.5
x = r*cos(th)
y = r*sin(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='NURBS Curve')
ax.plot(x, y, label='Ellipse')
ax.set_aspect('equal')
_ = ax.legend()

#%%
# Bezier Curve Class
class BezierCurve2D():
    points: 'ArrayVector2D' = None
    degree: int = None

    def __init__(self, points: 'ArrayVector2D') -> None:
        self.points = points
        self.degree = points.size - 1

    def evaluate_at_t(self, t: 'Numeric') -> 'VectorLike':
        polys = bernstein_polys(self.degree, t)
        if isinstance(t, float64):
            t = asarray([t], dtype=float64)
        points = zero_arrayvector2d(t.shape, dtype=float64)
        for point, poly in zip(self.points, polys):
            points += point * poly
        if points.size == 1:
            points = points[0]
        return points

    def symbolic_expression(self) -> Tuple[Add, Add]:
        t = Symbol('t', real=True)
        polys = bernstein_polys(self.degree, t)
        expr_x = 0
        expr_y = 0
        for point, poly in zip(self.points, polys):
            expr_x += point.x * poly
            expr_y += point.y * poly
        return expr_x, expr_y

    def symbolic_derivative(self) -> Tuple[Add, Add]:
        t = Symbol('t', real=True)
        polys = bernstein_poly_derivatives(self.degree, t)
        der_x = 0
        der_y = 0
        for point, poly in zip(self.points, polys):
            der_x += point.x * poly
            der_y += point.y * poly
        return der_x, der_y

    def evaluate_points(self, num: int) -> 'ArrayVector2D':
        t = linspace(0.0, 1.0, num, dtype=float64)
        return self.evaluate_at_t(t)

num = 101

#%%
# Define the control points
# ctlpts = zero_arrayvector2d(7)
# ctlpts[0] = Vector2D(0.0, 0.0)
# ctlpts[1] = Vector2D(0.0, 0.05)
# ctlpts[2] = Vector2D(0.2, 0.15)
# ctlpts[3] = Vector2D(0.4, 0.12)
# ctlpts[4] = Vector2D(0.6, 0.08)
# ctlpts[5] = Vector2D(0.8, 0.04)
# ctlpts[6] = Vector2D(1.0, 0.0)

ctlpts = zero_arrayvector2d(3)
ctlpts[0] = Vector2D(0.0, 0.0)
ctlpts[1] = Vector2D(0.0, 0.05)
ctlpts[2] = Vector2D(1.0, 0.0)

beziercurve = BezierCurve2D(ctlpts)

pnts = beziercurve.evaluate_points(num)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='Bezier Curve')
ax.set_aspect('equal')
_ = ax.legend()

#%%
# Symbolic
t = Symbol('t', real=True)

expr_x, expr_y = beziercurve.symbolic_expression()

print(f'expr_x = {expr_x.expand()}\n')
print(f'expr_y = {expr_y.expand()}\n')

der_x, der_y = beziercurve.symbolic_derivative()

print(f'der_x = {der_x.expand()}\n')
print(f'der_y = {der_y.expand()}\n')

der_x_check = expr_x.diff(t)
der_y_check = expr_y.diff(t)

print(f'der_x_check = {der_x_check.expand()}\n')
print(f'der_y_check = {der_y_check.expand()}\n')

#%%
# Import Dependencies
from math import comb
from typing import TYPE_CHECKING, List, Union

from matplotlib.pyplot import figure
from numpy import arctan2, asarray, cos, float64, linspace, sin, zeros
from pygeom.array2d import zero_arrayvector2d
from pygeom.geom2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]

#%%
# Bernstien Polynomial
def bernstein_poly(n: int, i: int, t: 'Numeric') -> 'Numeric':
    return comb(n, i) * (1.0 - t)**(n - i) * t**i

def bernstein_polys(n: int, t: 'Numeric') -> List['Numeric']:
    return [bernstein_poly(n, i, t) for i in range(n + 1)]

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

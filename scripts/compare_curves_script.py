#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import asarray, float64, linspace, sqrt
from pygeom.array2d import (BezierCurve2D, BSplineCurve2D, NurbsCurve2D,
                            RationalBezierCurve2D, zero_arrayvector2d)
from pygeom.geom2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]


#%%
# Comparing All Curve Types
num = 21
a = 2.0
b = 1.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(a, 0.0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0.0, b)


weights = asarray([1.0, 1.0/sqrt(2.0), 1.0], dtype=float64)
knots = asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=float64)*2

bsc = BSplineCurve2D(ctlpts, knots)
nc = NurbsCurve2D(ctlpts, weights, knots)
bc = BezierCurve2D(ctlpts)
rbc = RationalBezierCurve2D(ctlpts, weights)

u = linspace(knots.min(), knots.max(), num, dtype=float64)
t = linspace(0.0, 1.0, num, dtype=float64)
bspnts = bsc.evaluate_points_at_u(u)
bsvecs = bsc.evaluate_tangents_at_u(u)
npnts = nc.evaluate_points_at_u(u)
nvecs = nc.evaluate_tangents_at_u(u)
bpnts = bc.evaluate_points_at_t(u)
bvecs = bc.evaluate_tangents_at_t(u)
rbpnts = rbc.evaluate_points_at_t(u)
rbvecs = rbc.evaluate_tangents_at_t(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
ax.plot(bspnts.x, bspnts.y, '.', label='BSpline Curve')
ax.plot(npnts.x, npnts.y, '.', label='NURBS Curve')
ax.plot(bpnts.x, bpnts.y, '-.', label='Bezier Curve')
ax.plot(rbpnts.x, rbpnts.y, '--', label='Rational Bezier Curve')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

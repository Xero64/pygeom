#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import asarray, cos, float64, linspace, pi, sqrt
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

ang = pi/2
Kc = 4.0/3.0/(1.0/cos(ang/2) + 1.0)

ctlpts1 = zero_arrayvector2d(3, dtype=float64)
ctlpts1[0] = Vector2D(a, 0.0)
ctlpts1[1] = Vector2D(a, b)
ctlpts1[2] = Vector2D(0.0, b)

weights1 = asarray([1.0, 1.0/sqrt(2.0), 1.0], dtype=float64)
knots1 = asarray([0.0, 1.0], dtype=float64)
knots2 = asarray([0.0, 1.0], dtype=float64)

ctlpts2 = zero_arrayvector2d(4, dtype=float64)
ctlpts2[0] = Vector2D(a, 0.0)
ctlpts2[1] = Vector2D(a, Kc*b)
ctlpts2[2] = Vector2D(Kc*a, b)
ctlpts2[3] = Vector2D(0.0, b)

bsc = BSplineCurve2D(ctlpts2)
nc = NurbsCurve2D(ctlpts1, weights=weights1)
bc = BezierCurve2D(ctlpts2)
rbc = RationalBezierCurve2D(ctlpts1, weights1)

u = linspace(knots1.min(), knots1.max(), num, dtype=float64)
bspnts = bsc.evaluate_points_at_u(u)
bsvecs = bsc.evaluate_first_derivatives_at_u(u)
npnts = nc.evaluate_points_at_u(u)
nvecs = nc.evaluate_first_derivatives_at_u(u)
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
ax.plot(rbpnts.x, rbpnts.y, '--', label='Rational Bezier Curve')
ax.plot(bpnts.x, bpnts.y, '-.', label='Bezier Curve')
ax.scatter(ctlpts1.x, ctlpts1.y, color='r', label='Control Points 1')
ax.scatter(ctlpts2.x, ctlpts2.y, color='g', label='Control Points 2')
_ = ax.legend()

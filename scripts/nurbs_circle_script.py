#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import asarray, cos, float64, pi, sin, sqrt
from pygeom.array2d import NurbsCurve2D, zero_arrayvector2d
from pygeom.geom2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]

#%%
# Define the control points and weights
num = 20
r = 1.0

K = 4.0/3.0*(sqrt(2.0) - 1.0)
ang = pi/2
Kchk = 4.0/3.0/(1.0/cos(ang/2) + 1.0)
print(f'K = {K}')
print(f'Kchk = {Kchk}')

ctlpts = zero_arrayvector2d(4, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, K*r)
ctlpts[2] = Vector2D(K*r, r)
ctlpts[3] = Vector2D(0.0, r)

w = 1.0

weights = asarray([1.0, w, w, 1.0], dtype=float64)

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

print(f'nurbscurve.ctlpnts = {nurbscurve.ctlpnts}')
print(f'nurbscurve.weights = {nurbscurve.weights}')
print(f'nurbscurve.degree = {nurbscurve.degree}')
print(f'nurbscurve.knots = {nurbscurve.knots}')

u = nurbscurve.evaluate_u(num)
npnts = nurbscurve.evaluate_points(num)
nvecs = nurbscurve.evaluate_tangents(num)

th = u*pi/2
x = r*cos(th)
y = r*sin(th)
dxdt = -r*sin(th)*pi/2
dydt = r*cos(th)*pi/2

# tp = linspace(0.5, 0.5, 1)
# npnt = nurbscurve.evaluate_points_at_u(tp)
# nvec = nurbscurve.evaluate_tangents_at_u(tp)

# print(f'tp = {tp}')
# print(f'npnt = {npnt}')
# print(f'npnt.return_angle() = {npnt.return_angle()/(pi/2)}')
# print(f'nvec = {nvec}')
# print(f'nvec.return_angle() = {nvec.return_angle()/(pi/2)}')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x, y, label='Circle')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, x, label='Circle X')
# ax.plot(u, y, label='Circle Y')
# ax.plot(u, npnts.x, '-.', label='NURBS Curve X')
# ax.plot(u, npnts.y, '-.', label='NURBS Curve Y')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, dxdt, label='Circle dX')
# ax.plot(u, dydt, label='Circle dY')
# ax.plot(u, nvecs.x, '-.', label='NURBS Curve dX')
# ax.plot(u, nvecs.y, '-.', label='NURBS Curve dY')
# _ = ax.legend()

# actual_th = arctan2(npnts.y, npnts.x)
# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, actual_th, label='Actual Angle')
# ax.plot(u, th, label='Expected Angle')
# _ = ax.legend()

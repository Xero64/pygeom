#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import arctan2, asarray, cos, float64, linspace, pi, sin, sqrt
from pygeom.array2d import NurbsCurve2D, zero_arrayvector2d
from pygeom.geom2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]

#%%
# Define the control points and weights
num = 21
a = 2.0
b = 1.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(a, 0.0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0.0, b)

weights = asarray([1.0, 1.0, 2.0], dtype=float64)
knots = asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=float64)*pi

nurbscurve = NurbsCurve2D(ctlpts, weights, knots)

u = linspace(knots.min()-pi/2, knots.max()+pi/2, (num-1)*3+1, dtype=float64)
pnts = nurbscurve.evaluate_points_at_u(u)
vecs = nurbscurve.evaluate_tangents_at_u(u)

th = linspace(0.0, 0.5, num, dtype=float64)*pi
x = a*cos(th)
y = b*sin(th)
dx = -a*sin(th)
dy = b*cos(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='NURBS Curve')
ax.scatter(pnts.x, pnts.y, label='NURBS Points')
ax.plot(x, y, label='Ellipse')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, pnts.x, label='NURBS Curve X')
ax.plot(u, pnts.y, label='NURBS Curve Y')
ax.plot(th, x, label='Ellipse X')
ax.plot(th, y, label='Ellipse Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, vecs.x, label='NURBS Curve dX')
ax.plot(u, vecs.y, label='NURBS Curve dY')
ax.plot(th, dx, label='Ellipse dX')
ax.plot(th, dy, label='Ellipse dY')
_ = ax.legend()

#%%
# Define the control points and weights
num = 37
r = 2.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)

weights = asarray([1.0, 2.0**0.5/2.0, 1.0], dtype=float64)
knots = asarray([0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=float64)*pi

nurbscurve = NurbsCurve2D(ctlpts, weights, knots)

u = linspace(knots.min()-pi/2, knots.max()+1.0, (num-1)*3+1, dtype=float64)

pnts = nurbscurve.evaluate_points_at_u(u)

th = arctan2(pnts.y, pnts.x)
x = r*cos(th)
y = r*sin(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='NURBS Curve')
ax.scatter(pnts.x, pnts.y, label='NURBS Points')
ax.plot(x, y, label='Circle')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
ax.set_aspect('equal')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, pnts.x, label='NURBS Curve X')
ax.plot(u, pnts.y, label='NURBS Curve Y')
ax.plot(th, x, label='Circle X')
ax.plot(th, y, label='Circle Y')
_ = ax.legend()

#%%
# Define the control points and weights
num = 73
r = 2.0

ctlpts = zero_arrayvector2d(9, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)
ctlpts[3] = Vector2D(-r, r)
ctlpts[4] = Vector2D(-r, 0.0)
ctlpts[5] = Vector2D(-r, -r)
ctlpts[6] = Vector2D(0.0, -r)
ctlpts[7] = Vector2D(r, -r)
ctlpts[8] = Vector2D(r, 0.0)

w = 1.0/sqrt(2.0)

weights = asarray([1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0], dtype=float64)
knots = asarray([0.0, 0.0, 0.0,
                 0.5, 0.5,
                 1.0, 1.0,
                 1.5, 1.5,
                 2.0, 2.0, 2.0], dtype=float64)*pi

nurbscurve = NurbsCurve2D(ctlpts, weights, knots, degree=2)

u = linspace(knots.min(), knots.max(), num, dtype=float64)

basis = nurbscurve.basis_functions(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(basis.shape[0]):
    ax.plot(u, basis[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

pnts = nurbscurve.evaluate_points(num)
vecs = nurbscurve.evaluate_tangents(num)

th = arctan2(pnts.y, pnts.x)
th[th < 0.0] += 2.0*pi
th[-1] = 2.0*pi
x = r*cos(th)
y = r*sin(th)
dx = -r*sin(th)
dy = r*cos(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='NURBS Curve')
ax.scatter(pnts.x, pnts.y, label='NURBS Points')
ax.plot(x, y, label='Circle')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
ax.set_aspect('equal')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, pnts.x, label='NURBS Curve X')
ax.plot(u, pnts.y, label='NURBS Curve Y')
ax.plot(th, x, label='Circle X')
ax.plot(th, y, label='Circle Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, vecs.x, label='NURBS Curve dX')
ax.plot(u, vecs.y, label='NURBS Curve dY')
ax.plot(th, dx, label='Circle dX')
ax.plot(th, dy, label='Circle dY')
_ = ax.legend()

#%%
# Define the control points and weights
num = 73
r = 2.0

ctlpts = zero_arrayvector2d(7, dtype=float64)
ctlpts[0] = Vector2D(0.0, -r)
ctlpts[1] = Vector2D(2*r*cos(pi/6), -r)
ctlpts[2] = Vector2D(r*cos(pi/6), r*sin(pi/6))
ctlpts[3] = Vector2D(0.0, r/sin(pi/6))
ctlpts[4] = Vector2D(-r*cos(pi/6), r*sin(pi/6))
ctlpts[5] = Vector2D(-2*r*cos(pi/6), -r)
ctlpts[6] = Vector2D(0.0, -r)

w = 0.5

weights = asarray([1.0, w, 1.0, w, 1.0, w, 1.0], dtype=float64)
knots = asarray([-0.5, -0.5, -0.5,
                 -0.5+2/3, -0.5+2/3,
                 -0.5+4/3, -0.5+4/3,
                 1.5, 1.5, 1.5], dtype=float64)*pi

nurbscurve = NurbsCurve2D(ctlpts, weights, knots, degree=2)

u = linspace(knots.min(), knots.max(), num, dtype=float64)

basis = nurbscurve.basis_functions(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(basis.shape[0]):
    ax.plot(u, basis[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

pnts = nurbscurve.evaluate_points(num)
vecs = nurbscurve.evaluate_tangents(num)

th = linspace(-pi/2, 3*pi/2, num, dtype=float64)
# th[-1] = 2.0*pi
x = r*cos(th)
y = r*sin(th)
dx = -r*sin(th)
dy = r*cos(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='NURBS Curve')
ax.scatter(pnts.x, pnts.y, label='NURBS Points')
ax.plot(x, y, label='Circle')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
ax.plot(ctlpts.x, ctlpts.y, color='r', label='Control Points')
ax.set_aspect('equal')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, pnts.x, label='NURBS Curve X')
ax.plot(u, pnts.y, label='NURBS Curve Y')
ax.plot(th, x, label='Circle X')
ax.plot(th, y, label='Circle Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, vecs.x, label='NURBS Curve dX')
ax.plot(u, vecs.y, label='NURBS Curve dY')
ax.plot(th, dx, label='Circle dX')
ax.plot(th, dy, label='Circle dY')
_ = ax.legend()

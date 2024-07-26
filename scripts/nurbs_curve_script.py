#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import arctan2, asarray, cos, float64, linspace, pi, sin, sqrt, gradient
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
a = 2.0
b = 1.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(a, 0.0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0.0, b)

weights = asarray([1.0, 1.0/sqrt(2.0), 1.0], dtype=float64)

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

u = nurbscurve.evaluate_u(num)

npnts = nurbscurve.evaluate_points_at_u(u)
nvecs = nurbscurve.evaluate_first_derivatives_at_u(u)
ncurs = nurbscurve.evaluate_second_derivatives_at_u(u)

dnvecx = gradient(npnts.x, u)
dnvecy = gradient(npnts.y, u)
d2nvecx = gradient(nvecs.x, u)
d2nvecy = gradient(nvecs.y, u)

kappan = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

bt = arctan2(npnts.y, npnts.x)
# th = bt
th = u*pi/2
x = a*cos(th)
y = b*sin(th)
r = sqrt(x**2 + y**2)
dxdt = -a*sin(th)*pi/2
dydt = b*cos(th)*pi/2
d2xdt2 = -a*cos(th)*(pi/2)**2
d2ydt2 = -b*sin(th)*(pi/2)**2

kappae = (d2ydt2*dxdt - d2xdt2*dydt)/(dxdt**2 + dydt**2)**(3/2)

tp = linspace(0.5, 0.5, 1)
npnt = nurbscurve.evaluate_points_at_u(tp)
nvec = nurbscurve.evaluate_first_derivatives_at_u(tp)

print(f'tp = {tp}')
print(f'npnt = {npnt}')
print(f'npnt.return_angle() = {npnt.return_angle()/(pi/2)}')
print(f'nvec = {nvec}')
print(f'nvec.return_angle() = {nvec.return_angle()/(pi/2)}')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x, y, label='Ellipse Curve')
ax.scatter(x, y, label='Ellipse Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

the = arctan2(y, x)
thn = arctan2(npnts.y, npnts.x)

ale = arctan2(dydt, dxdt)
aln = arctan2(nvecs.y, nvecs.x)

bte = arctan2(d2ydt2, d2xdt2)
btn = arctan2(ncurs.y, ncurs.x)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, the, label='Ellipse Position Angle')
ax.plot(u, thn, '-.', label='NURBS Position Angle')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ale, label='Ellipse First Derivative Angle')
ax.plot(u, aln, '-.', label='NURBS First Derivative Angle')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dxdt, label='Ellipse dX')
ax.plot(u, dydt, label='Ellipse dY')
ax.plot(u, nvecs.x, '-.', label='NURBS dX')
ax.plot(u, nvecs.y, '-.', label='NURBS dY')
ax.plot(u, dnvecx, '-.', label='NURBS dX Gradient')
ax.plot(u, dnvecy, '-.', label='NURBS dY Gradient')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, bte, label='Ellipse Second Derivative Angle')
ax.plot(u, btn, '-.', label='NURBS Second Derivative Angle')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, d2xdt2, label='Ellipse d2X')
ax.plot(u, d2ydt2, label='Ellipse d2Y')
ax.plot(u, ncurs.x, '-.', label='NURBS d2X')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Y')
ax.plot(u, d2nvecx, '-.', label='NURBS d2X Gradient')
ax.plot(u, d2nvecy, '-.', label='NURBS d2Y Gradient')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, kappae, label='Ellipse Curvature')
ax.plot(u, kappan, '-.', label='NURBS Curvative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, thn, label='Actual Angle')
ax.plot(u, th, label='Expected Angle')
_ = ax.legend()

#%%
# Define the control points and weights
num = 20
r = 1.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)

weights = asarray([1.0, 1.0/sqrt(2.0), 1.0], dtype=float64)

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

u = nurbscurve.evaluate_u(num)
npnts = nurbscurve.evaluate_points(num)
nvecs = nurbscurve.evaluate_first_derivatives(num)
ncurs = nurbscurve.evaluate_second_derivatives(num)
kappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

th = u*pi/2
x = r*cos(th)
y = r*sin(th)
dxdt = -r*sin(th)*pi/2
dydt = r*cos(th)*pi/2

tp = linspace(0.5, 0.5, 1)
npnt = nurbscurve.evaluate_points_at_u(tp)
nvec = nurbscurve.evaluate_first_derivatives_at_u(tp)

print(f'tp = {tp}')
print(f'npnt = {npnt}')
print(f'npnt.return_angle() = {npnt.return_angle()/(pi/2)}')
print(f'nvec = {nvec}')
print(f'nvec.return_angle() = {nvec.return_angle()/(pi/2)}')

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x, y, label='Circle')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, x, label='Circle X')
ax.plot(u, y, label='Circle Y')
ax.plot(u, npnts.x, '-.', label='NURBS Curve X')
ax.plot(u, npnts.y, '-.', label='NURBS Curve Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dxdt, label='Circle dX')
ax.plot(u, dydt, label='Circle dY')
ax.plot(u, nvecs.x, '-.', label='NURBS Curve dX')
ax.plot(u, nvecs.y, '-.', label='NURBS Curve dY')
_ = ax.legend()

actual_th = arctan2(npnts.y, npnts.x)
fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, actual_th, label='Actual Angle')
ax.plot(u, th, label='Expected Angle')
_ = ax.legend()

#%%
# Define the control points and weights
num = 37
r = 2.0

ctlpts = zero_arrayvector2d(3, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)

weights = asarray([1.0, 1.0/sqrt(2.0), 1.0], dtype=float64)

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

u = nurbscurve.evaluate_u(num)
pnts = nurbscurve.evaluate_points(num)

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
ax.plot(th*2/pi, x, label='Circle X')
ax.plot(th*2/pi, y, label='Circle Y')
_ = ax.legend()

#%%
# Define the control points and weights
num = 20
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
# knots = asarray([0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0], dtype=float64)

nurbscurve = NurbsCurve2D(ctlpts, weights=weights, degree=2)
print(f'nurbscurve.degree = {nurbscurve.degree}')
print(f'nurbscurve.knots = {nurbscurve.knots}')


u = nurbscurve.evaluate_u(num)

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_derivatives(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{nurbscurve.degree}')
_ = ax.legend()

pnts = nurbscurve.evaluate_points(num)
vecs = nurbscurve.evaluate_first_derivatives(num)

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
ax.plot(th/2/pi, x, label='Circle X')
ax.plot(th/2/pi, y, label='Circle Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, vecs.x, label='NURBS Curve dX')
ax.plot(u, vecs.y, label='NURBS Curve dY')
ax.plot(th/2/pi, dx, label='Circle dX')
ax.plot(th/2/pi, dy, label='Circle dY')
_ = ax.legend()

#%%
# Define the control points and weights
num = 20
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

nurbscurve = NurbsCurve2D(ctlpts, weights=weights, degree=2)

u = nurbscurve.evaluate_u(num)

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_derivatives(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{nurbscurve.degree}')
_ = ax.legend()

pnts = nurbscurve.evaluate_points(num)
vecs = nurbscurve.evaluate_first_derivatives(num)

th = linspace(0.0, 2*pi, 4*num, dtype=float64)
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
ax.plot(th/2/pi, x, label='Circle X')
ax.plot(th/2/pi, y, label='Circle Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, vecs.x, label='NURBS Curve dX')
ax.plot(u, vecs.y, label='NURBS Curve dY')
ax.plot(th/2/pi, dx, label='Circle dX')
ax.plot(th/2/pi, dy, label='Circle dY')
_ = ax.legend()

#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from numpy import arctan2, asarray, cos, float64, hstack, sin, sqrt
from pygeom.array3d import ArrayVector, NurbsCurve, zero_arrayvector
from pygeom.geom3d import Vector
from pygeom.tools.k3d import (Plot, k3d_nurbs_control_points, k3d_nurbs_curve,
                              line)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]

#%%
# Full Helix
num = 20
r = 2.0
l = 8.0

ctlpts = zero_arrayvector(9, dtype=float64)
ctlpts[0] = Vector(r, 0.0, 0.0)
ctlpts[1] = Vector(r, r, 1*l/8)
ctlpts[2] = Vector(0.0, r, 2*l/8)
ctlpts[3] = Vector(-r, r, 3*l/8)
ctlpts[4] = Vector(-r, 0.0, 4*l/8)
ctlpts[5] = Vector(-r, -r, 5*l/8)
ctlpts[6] = Vector(0.0, -r, 6*l/8)
ctlpts[7] = Vector(r, -r, 7*l/8)
ctlpts[8] = Vector(r, 0.0, 8*l/8)

w = 1.0/sqrt(2.0)

weights = asarray([1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0], dtype=float64)

nurbscurve = NurbsCurve(ctlpts, weights=weights, degree=2)
print(nurbscurve)

u = nurbscurve.evaluate_u(num)

# Nu = nurbscurve.basis_functions(u)
# dNu = nurbscurve.basis_first_derivatives(u)

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# for i in range(Nu.shape[0]):
#     ax.plot(u, Nu[i, :], label=f'N_{i}^{nurbscurve.degree}')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# for i in range(dNu.shape[0]):
#     ax.plot(u, dNu[i, :], label=f'dN_{i}^{nurbscurve.degree}')
# _ = ax.legend()

npnts = nurbscurve.evaluate_points(num)
# nvecs = nurbscurve.evaluate_first_derivatives(num)
# ncurs = nurbscurve.evaluate_second_derivatives(num)
# nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

# Nu = nurbscurve.basis_functions(u)
# dNu = nurbscurve.basis_first_derivatives(u)
# d2Nu = nurbscurve.basis_second_derivatives(u)

# wpNu = nurbscurve.wpoints@Nu
# wpdNu = nurbscurve.wpoints@dNu
# wpd2Nu = nurbscurve.wpoints@d2Nu

th = arctan2(npnts.y, npnts.x)

# magwpNu2 = wpNu.return_magnitude()**2

# dthdu = wpNu.cross(wpdNu)/magwpNu2
# d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)
z = l*u

# drdth = 0
# dxdth = -r*sin(th)
# dydth = r*cos(th)

# d2xdth2 = -r*cos(th)
# d2ydth2 = -r*sin(th)

# cpnts = ArrayVector(x, y, z)
# cvecs = ArrayVector(dxdth, dydth)*dthdu
# ccurs = ArrayVector(d2xdth2, d2ydth2)*dthdu**2 + ArrayVector(dxdth, dydth)*d2thdu2

# ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, th, label='Circle Angle')
# ax.plot(u, dthdu, label='Circle Angle Derivative')
# ax.plot(u, d2thdu2, label='Circle Angle Second Derivative')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
# ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
# ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
# ax.scatter(npnts.x, npnts.y, label='NURBS Points')
# ax.set_aspect('equal')
# ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, cpnts.x, label='Circle X')
# ax.plot(u, cpnts.y, label='Circle Y')
# ax.plot(u, npnts.x, '-.', label='NURBS X')
# ax.plot(u, npnts.y, '-.', label='NURBS Y')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, cvecs.x, label='Circle dXdu')
# ax.plot(u, cvecs.y, label='Circle dYdu')
# ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
# ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, ccurs.x, label='Circle d2Xdu2')
# ax.plot(u, ccurs.y, label='Circle d2Ydu2')
# ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
# ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
# _ = ax.legend()

# fig = figure(figsize=(12, 8))
# ax = fig.gca()
# ax.grid(True)
# ax.plot(u, ckappa, label='Circle Curvature')
# ax.plot(u, nkappa, '-.', label='NURBS Curvative')
# _ = ax.legend()

#%%
# k3d Plot
plot = Plot()
plot += k3d_nurbs_curve(nurbscurve)
plot += k3d_nurbs_control_points(nurbscurve, scale=0.2)
plot += line(hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1))).astype('float32'),
             color=0xff0000)
plot.display()

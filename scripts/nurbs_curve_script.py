#%%
# Import Dependencies
from k3d import Plot as k3dPlot, line
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import asarray, cos, float64, linspace, pi, sin, sqrt
from pygeom.array3d import BSplineCurve, zero_arrayvector
from pygeom.geom3d import Vector

#%%
# Define the control points and weights
num = 73
r = 1.0

ctlpts = zero_arrayvector(7, dtype=float64)
ctlpts[0] = Vector(1.0, 0.0, 1.0)
ctlpts[1] = Vector(1.0, sqrt(3.0), 0.5)
ctlpts[2] = Vector(-0.5, sqrt(3.0)/2, 1.0)
ctlpts[3] = Vector(-2.0, 0.0, 0.5)
ctlpts[4] = Vector(-0.5, -sqrt(3.0)/2, 1.0)
ctlpts[5] = Vector(1.0, -sqrt(3.0), 0.5)
ctlpts[6] = Vector(1.0, 0.0, 1.0)

knots = asarray([0.0, 0.0, 0.0, 1.0/3.0, 1.0/3.0,
                 2.0/3.0, 2.0/3.0, 1.0, 1.0, 1.0], dtype=float64)*2*pi

nurbscurve = BSplineCurve(ctlpts, knots, degree=2)

u = linspace(knots.min(), knots.max(), num, dtype=float64)

basis = nurbscurve.basis_functions(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(basis.shape[0]):
    ax.plot(u, basis[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

pnts = nurbscurve.evaluate_points(num)

th = linspace(0.0, 2.0*pi, num, dtype=float64)
x = r*cos(th)
y = r*sin(th)
dx = -r*sin(th)
dy = r*cos(th)

fig = figure(figsize=(12, 8))
ax = Axes3D(fig)
fig.add_axes(ax)
ax.grid(True)
ax.plot(pnts.x, pnts.y, pnts.z, label='NURBS Curve')
ax.scatter(pnts.x, pnts.y, pnts.z, label='NURBS Points')
ax.plot(x, y, 1.0, label='Circle')
ax.scatter(ctlpts.x, ctlpts.y, ctlpts.z, color='r', label='Control Points')
ax.plot(ctlpts.x, ctlpts.y, ctlpts.z, color='r', label='Control Points')
ax.set_aspect('equal')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, pnts.x, label='NURBS Curve X')
ax.plot(u, pnts.y, label='NURBS Curve Y')
ax.plot(u, pnts.z, label='NURBS Curve Z')
ax.plot(th, x, label='Circle X')
ax.plot(th, y, label='Circle Y')
_ = ax.legend()

#%%
# k3d plot
k3dplot = k3dPlot()

k3dpnts = pnts.stack_xyz().astype('float32')
k3dctls = ctlpts.stack_xyz().astype('float32')
k3dplot += line(k3dpnts, color=0x00ff00)
k3dplot += line(k3dctls, color=0xff0000)

k3dplot.display()

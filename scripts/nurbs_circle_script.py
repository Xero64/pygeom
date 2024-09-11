#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import cos, float64, pi, sqrt, linspace, set_printoptions
from pygeom.geom2d import NurbsCurve2D, Vector2D, zero_vector2d
from pygeom.tools.solvers import cubic_bspline_fit_solver

from scipy.interpolate import splprep, splev

set_printoptions(suppress=True)

#%%
# Define the control points and weights
num = 90
r = 2.0

K = 4.0/3.0*(sqrt(2.0) - 1.0)
ang = pi/2
Kchk = 4.0/3.0/(1.0/cos(ang/2) + 1.0)
print(f'K = {K}')
print(f'Kchk = {Kchk}')

K = 0.5

ctlpts = zero_vector2d(13, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, K*r)
ctlpts[2] = Vector2D(K*r, r)
ctlpts[3] = Vector2D(0.0, r)
ctlpts[4] = Vector2D(-K*r, r)
ctlpts[5] = Vector2D(-r, K*r)
ctlpts[6] = Vector2D(-r, 0.0)
ctlpts[7] = Vector2D(-r, -K*r)
ctlpts[8] = Vector2D(-K*r, -r)
ctlpts[9] = Vector2D(0.0, -r)
ctlpts[10] = Vector2D(K*r, -r)
ctlpts[11] = Vector2D(r, -K*r)
ctlpts[12] = Vector2D(r, 0.0)

nurbscurve = NurbsCurve2D(ctlpts, degree=3)

print(nurbscurve)

u = nurbscurve.evaluate_u(num)
npnts = nurbscurve.evaluate_points_at_u(u)
nvecs = nurbscurve.evaluate_first_derivatives_at_u(u)
ncurs = nurbscurve.evaluate_second_derivatives_at_u(u)
nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

splpnts = ctlpts[::3].stack_xy()

print('\n')
print(f'splpnts = \n{splpnts}\n')

tck, splu = splprep(splpnts.T, s=0, per=1)

print(f'tck = \n{tck}\n')
print(f'splu = {splu}\n')

tck1 = tck[0]
tck2a = tck[1][0]
tck2b = tck[1][1]
tck3 = tck[2]

print(f'tck1 = {tck1}\n')
print(f'tck2a = {tck2a}\n')
print(f'tck2b = {tck2b}\n')
print(f'tck3 = {tck3}\n')

u_spl = linspace(splu.min(), splu.max(), 4*num)
x_spl, y_spl = splev(u_spl, tck, der=0)
dx_spl, dy_spl = splev(u_spl, tck, der=1)
d2x_spl, d2y_spl = splev(u_spl, tck, der=2)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x_spl, y_spl, label='Scipy Curve')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, x_spl, label='Scipy X')
ax.plot(u_spl, y_spl, label='Scipy Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, dx_spl, label='Scipy dXdu')
ax.plot(u_spl, dy_spl, label='Scipy dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, d2x_spl, label='Scipy d2Xdu2')
ax.plot(u_spl, d2y_spl, label='Scipy d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# BSpline Cubic Fit
rmat = cubic_bspline_fit_solver(5, bc_type='periodic')

pnts = ctlpts[::3]

newpts = pnts.rmatmul(rmat)

bsplinecurve = NurbsCurve2D(newpts, degree=3)

print(bsplinecurve)

ub = bsplinecurve.evaluate_u(num)
bpnts = bsplinecurve.evaluate_points_at_u(ub)
bvecs = bsplinecurve.evaluate_first_derivatives_at_u(ub)
bcurs = bsplinecurve.evaluate_second_derivatives_at_u(ub)
bkappa = bvecs.cross(bcurs)/bvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x_spl, y_spl, label='Scipy Curve')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.plot(bpnts.x, bpnts.y, '-.', label='BSpline Curve')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='NURBS Control Points')
ax.scatter(newpts.x, newpts.y, color='g', label='BSpline Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, x_spl, label='Scipy X')
ax.plot(u_spl, y_spl, label='Scipy Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
ax.plot(ub, bpnts.x, '-.', label='BSpline X')
ax.plot(ub, bpnts.y, '-.', label='BSpline Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, dx_spl, label='Scipy dXdu')
ax.plot(u_spl, dy_spl, label='Scipy dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
ax.plot(ub, bvecs.x, '-.', label='BSpline dXdu')
ax.plot(ub, bvecs.y, '-.', label='BSpline dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u_spl, d2x_spl, label='Scipy d2Xdu2')
ax.plot(u_spl, d2y_spl, label='Scipy d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
ax.plot(ub, bcurs.x, '-.', label='BSpline d2Xdu2')
ax.plot(ub, bcurs.y, '-.', label='BSpline d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
ax.plot(ub, bkappa, '-.', label='BSpline Curvative')
_ = ax.legend()

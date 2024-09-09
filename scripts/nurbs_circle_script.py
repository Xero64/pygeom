#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, cos, float64, pi, sin, sqrt
from pygeom.geom2d import NurbsCurve2D, Vector2D, zero_vector2d

#%%
# Define the control points and weights
num = 20
r = 1.0

K = 4.0/3.0*(sqrt(2.0) - 1.0)
ang = pi/2
Kchk = 4.0/3.0/(1.0/cos(ang/2) + 1.0)
print(f'K = {K}')
print(f'Kchk = {Kchk}')

ctlpts = zero_vector2d(4, dtype=float64)
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
nvecs = nurbscurve.evaluate_first_derivatives(num)

th = u*pi/2
x = r*cos(th)
y = r*sin(th)
dxdt = -r*sin(th)*pi/2
dydt = r*cos(th)*pi/2

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(x, y, label='Circle')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

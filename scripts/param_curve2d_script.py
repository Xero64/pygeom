#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import cos, pi, sin
from pygeom.geom2d import ParamCurve2D, Vector2D
from pygeom.tools.k3d import Plot, k3d_curve

#%%
# Parametric Circle
num = 200

radius = 2.0
twopi = 2*pi
twopi2 = twopi**2

ru = lambda u: radius*Vector2D(cos(twopi*u), sin(twopi*u))
drdu = lambda u: twopi*radius*Vector2D(-sin(twopi*u), cos(twopi*u))
d2rdu2 = lambda u: -twopi2*radius*Vector2D(cos(twopi*u), sin(twopi*u))

paramcurve = ParamCurve2D(ru, drdu, d2rdu2)

u = paramcurve.evaluate_u(num)
cpnts = paramcurve.evaluate_points(num)
cvecs = paramcurve.evaluate_first_derivatives(num)
ccurs = paramcurve.evaluate_second_derivatives(num)

ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
ax.set_aspect('equal')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cpnts.x, label='Circle X')
ax.plot(u, cpnts.y, label='Circle Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cvecs.x, label='Circle dXdu')
ax.plot(u, cvecs.y, label='Circle dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ccurs.x, label='Circle d2Xdu2')
ax.plot(u, ccurs.y, label='Circle d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ckappa, label='Circle Curvature')
_ = ax.legend()

#%%
# k3d Plot
plot = Plot()
plot += k3d_curve(paramcurve, unum=num, color=0xff0000)
plot.display()

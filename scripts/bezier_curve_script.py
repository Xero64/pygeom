#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, cos, linspace, pi, sin
from pygeom.geom2d import (BezierCurve2D, RationalBezierCurve2D, Vector2D,
                           zero_vector2d)
from sympy import Symbol, sqrt

#%%
# Create Symbols
w = Symbol('w', real=True, positive=True)
t = Symbol('t', real=True, positive=True)

#%%
# Create Bezier Curve
num = 21

ctlpts = zero_vector2d(3)
ctlpts[0] = Vector2D(0.0, 0.0)
ctlpts[1] = Vector2D(0.0, 0.05)
ctlpts[2] = Vector2D(1.0, 0.0)

beziercurve = BezierCurve2D(ctlpts)

pnts = beziercurve.evaluate_points(num)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='Bezier Curve')
ax.set_aspect('equal')
_ = ax.legend()

#%%
# Define the control points
num = 21
a = 2
b = 1

ctlpts = zero_vector2d(3, dtype=int)
ctlpts[0] = Vector2D(a, 0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0, b)

weights = asarray([1, 1, 2], dtype=int)

beziercurve = RationalBezierCurve2D(ctlpts, weights)

pnts = beziercurve.evaluate_points(num)

th = linspace(0.0, 0.5, num)*pi
x = a*cos(th)
y = b*sin(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='Rational Bezier Curve')
ax.scatter(pnts.x, pnts.y, label='Points')
ax.plot(x, y, label='Ellipse')
ax.set_aspect('equal')
_ = ax.legend()

#%%
# Define the control points
num = 21
r = 2

ctlpts = zero_vector2d(3, dtype=int)
ctlpts[0] = Vector2D(r, 0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0, r)

weights = asarray([1, 1, 2], dtype=int)

beziercurve = RationalBezierCurve2D(ctlpts, weights)

pnts = beziercurve.evaluate_points(num)

th = linspace(0, 0.5, num)*pi
x = r*cos(th)
y = r*sin(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='Rational Bezier Curve')
ax.scatter(pnts.x, pnts.y, label='Points')
ax.plot(x, y, label='Circle')
ax.set_aspect('equal')
_ = ax.legend()

#%%
# Define the control points
num = 21
r = 2

K = 4/3*(sqrt(2) - 1)
ang = pi/2
K = 4.0/3.0/(1.0/cos(ang/2) + 1.0)

ctlpts = zero_vector2d(4, dtype=int)
ctlpts[0] = Vector2D(r, 0)
ctlpts[1] = Vector2D(r, K*r)
ctlpts[2] = Vector2D(K*r, r)
ctlpts[3] = Vector2D(0, r)

beziercurve = BezierCurve2D(ctlpts)

pnts = beziercurve.evaluate_points(num)

th = linspace(0, 0.5, num)*pi
x = r*cos(th)
y = r*sin(th)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(pnts.x, pnts.y, label='Bezier Curve')
ax.scatter(pnts.x, pnts.y, label='Points')
ax.plot(x, y, label='Circle')
ax.set_aspect('equal')
_ = ax.legend()

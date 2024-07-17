#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from matplotlib.pyplot import figure
from numpy import cos, float64, linspace, pi, sin, asarray
from pygeom.array2d import (BezierCurve2D, RationalBezierCurve2D,
                            zero_arrayvector2d)
from pygeom.geom2d import Vector2D
from sympy import Symbol

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]

#%%
# Create Symbols
w = Symbol('w', real=True, positive=True)
t = Symbol('t', real=True, positive=True)

#%%
# Create Bezier Curve
num = 21

ctlpts = zero_arrayvector2d(3)
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
# Symbolic
expr = beziercurve.symbolic_expression()

print(f'expr = {expr.expand()}\n')

dexpr = beziercurve.symbolic_derivative()

print(f'dexpr = {dexpr.expand()}\n')

dexpr_check = expr.diff(t)

print(f'dexpr_check = {dexpr_check.expand()}\n')

#%%
# Define the control points
num = 21
a = 2
b = 1

ctlpts = zero_arrayvector2d(3, dtype=int)
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
# Symbolic
expr = beziercurve.symbolic_expression()

print(f'expr = {expr.expand()}\n')

dexpr = beziercurve.symbolic_derivative()

print(f'dexpr = {dexpr.expand()}\n')

dexpr_check = expr.diff(t)

print(f'dexpr_check = {dexpr_check.expand()}\n')

d2expr = dexpr.diff(t)

print(f'd2expr = {d2expr.simplify()}\n')

rc = dexpr.return_magnitude()**3/dexpr_check.cross(d2expr)

print(f'r = {rc.simplify().factor()}\n')

#%%
# Define the control points
num = 21
r = 2

ctlpts = zero_arrayvector2d(3, dtype=int)
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
# Symbolic
expr = beziercurve.symbolic_expression()

print(f'expr = {expr.simplify()}\n')

dexpr = beziercurve.symbolic_derivative()

print(f'dexpr = {dexpr.simplify()}\n')

dexpr_check = expr.diff(t)

print(f'dexpr_check = {dexpr_check.simplify()}\n')

d2expr = dexpr.diff(t)

print(f'd2expr = {d2expr.simplify()}\n')

rc = dexpr.return_magnitude()**3/dexpr_check.cross(d2expr)

print(f'r = {rc.simplify().factor()}\n')

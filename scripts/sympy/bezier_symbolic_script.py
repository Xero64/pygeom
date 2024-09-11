#%%
# Import Dependencies
from sympy import Symbol
from pygeom.geom2d import Vector2D
from numpy import empty
from pygeom.geom2d.beziercurve2d import BezierCurve2D

#%%
# Create Symbols
x = Symbol('x', real=True)
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xc', real=True)
xd = Symbol('xd', real=True)
xe = Symbol('xe', real=True)
xf = Symbol('xf', real=True)
xg = Symbol('xg', real=True)
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
yc = Symbol('yc', real=True)
yd = Symbol('yd', real=True)
ye = Symbol('ye', real=True)
yf = Symbol('yf', real=True)
yg = Symbol('yg', real=True)
t = Symbol('t', real=True)

#%%
# Define the Symbolic Bezier Curve
x1_arr = empty(4, dtype=object)
y1_arr = empty(4, dtype=object)
x1_arr[0] = xa
x1_arr[1] = xb
x1_arr[2] = xc
x1_arr[3] = xd
y1_arr[0] = ya
y1_arr[1] = yb
y1_arr[2] = yc
y1_arr[3] = yd

ctlpnts1 = Vector2D(x1_arr, y1_arr)

bezier_curve1 = BezierCurve2D(ctlpnts1)

expr1 = bezier_curve1.symbolic_expression().expand()

print(f'expr1 = {expr1}\n')

dexpr1 = expr1.diff(t).expand()

print(f'dexpr1 = {dexpr1}\n')

d2expr1 = dexpr1.diff(t).expand()

print(f'd2expr1 = {d2expr1}\n')

d3expr1 = d2expr1.diff(t).expand()

print(f'd3expr1 = {d3expr1}\n')

ta = 0
td = 1

dexpr1_a = dexpr1.subs(t, ta).expand()
dexpr1_d = dexpr1.subs(t, td).expand()

print(f'dexpr1_a = {dexpr1_a}')
print(f'dexpr1_d = {dexpr1_d}')

d2expr1_a = d2expr1.subs(t, ta).expand()
d2expr1_d = d2expr1.subs(t, td).expand()

print(f'd2expr1_a = {d2expr1_a}')
print(f'd2expr1_d = {d2expr1_d}')

print('\n---\n')

x2_arr = empty(4, dtype=object)
y2_arr = empty(4, dtype=object)
x2_arr[0] = xd
x2_arr[1] = xe
x2_arr[2] = xf
x2_arr[3] = xg
y2_arr[0] = yd
y2_arr[1] = ye
y2_arr[2] = yf
y2_arr[3] = yg

ctlpnts2 = Vector2D(x2_arr, y2_arr)

bezier_curve2 = BezierCurve2D(ctlpnts2)

expr2 = bezier_curve2.symbolic_expression().expand()

print(f'expr2 = {expr2}\n')

dexpr2 = expr2.diff(t).expand()

print(f'dexpr2 = {dexpr2}\n')

d2expr2 = dexpr2.diff(t).expand()

print(f'd2expr2 = {d2expr2}\n')

d3expr2 = d2expr2.diff(t).expand()

print(f'd3expr2 = {d3expr2}\n')

td = 0
tg = 1

dexpr2_d = dexpr2.subs(t, td).expand()
dexpr2_g = dexpr2.subs(t, tg).expand()

print(f'dexpr2_d = {dexpr2_d}')
print(f'dexpr2_g = {dexpr2_g}')

d2expr2_d = d2expr2.subs(t, td).expand()
d2expr2_g = d2expr2.subs(t, tg).expand()

print(f'd2expr2_d = {d2expr2_d}')
print(f'd2expr2_g = {d2expr2_g}')

print('\n---\n')

d2eqna = d2expr1_a
deqnd = dexpr1_d - dexpr2_d
d2eqnd = d2expr1_d - d2expr2_d
d2eqng = d2expr2_g

d3eqnd = d3expr1 - d3expr2

print(f'd2eqna = {d2eqna}')
print(f'deqnd = {deqnd}')
print(f'd2eqnd = {d2eqnd}')
print(f'd2eqng = {d2eqng}')

print(f'd3eqnd = {d3eqnd}')

eqns = [d2eqna, deqnd, d2eqnd, d2eqng]
knowns = [xa, xd, xg]
unknowns = [xb, xc, xe, xf]

num_eqns = len(eqns)
num_knowns = len(knowns)
num_unknowns = len(unknowns)

A = empty((num_eqns, num_unknowns), dtype=object)
b = empty((num_eqns, num_knowns), dtype=object)
for i, eqn in enumerate(eqns):
    for j, unknown in enumerate(unknowns):
        A[i, j] = eqn.x.coeff(unknown)
    for k, known in enumerate(knowns):
        b[i, k] = -eqn.x.coeff(known)

print('\n---\n')

print(f'A = \n{A}\n')
print(f'b = \n{b}\n')

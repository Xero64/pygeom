#%%
# Import Dependencies
from numpy import empty
from pygeom.geom2d import BezierCurve2D
from pygeom.symbol2d import SymbolicVector2D, empty_symbolicvector2d
from sympy import Symbol
from sympy.solvers import solve

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
ctlpnts1 = empty_symbolicvector2d(4)
ctlpnts1[0] = SymbolicVector2D(xa, ya)
ctlpnts1[1] = SymbolicVector2D(xb, yb)
ctlpnts1[2] = SymbolicVector2D(xc, yc)
ctlpnts1[3] = SymbolicVector2D(xd, yd)

bezier_curve1 = BezierCurve2D(ctlpnts1)

expr1: SymbolicVector2D = bezier_curve1.evaluate_points_at_t(t)
expr1.__class__ = SymbolicVector2D
expr1 = expr1.expand()

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


ctlpnts2 = empty_symbolicvector2d(4)
ctlpnts2[0] = SymbolicVector2D(xd, yd)
ctlpnts2[1] = SymbolicVector2D(xe, ye)
ctlpnts2[2] = SymbolicVector2D(xf, yf)
ctlpnts2[3] = SymbolicVector2D(xg, yg)

bezier_curve2 = BezierCurve2D(ctlpnts1)

expr2: SymbolicVector2D = bezier_curve2.evaluate_points_at_t(t)
expr2.__class__ = SymbolicVector2D
expr2 = expr2.expand()

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

print('\n---\n')

kappa1 = dexpr1.cross(d2expr1)

kappa1 = kappa1.simplify()

print(f'kappa1 = {kappa1}\n')

#%%
# Solve in terms of curvatures
d2xa = Symbol('d2xa', real=True)
d2xd = Symbol('d2xd', real=True)
d2ya = Symbol('d2ya', real=True)
d2yd = Symbol('d2yd', real=True)

eqnx_a = d2expr1_a.x - d2xa
eqnx_d = d2expr1_d.x - d2xd
eqny_a = d2expr1_a.y - d2ya
eqny_d = d2expr1_d.y - d2yd

res = solve([eqnx_a, eqnx_d, eqny_a, eqny_d], [xb, xc, yb, yc])

for sym in res:
    print(f'{sym} = {res[sym]}\n')

expr1_02 = expr1.subs(res).expand()

print(f'expr1_02 = {expr1_02}\n')

dexpr1_02 = expr1_02.diff(t).expand()

print(f'dexpr1_02 = {dexpr1_02}\n')

d2expr1_02 = dexpr1_02.diff(t).expand()

print(f'd2expr1_02 = {d2expr1_02}\n')

d3expr1_02 = d2expr1_02.diff(t).expand()

print(f'd3expr1_02 = {d3expr1_02}\n')

ta = 0
td = 1

dexpr1_02_a = dexpr1_02.subs(t, ta).expand()
dexpr1_02_d = dexpr1_02.subs(t, td).expand()

print(f'dexpr1_02_a = {dexpr1_02_a}')
print(f'dexpr1_02_d = {dexpr1_02_d}')

d2expr1_02_a = d2expr1_02.subs(t, ta).expand()
d2expr1_02_d = d2expr1_02.subs(t, td).expand()

print(f'd2expr1_02_a = {d2expr1_02_a}')
print(f'd2expr1_02_d = {d2expr1_02_d}')

#%%
# Solve in terms of tangents
dxa = Symbol('dxa', real=True)
dxd = Symbol('dxd', real=True)
dya = Symbol('dya', real=True)
dyd = Symbol('dyd', real=True)

eqnx_a = dexpr1_a.x - dxa
eqnx_d = dexpr1_d.x - dxd
eqny_a = dexpr1_a.y - dya
eqny_d = dexpr1_d.y - dyd

res = solve([eqnx_a, eqnx_d, eqny_a, eqny_d], [xb, xc, yb, yc])

for sym in res:
    print(f'{sym} = {res[sym]}\n')

expr1_01 = expr1.subs(res).expand()

print(f'expr1_01 = {expr1_01}\n')

dexpr1_01 = expr1_01.diff(t).expand()

print(f'dexpr1_01 = {dexpr1_01}\n')

d2expr1_01 = dexpr1_01.diff(t).expand()

print(f'd2expr1_01 = {d2expr1_01}\n')

d3expr1_01 = d2expr1_01.diff(t).expand()

print(f'd3expr1_01 = {d3expr1_01}\n')

ta = 0
td = 1

dexpr1_01_a = dexpr1_01.subs(t, ta).expand()
dexpr1_01_d = dexpr1_01.subs(t, td).expand()

print(f'dexpr1_01_a = {dexpr1_01_a}')
print(f'dexpr1_01_d = {dexpr1_01_d}')

d2expr1_01_a = d2expr1_01.subs(t, ta).expand()
d2expr1_01_d = d2expr1_01.subs(t, td).expand()

print(f'd2expr1_01_a = {d2expr1_01_a}')
print(f'd2expr1_01_d = {d2expr1_01_d}')

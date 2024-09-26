#%%
# Import Dependencies
from sympy import Symbol, Expr
from sympy.solvers import solve
from pygeom.tools.bernstein import symbolic_bernstein_polynomials

#%%
# Create Symbols
s = Symbol('s', real=True)
sa = Symbol('sa', real=True)
sb = Symbol('sb', real=True)
sba = Symbol('sba', real=True)

xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
dxa = Symbol('dxa', real=True)
dxb = Symbol('dxb', real=True)
d2xa = Symbol('d2xa', real=True)
d2xb = Symbol('d2xb', real=True)

#%%
# Create Expression
A = (sb - s)/sba
B = (s - sa)/sba
C = (A**3 - A)/6*sba**2
D = (B**3 - B)/6*sba**2

x: Expr = A*xa + B*xb + C*d2xa + D*d2xb
x = x.expand()

print(f'x = {x}\n')

dx = x.diff(s)
dx: Expr = dx.expand()

print(f'dx = {dx.collect((xa, xb, d2xa, d2xb))}\n')

d2x = dx.diff(s)
d2x: Expr = d2x.expand()

print(f'd2x = {d2x.collect((xa, xb, d2xa, d2xb))}\n')

d3x = d2x.diff(s)
d3x: Expr = d3x.expand()

print(f'd3x = {d3x.collect((xa, xb, d2xa, d2xb))}\n')

x_a = x.subs(s, sa).subs(sa, sb - sba)
x_a = x_a.expand()

x_b = x.subs(s, sb).subs(sb, sa + sba)
x_b = x_b.expand()

print(f'x_a = {x_a}\n')
print(f'x_b = {x_b}\n')

dx_a = dx.subs(s, sa).subs(sa, sb - sba)
dx_a = dx_a.expand()

dx_b = dx.subs(s, sb).subs(sb, sa + sba)
dx_b = dx_b.expand()

print(f'dx_a = {dx_a}\n')
print(f'dx_b = {dx_b}\n')

d2x_a = d2x.subs(s, sa).subs(sa, sb - sba)
d2x_a = d2x_a.expand()

d2x_b = d2x.subs(s, sb).subs(sb, sa + sba)
d2x_b = d2x_b.expand()

print(f'd2x_a = {d2x_a}\n')
print(f'd2x_b = {d2x_b}\n')

#%%
# Solve for Bezier Control Points
t = Symbol('t', real=True)
polys = symbolic_bernstein_polynomials(3, t)

print(f'polys = {polys}\n')

xc = Symbol('xc', real=True)
xd = Symbol('xd', real=True)

xlst = [xa, xc, xd, xb]

poly: Expr = 0
for var, expr in zip(xlst, polys):
    poly += expr*var

print(f'poly = {poly}\n')

dpoly: Expr = poly.diff(t)
d2poly: Expr = dpoly.diff(t)

dsdt: Expr = sba

dpolyds: Expr = dpoly/dsdt

print(f'dpolyds = {dpolyds}\n')

d2polyds: Expr = d2poly/dsdt**2

print(f'd2polyds = {d2polyds}\n')

d2polyds_a = d2polyds.subs(t, 0)
d2polyds_b = d2polyds.subs(t, 1)

print(f'd2polyds_a = {d2polyds_a}\n')
print(f'd2polyds_b = {d2polyds_b}\n')

eqna = d2polyds_a - d2xa
eqnb = d2polyds_b - d2xb

sol = solve((eqna, eqnb), (xc, xd))

for sym in sol:
    print(f'{sym} = {sol[sym]}\n')

#%%
# Import Depenedencies
from sympy import Symbol, Poly
from sympy.solvers import solve

#%%
# Create Symbols
x = Symbol('x', real=True)
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xc', real=True)
xo = Symbol('xo', real=True)

y = Symbol('y', real=True)
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
yc = Symbol('yc', real=True)
yo = Symbol('yo', real=True)

r = Symbol('r', real=True, positive=True)

a = Symbol('a', real=True)
b = Symbol('b', real=True)
c = Symbol('c', real=True)

#%%
# General Equation
eqn = (x - xo)**2 + (y - yo)**2 - r**2
eqn = eqn.expand()
print(f'eqn = {eqn}\n')

poly = Poly(eqn, (x, y))
print(f'poly = {poly}\n')

coeffs = poly.coeffs()
a_subs = coeffs[1]
b_subs = coeffs[3]
c_subs = coeffs[4]

print(f'a = {a_subs}\n')
print(f'b = {b_subs}\n')
print(f'c = {c_subs}\n')

eqn = eqn.subs([(a_subs, a), (b_subs, b), (c_subs, c)])
print(f'eqn = {eqn}\n')

#%%
# Create Equations
eqna = eqn.subs([(x, xa), (y, ya)])
eqnb = eqn.subs([(x, xb), (y, yb)])
eqnc = eqn.subs([(x, xc), (y, yc)])

print(f'eqna = {eqna}\n')
print(f'eqnb = {eqnb}\n')
print(f'eqnc = {eqnc}\n')

eqns = [eqna, eqnb, eqnc]
syms = [a, b, c]

res = solve(eqns, syms)

for i, resi in res.items():
    print(f'{a} = {resi}')

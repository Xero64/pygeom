#%%
# Import Dependencies


from sympy import Expr, Symbol
from sympy.solvers import solve

#%%
# Create Symbols
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xd', real=True)
xd = Symbol('xd', real=True)
dxa = Symbol('dxa', real=True)
dxd = Symbol('dxd', real=True)
la = Symbol('la', real=True)
ld = Symbol('ld', real=True)

#%%
# Solve Equations
x1: Expr = xa + la*dxa
x2: Expr = xd + ld*dxd
eqn_x = x1 - x2
eqn_s = la - ld - 1

res: dict[Symbol, Expr] = solve([eqn_x, eqn_s], [la, ld])

for sym in res:
    print(f'{sym} = {res[sym]}')

xc_1: Expr = x1.subs(res)
xc_1 = xc_1.expand()
xc_1 = xc_1.simplify()

print(f'xc_1 = {xc_1}')

xc_2: Expr = x2.subs(res)
xc_2 = xc_2.expand()
xc_2 = xc_2.simplify()

print(f'xc_2 = {xc_2}')

xc_check: Expr = xc_1.subs({dxa: 2*(xd - xa), dxd: 3*(xd - xa)})
xc_check = xc_check.expand()
xc_check = xc_check.simplify()

print(f'xc_check = {xc_check}')

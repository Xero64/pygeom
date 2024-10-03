#%%
# Import Dependencies
from sympy import Derivative, Expr, Function, Symbol

#%%
# Define the Symbols and Functions
u = Symbol('u', real=True)
numer = Function('numer', real=True)(u)
denom = Function('denom', real=True)(u)

numer_val = Symbol('numer', real=True)
dnumer_val = Symbol('dnumer', real=True)
d2numer_val = Symbol('d2numer', real=True)
denom_val = Symbol('denom', real=True)
ddenom_val = Symbol('ddenom', real=True)
d2denom_val = Symbol('d2denom', real=True)

#%%
# Basis Functions
Nu: Expr = numer/denom
print(f'Nu = {Nu}\n')

dNu: Expr = Nu.diff(u)
print(f'dNu = {dNu}\n')

d2Nu: Expr = dNu.diff(u)
print(f'd2Nu = {d2Nu}\n')

#%%
# Substitute the Values
sbs = {
    numer: numer_val,
    Derivative(numer, u): dnumer_val,
    Derivative(numer, (u, 2)): d2numer_val,
    denom: denom_val,
    Derivative(denom, u): ddenom_val,
    Derivative(denom, (u, 2)): d2denom_val
}

Nu: Expr = Nu.subs(sbs).together()
print(f'Nu = {Nu}\n')

dNu: Expr = dNu.subs(sbs).together()
print(f'dNu = {dNu}\n')

d2Nu: Expr = d2Nu.subs(sbs).together()
print(f'd2Nu = {d2Nu}\n')

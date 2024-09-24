#%%
# Import Dependencies
from sympy import Derivative, Function, Symbol, atan2, cos, sin
from sympy.solvers import solve

#%%
# Create Symbols and Functions
x = Symbol('x', real=True)
y = Symbol('y', real=True)
th = Symbol('th', real=True)
a = Symbol('a', positive=True, real=True)
b = Symbol('b', positive=True, real=True)
u = Symbol('u', real=True)
rth = Function('rn', real=True)(th)
xn = Function('xn', real=True)(u)
yn = Function('yn', real=True)(u)
xn_val = Symbol('npnts.x', real=True)
yn_val = Symbol('npnts.y', real=True)
dxn_val = Symbol('nvecs.x', real=True)
dyn_val = Symbol('nvecs.y', real=True)
d2xn_val = Symbol('ncurs.x', real=True)
d2yn_val = Symbol('ncurs.y', real=True)
drdth_val = Symbol('drdth', real=True)
d2rdth2_val = Symbol('d2rdth2', real=True)
r_val = Symbol('r', real=True, positive=True)

#%%
# Create Ellipse Equation
eqn = 1 - (x/a)**2 - (y/b)**2
eqn = eqn.subs({x: r_val*cos(th), y: r_val*sin(th)}).simplify()
print(f'eqn = {eqn}\n')

r = solve(eqn, r_val)[1]
print(f'r = {r}\n')

sbs = {
    Derivative(rth, th): drdth_val,
    Derivative(rth, (th, 2)): d2rdth2_val
}

drdth = r.diff(th).together()
d2rdth2 = drdth.diff(th).together()

drdth = drdth.subs(sbs)
d2rdth2 = d2rdth2.subs(sbs)

print(f'drdth = {drdth}\n')
print(f'd2rdth2 = {d2rdth2}\n')

x = rth*cos(th).simplify()
y = rth*sin(th).simplify()

dx = x.diff(th).simplify()
dy = y.diff(th).simplify()

d2x = dx.diff(th).simplify()
d2y = dy.diff(th).simplify()

x = x.subs(sbs).simplify()
y = y.subs(sbs).simplify()

dx = dx.subs(sbs).simplify()
dy = dy.subs(sbs).simplify()

d2x = d2x.subs(sbs).simplify()
d2y = d2y.subs(sbs).simplify()

print(f'x = {x}\n')
print(f'y = {y}\n')

print(f'dx = {dx}\n')
print(f'dy = {dy}\n')

print(f'd2x = {d2x}\n')
print(f'd2y = {d2y}\n')

#%%
# Arc Tangent Function and Derivatives
thn = atan2(yn, xn)
print(f'thn = {thn}\n')

dthndu = thn.diff(u)
print(f'dthndu = {dthndu}\n')

d2thndu = dthndu.diff(u)
print(f'd2thndu = {d2thndu}\n')

sbs = {
    xn: xn_val,
    yn: yn_val,
    Derivative(xn, u): dxn_val,
    Derivative(yn, u): dyn_val,
    Derivative(xn, (u, 2)): d2xn_val,
    Derivative(yn, (u, 2)): d2yn_val
}

thn = thn.subs(sbs).simplify()
print(f'thn = {thn}\n')

dthndu = dthndu.subs(sbs).simplify()
print(f'dthndu = {dthndu}\n')

d2thndu = d2thndu.subs(sbs).simplify()
print(f'd2thndu = {d2thndu}\n')

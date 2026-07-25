#%%
# Import Dependencies
from sympy import Symbol, Expr
from sympy.solvers import solve
from numpy import linspace, ones
from matplotlib.pyplot import figure

#%%
# Create Symbolic Variables
x = Symbol('x', real=True)
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xc', real=True)
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
yc = Symbol('yc', real=True)
yd = Symbol('yd', real=True)
ye = Symbol('ye', real=True)
a = Symbol('a', real=True)
b = Symbol('b', real=True)
c = Symbol('c', real=True)
xbc = Symbol('xbc', real=True)
xca = Symbol('xca', real=True)
xab = Symbol('xab', real=True)
xbc2 = Symbol('xbc2', real=True)
xca2 = Symbol('xca2', real=True)
xab2 = Symbol('xab2', real=True)
xbc3 = Symbol('xbc3', real=True)
xca3 = Symbol('xca3', real=True)
xab3 = Symbol('xab3', real=True)
jac = Symbol('jac', real=True)
a0 = Symbol('a0', real=True)
a1 = Symbol('a1', real=True)
a2 = Symbol('a2', real=True)
b0 = Symbol('b0', real=True)
b1 = Symbol('b1', real=True)
b2 = Symbol('b2', real=True)
c0 = Symbol('c0', real=True)
c1 = Symbol('c1', real=True)
c2 = Symbol('c2', real=True)
Cdyc_a = Symbol('Cdyc_a', real=True)
Cdyc_b = Symbol('Cdyc_b', real=True)
Cdyc_c = Symbol('Cdyc_c', real=True)
Cdya_c = Symbol('Cdya_c', real=True)
Cdya_d = Symbol('Cdya_d', real=True)
Cdya_e = Symbol('Cdya_e', real=True)

#%%
# Create Symbolic Quadratic Function
y_expr: Expr = a*x**2 + b*x + c

eqna: Expr = y_expr.subs(x, xa) - ya
eqnb: Expr = y_expr.subs(x, xb) - yb
eqnc: Expr = y_expr.subs(x, xc) - yc

res: dict[Symbol, Expr] = solve([eqna, eqnb, eqnc], [a, b, c])
print(f'a = {res[a]}')
print(f'b = {res[b]}')
print(f'c = {res[c]}')

y_expr = y_expr.subs(res)
y_expr = y_expr.expand().simplify()
print(f'y = {y_expr}')

#%%
# Simplify Numerator and Denominator
y_num, y_den = y_expr.as_numer_denom()
y_num: Expr = y_num.collect([ya, yb, yc])
print(f'y_num = {y_num}')
print(f'y_den = {y_den}')

y_num_ya: Expr = y_num.coeff(ya)
y_num_yb: Expr = y_num.coeff(yb)
y_num_yc: Expr = y_num.coeff(yc)

y_num_ya = y_num_ya.collect([x])
y_num_yb = y_num_yb.collect([x])
y_num_yc = y_num_yc.collect([x])

print(f'y_num_ya = {y_num_ya}')
print(f'y_num_yb = {y_num_yb}')
print(f'y_num_yc = {y_num_yc}')

a_0 = y_num_ya.coeff(x, 0)
a_1 = y_num_ya.coeff(x, 1)
a_2 = y_num_ya.coeff(x, 2)
b_0 = y_num_yb.coeff(x, 0)
b_1 = y_num_yb.coeff(x, 1)
b_2 = y_num_yb.coeff(x, 2)
c_0 = y_num_yc.coeff(x, 0)
c_1 = y_num_yc.coeff(x, 1)
c_2 = y_num_yc.coeff(x, 2)

print(f'a_0 = {a_0}')
print(f'a_1 = {a_1}')
print(f'a_2 = {a_2}')
print(f'b_0 = {b_0}')
print(f'b_1 = {b_1}')
print(f'b_2 = {b_2}')
print(f'c_0 = {c_0}')
print(f'c_1 = {c_1}')
print(f'c_2 = {c_2}')

jac_expr: Expr = a_0 + b_0 + c_0
print(f'jac = {jac_expr}')

y_den_check: Expr = jac_expr - y_den
y_den_check = y_den_check.expand().simplify()

print(f'y_den_check = {y_den_check}')

a0_expr: Expr = a_0 / jac
a1_expr: Expr = a_1 / jac
a2_expr: Expr = a_2 / jac
b0_expr: Expr = b_0 / jac
b1_expr: Expr = b_1 / jac
b2_expr: Expr = b_2 / jac
c0_expr: Expr = c_0 / jac
c1_expr: Expr = c_1 / jac
c2_expr: Expr = c_2 / jac

print(f'a0 = {a0_expr}')
print(f'a1 = {a1_expr}')
print(f'a2 = {a2_expr}')
print(f'b0 = {b0_expr}')
print(f'b1 = {b1_expr}')
print(f'b2 = {b2_expr}')
print(f'c0 = {c0_expr}')
print(f'c1 = {c1_expr}')
print(f'c2 = {c2_expr}')

#%%
# New Quadratic Function
y: Expr = ya*(a0 + a1*x + a2*x**2) + yb*(b0 + b1*x + b2*x**2) + yc*(c0 + c1*x + c2*x**2)
print(f'y = {y}')

dydx: Expr = y.diff(x)
print(f'dydx = {dydx}')

dydx_a: Expr = dydx.subs(x, xa)
print(f'dydx_a = {dydx_a}')

dydx_b: Expr = dydx.subs(x, xb)
print(f'dydx_b = {dydx_b}')

dydx_c: Expr = dydx.subs(x, xc)
print(f'dydx_c = {dydx_c}')

#%%
# Checks
ya_check_a: Expr = a0_expr + a1_expr*xa + a2_expr*xa**2
ya_check_b: Expr = a0_expr + a1_expr*xb + a2_expr*xb**2
ya_check_c: Expr = a0_expr + a1_expr*xc + a2_expr*xc**2
yb_check_a: Expr = b0_expr + b1_expr*xa + b2_expr*xa**2
yb_check_b: Expr = b0_expr + b1_expr*xb + b2_expr*xb**2
yb_check_c: Expr = b0_expr + b1_expr*xc + b2_expr*xc**2
yc_check_a: Expr = c0_expr + c1_expr*xa + c2_expr*xa**2
yc_check_b: Expr = c0_expr + c1_expr*xb + c2_expr*xb**2
yc_check_c: Expr = c0_expr + c1_expr*xc + c2_expr*xc**2

ya_check_a = ya_check_a.subs(jac, jac_expr).expand().simplify()
ya_check_b = ya_check_b.subs(jac, jac_expr).expand().simplify()
ya_check_c = ya_check_c.subs(jac, jac_expr).expand().simplify()
yb_check_a = yb_check_a.subs(jac, jac_expr).expand().simplify()
yb_check_b = yb_check_b.subs(jac, jac_expr).expand().simplify()
yb_check_c = yb_check_c.subs(jac, jac_expr).expand().simplify()
yc_check_a = yc_check_a.subs(jac, jac_expr).expand().simplify()
yc_check_b = yc_check_b.subs(jac, jac_expr).expand().simplify()
yc_check_c = yc_check_c.subs(jac, jac_expr).expand().simplify()

print(f'ya_check_a = {ya_check_a}')
print(f'ya_check_b = {ya_check_b}')
print(f'ya_check_c = {ya_check_c}')
print(f'yb_check_a = {yb_check_a}')
print(f'yb_check_b = {yb_check_b}')
print(f'yb_check_c = {yb_check_c}')
print(f'yc_check_a = {yc_check_a}')
print(f'yc_check_b = {yc_check_b}')
print(f'yc_check_c = {yc_check_c}')

dya_check_a: Expr = a1_expr + 2*a2_expr*xa
dya_check_b: Expr = a1_expr + 2*a2_expr*xb
dya_check_c: Expr = a1_expr + 2*a2_expr*xc
dyb_check_a: Expr = b1_expr + 2*b2_expr*xa
dyb_check_b: Expr = b1_expr + 2*b2_expr*xb
dyb_check_c: Expr = b1_expr + 2*b2_expr*xc
dyc_check_a: Expr = c1_expr + 2*c2_expr*xa
dyc_check_b: Expr = c1_expr + 2*c2_expr*xb
dyc_check_c: Expr = c1_expr + 2*c2_expr*xc

dya_check_a = dya_check_a.subs(jac, jac_expr).expand().simplify()
dya_check_b = dya_check_b.subs(jac, jac_expr).expand().simplify()
dya_check_c = dya_check_c.subs(jac, jac_expr).expand().simplify()
dyb_check_a = dyb_check_a.subs(jac, jac_expr).expand().simplify()
dyb_check_b = dyb_check_b.subs(jac, jac_expr).expand().simplify()
dyb_check_c = dyb_check_c.subs(jac, jac_expr).expand().simplify()
dyc_check_a = dyc_check_a.subs(jac, jac_expr).expand().simplify()
dyc_check_b = dyc_check_b.subs(jac, jac_expr).expand().simplify()
dyc_check_c = dyc_check_c.subs(jac, jac_expr).expand().simplify()

print(f'dya_check_a = {dya_check_a}')
print(f'dya_check_b = {dya_check_b}')
print(f'dya_check_c = {dya_check_c}')
print(f'dyb_check_a = {dyb_check_a}')
print(f'dyb_check_b = {dyb_check_b}')
print(f'dyb_check_c = {dyb_check_c}')
print(f'dyc_check_a = {dyc_check_a}')
print(f'dyc_check_b = {dyc_check_b}')
print(f'dyc_check_c = {dyc_check_c}')

#%%
# Integral of Quadratic Function
int_y: Expr = y.integrate((x, xa, x))
int_y = int_y.expand().simplify()
print(f'int_y = {int_y}')

int_y_a: Expr = int_y.coeff(ya)
int_y_b: Expr = int_y.coeff(yb)
int_y_c: Expr = int_y.coeff(yc)

print(f'int_y_a = {int_y_a}')
print(f'int_y_b = {int_y_b}')
print(f'int_y_c = {int_y_c}')

int_y_a_ab = int_y_a.subs(x, xb)
int_y_a_ac = int_y_a.subs(x, xc)
int_y_a_bc = int_y_a_ac - int_y_a_ab

print(f'int_y_a_ab = {int_y_a_ab}')
# print(f'int_y_a_ac = {int_y_a_ac}')
print(f'int_y_a_bc = {int_y_a_bc}')

int_y_b_ab = int_y_b.subs(x, xb)
int_y_b_ac = int_y_b.subs(x, xc)
int_y_b_bc = int_y_b_ac - int_y_b_ab

print(f'int_y_b_ab = {int_y_b_ab}')
# print(f'int_y_b_ac = {int_y_b_ac}')
print(f'int_y_b_bc = {int_y_b_bc}')

int_y_c_ab = int_y_c.subs(x, xb)
int_y_c_ac = int_y_c.subs(x, xc)
int_y_c_bc = int_y_c_ac - int_y_c_ab

print(f'int_y_c_ab = {int_y_c_ab}')
# print(f'int_y_c_ac = {int_y_c_ac}')
print(f'int_y_c_bc = {int_y_c_bc}')

#%%
# Equation at x = xc, y = yc
dyc = Cdyc_a*ya + Cdyc_b*yb + Cdyc_c*yc
dya = Cdya_c*yc + Cdya_d*yd + Cdya_e*ye

eqn: Expr = dya - dyc
print(f'eqn = {eqn}')

eqn = eqn.collect([ya, yb, yc, yd, ye])
print(f'eqn = {eqn}')

# #%%
# # Plots
# xa_val = 0.0
# xb_val = 1.0
# xc_val = 1.8
# ya_val = 1.0
# yb_val = 2.0
# yc_val = 1.8

# jac_val = jac_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val}).evalf()
# a0_val = a0_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# a1_val = a1_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# a2_val = a2_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# b0_val = b0_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# b1_val = b1_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# b2_val = b2_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# c0_val = c0_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# c1_val = c1_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()
# c2_val = c2_expr.subs({xa: xa_val, xb: xb_val, xc: xc_val, jac: jac_val}).evalf()

# xv = linspace(xa_val, xc_val, 101)

# yv = (a0_val + a1_val*xv + a2_val*xv**2)*ya_val + \
#      (b0_val + b1_val*xv + b2_val*xv**2)*yb_val + \
#      (c0_val + c1_val*xv + c2_val*xv**2)*yc_val

# fig0 = figure(figsize=(12, 8))
# ax0 = fig0.gca()
# ax0.grid(True)
# ax0.plot(xv, yv, label='Quadratic Interpolation')
# _ = ax0.legend()

# dyv = (a1_val + 2*a2_val*xv)*ya_val + \
#      (b1_val + 2*b2_val*xv)*yb_val + \
#      (c1_val + 2*c2_val*xv)*yc_val

# fig1 = figure(figsize=(12, 8))
# ax1 = fig1.gca()
# ax1.grid(True)
# ax1.plot(xv, dyv, label='Derivative of Quadratic Interpolation')
# _ = ax1.legend()

# d2yv = (2*a2_val*ones(xv.shape))*ya_val + \
#        (2*b2_val*ones(xv.shape))*yb_val + \
#        (2*c2_val*ones(xv.shape))*yc_val

# fig2 = figure(figsize=(12, 8))
# ax2 = fig2.gca()
# ax2.grid(True)
# ax2.plot(xv, d2yv, label='Second Derivative of Quadratic Interpolation')
# _ = ax2.legend()

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
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
a = Symbol('a', real=True)
b = Symbol('b', real=True)
xab = Symbol('xab', real=True)
xab2 = Symbol('xab2', real=True)
a0 = Symbol('a0', real=True)
a1 = Symbol('a1', real=True)
a2 = Symbol('a2', real=True)
b0 = Symbol('b0', real=True)
b1 = Symbol('b1', real=True)
b2 = Symbol('b2', real=True)
c0 = Symbol('c0', real=True)
c1 = Symbol('c1', real=True)
c2 = Symbol('c2', real=True)
jac = Symbol('jac', real=True)
Cdyc_a = Symbol('Cdyc_a', real=True)
Cdyc_b = Symbol('Cdyc_b', real=True)
Cdyc_c = Symbol('Cdyc_c', real=True)
Cdya_c = Symbol('Cdya_c', real=True)
Cdya_d = Symbol('Cdya_d', real=True)
Cdya_e = Symbol('Cdya_e', real=True)

#%%
# Create Symbolic Quadratic Function
y_expr: Expr = a*x + b

eqna: Expr = y_expr.subs(x, xa) - ya
eqnb: Expr = y_expr.subs(x, xb) - yb

res: dict[Symbol, Expr] = solve([eqna, eqnb], [a, b])
print(f'a = {res[a]}')
print(f'b = {res[b]}')

y_expr = y_expr.subs(res)
y_expr = y_expr.expand().simplify()
print(f'y = {y_expr}')

#%%
# Simplify Numerator and Denominator
y_num, y_den = y_expr.as_numer_denom()
y_num: Expr = y_num.collect([ya, yb])
print(f'y_num = {y_num}')
print(f'y_den = {y_den}')

y_num_ya: Expr = y_num.coeff(ya)
y_num_yb: Expr = y_num.coeff(yb)

y_num_ya = y_num_ya.collect([x])
y_num_yb = y_num_yb.collect([x])

print(f'y_num_ya = {y_num_ya}')
print(f'y_num_yb = {y_num_yb}')

a_0 = y_num_ya.coeff(x, 0)
a_1 = y_num_ya.coeff(x, 1)
b_0 = y_num_yb.coeff(x, 0)
b_1 = y_num_yb.coeff(x, 1)

print(f'a_0 = {a_0}')
print(f'a_1 = {a_1}')
print(f'b_0 = {b_0}')
print(f'b_1 = {b_1}')

jac_expr: Expr = a_0 + b_0
print(f'jac = {jac_expr}')

y_den_check: Expr = jac_expr - y_den
y_den_check = y_den_check.expand().simplify()

print(f'y_den_check = {y_den_check}')

a0_expr: Expr = a_0 / jac
a1_expr: Expr = a_1 / jac
b0_expr: Expr = b_0 / jac
b1_expr: Expr = b_1 / jac

print(f'a0 = {a0_expr}')
print(f'a1 = {a1_expr}')
print(f'b0 = {b0_expr}')
print(f'b1 = {b1_expr}')

#%%
# New Quadratic Function
y: Expr = ya*(a0 + a1*x) + yb*(b0 + b1*x)
print(f'y = {y}')

dydx: Expr = y.diff(x)
print(f'dydx = {dydx}')

dydx_a: Expr = dydx.subs(x, xa)
print(f'dydx_a = {dydx_a}')

dydx_b: Expr = dydx.subs(x, xb)
print(f'dydx_b = {dydx_b}')

#%%
# Checks
ya_check_a: Expr = a0_expr + a1_expr*xa
ya_check_b: Expr = a0_expr + a1_expr*xb
yb_check_a: Expr = b0_expr + b1_expr*xa
yb_check_b: Expr = b0_expr + b1_expr*xb

ya_check_a = ya_check_a.subs(jac, jac_expr).expand().simplify()
ya_check_b = ya_check_b.subs(jac, jac_expr).expand().simplify()
yb_check_a = yb_check_a.subs(jac, jac_expr).expand().simplify()
yb_check_b = yb_check_b.subs(jac, jac_expr).expand().simplify()

print(f'ya_check_a = {ya_check_a}')
print(f'ya_check_b = {ya_check_b}')
print(f'yb_check_a = {yb_check_a}')
print(f'yb_check_b = {yb_check_b}')

dya_check_a: Expr = a1_expr
dya_check_b: Expr = a1_expr
dyb_check_a: Expr = b1_expr
dyb_check_b: Expr = b1_expr

dya_check_a = dya_check_a.subs(jac, jac_expr).expand().simplify()
dya_check_b = dya_check_b.subs(jac, jac_expr).expand().simplify()
dyb_check_a = dyb_check_a.subs(jac, jac_expr).expand().simplify()
dyb_check_b = dyb_check_b.subs(jac, jac_expr).expand().simplify()

print(f'dya_check_a = {dya_check_a}')
print(f'dya_check_b = {dya_check_b}')
print(f'dyb_check_a = {dyb_check_a}')
print(f'dyb_check_b = {dyb_check_b}')

#%%
# Integral of Linear Function
int_y: Expr = y.integrate((x, xa, x))
int_y = int_y.expand().simplify()
print(f'int_y = {int_y}')

int_y_a: Expr = int_y.coeff(ya)
int_y_b: Expr = int_y.coeff(yb)

print(f'int_y_a = {int_y_a}')
print(f'int_y_b = {int_y_b}')

#%%
# Import Dependencies
from numpy import empty
from pygeom.geom2d import NurbsCurve2D, Vector2D
from sympy import Symbol

#%%
# Symbolic Degree 3 Bernstein Polynomials
t = Symbol('t', real=True)
xa = Symbol('xa', real=True)
xb = Symbol('xb', real=True)
xc = Symbol('xc', real=True)
xd = Symbol('xd', real=True)
ya = Symbol('ya', real=True)
yb = Symbol('yb', real=True)
yc = Symbol('yc', real=True)
yd = Symbol('yd', real=True)
wb = Symbol('wc', real=True)
wc = Symbol('wd', real=True)
dxadt = Symbol('dxadt', real=True)
dyadt = Symbol('dyadt', real=True)
dxbdt = Symbol('dxbdt', real=True)
dybdt = Symbol('dybdt', real=True)
u = Symbol('u', real=True)

#%%
# Compute the Numerator and Denominator of the Rational Bernstein Polynomial
x_arr = empty(4, dtype=object)
y_arr = empty(4, dtype=object)
w_arr = empty(4, dtype=object)
x_arr[0] = xa
x_arr[1] = xb
x_arr[2] = xc
x_arr[3] = xd
y_arr[0] = ya
y_arr[1] = yb
y_arr[2] = yc
y_arr[3] = yd
w_arr[0] = 1
w_arr[1] = wb
w_arr[2] = wc
w_arr[3] = 1

ctlpnts = Vector2D(x_arr, y_arr)

nurbscurve = NurbsCurve2D(ctlpnts, weights=w_arr)

pnt_u0 = nurbscurve.evaluate_points_at_t(0)
pnt_u1 = nurbscurve.evaluate_points_at_t(1)

print(f'pnt_u0 = {pnt_u0}\n')
print(f'pnt_u1 = {pnt_u1}\n')

pnt_u = nurbscurve.evaluate_points_at_t(u)
print(f'pnt_u = {pnt_u}\n')

# numerx = 0
# numery = 0
# denom = 0
# for i, (x, y, w, poly) in enumerate(zip(xs, ys, ws, polys)):
#     numerx += w*x*poly
#     numery += w*y*poly
#     denom += w*poly

# npolyx = numerx/denom
# npolyy = numery/denom

# print(f'npolyx = {npolyx}\n')
# print(f'npolyy = {npolyy}\n')

# dnpolyxdt = npolyx.diff(t).expand().simplify()
# dnpolyydt = npolyy.diff(t).expand().simplify()

# print(f'dnpolyxdt = {dnpolyxdt}\n')
# print(f'dnpolyydt = {dnpolyydt}\n')

# eqndxa = dxadt - dnpolyxdt.subs(t, 0).expand()
# eqndya = dyadt - dnpolyydt.subs(t, 0).expand()
# eqndxb = dxbdt - dnpolyxdt.subs(t, 1).expand()
# eqndyb = dybdt - dnpolyydt.subs(t, 1).expand()

# print(f'eqndxa = {eqndxa}\n')
# print(f'eqndya = {eqndya}\n')
# print(f'eqndxb = {eqndxb}\n')
# print(f'eqndyb = {eqndyb}\n')

# eqndxb2 = eqndxb.subs(wd, wc)
# eqndyb2 = eqndyb.subs(wd, wc)

# print(f'eqndxb2 = {eqndxb2}\n')
# print(f'eqndyb2 = {eqndyb2}\n')

# # print(f'numerx = {numerx}\n')
# # print(f'numery = {numery}\n')

# # print(f'denomx = {denomx}\n')
# # print(f'denomy = {denomy}\n')

# numerx = Poly(numerx, t)
# numery = Poly(numery, t)
# denom = Poly(denom, t)

# print(f'numerx = {numerx}\n')
# print(f'numery = {numery}\n')

# print(f'denom = {denom}\n')

# print(f'numerx.coeffs() = {numerx.coeffs()}\n')
# print(f'numery.coeffs() = {numery.coeffs()}\n')

# print(f'denom.coeffs() = {denom.coeffs()}\n')

# d2npolyxdt2 = dnpolyxdt.diff(t).expand()
# d2npolyydt2 = dnpolyydt.diff(t).expand()

# print(f'd2npolyxdt2 = {d2npolyxdt2}\n')
# print(f'd2npolyydt2 = {d2npolyydt2}\n')

# kappa = (dnpolyxdt*d2npolyydt2 - dnpolyydt*d2npolyxdt2)/(dnpolyxdt**2 + dnpolyydt**2)**Rational(3, 2)

# print(f'kappa = {kappa}\n')

# numerkappa, denomkappa = fraction(kappa)

# print(f'numerkappa = {numerkappa}\n')
# print(f'denomkappa = {denomkappa}\n')

# #%%
# # Compute the Numerator and Denominator of the Rational Bernstein Polynomial
# n = 2
# polys = symbolic_bernstein_polynomials(n, t)
# xs = [xa, xc, xb]
# ys = [ya, yc, yb]
# ws = [1, 1/sqrt(2), 1]

# numerx = 0
# numery = 0
# denom = 0
# for i, (x, y, w, poly) in enumerate(zip(xs, ys, ws, polys)):
#     numerx += w*x*poly
#     numery += w*y*poly
#     denom += w*poly

# npolyx = numerx/denom
# npolyy = numery/denom

# print(f'npolyx = {npolyx}\n')
# print(f'npolyy = {npolyy}\n')

# dnpolyxdt = npolyx.diff(t).expand()
# dnpolyydt = npolyy.diff(t).expand()

# print(f'dnpolyxdt = {dnpolyxdt}\n')
# print(f'dnpolyydt = {dnpolyydt}\n')

# eqndxa = dxadt - dnpolyxdt.subs(t, 0).expand()
# eqndya = dyadt - dnpolyydt.subs(t, 0).expand()
# eqndxb = dxbdt - dnpolyxdt.subs(t, 1).expand()
# eqndyb = dybdt - dnpolyydt.subs(t, 1).expand()

# print(f'eqndxa = {eqndxa}\n')
# print(f'eqndya = {eqndya}\n')
# print(f'eqndxb = {eqndxb}\n')
# print(f'eqndyb = {eqndyb}\n')

# eqndxb2 = eqndxb.subs(wd, wc)
# eqndyb2 = eqndyb.subs(wd, wc)

# print(f'eqndxb2 = {eqndxb2}\n')
# print(f'eqndyb2 = {eqndyb2}\n')

# # print(f'numerx = {numerx}\n')
# # print(f'numery = {numery}\n')

# # print(f'denomx = {denomx}\n')
# # print(f'denomy = {denomy}\n')

# numerx = Poly(numerx, t)
# numery = Poly(numery, t)
# denom = Poly(denom, t)

# print(f'numerx = {numerx}\n')
# print(f'numery = {numery}\n')

# print(f'denom = {denom}\n')

# print(f'numerx.coeffs() = {numerx.coeffs()}\n')
# print(f'numery.coeffs() = {numery.coeffs()}\n')

# print(f'denom.coeffs() = {denom.coeffs()}\n')

# d2npolyxdt2 = dnpolyxdt.diff(t).expand()
# d2npolyydt2 = dnpolyydt.diff(t).expand()

# print(f'd2npolyxdt2 = {d2npolyxdt2}\n')
# print(f'd2npolyydt2 = {d2npolyydt2}\n')

# kappa = (dnpolyxdt*d2npolyydt2 - dnpolyydt*d2npolyxdt2)/(dnpolyxdt**2 + dnpolyydt**2)**Rational(3, 2)

# print(f'kappa = {kappa}\n')

# numerkappa, denomkappa = fraction(kappa)

# print(f'numerkappa = {numerkappa}\n')
# print(f'denomkappa = {denomkappa}\n')

# numerkappa = numerkappa.expand()
# denomkappa = denomkappa.expand()

# print(f'numerkappa = {numerkappa}\n')
# print(f'denomkappa = {denomkappa}\n')

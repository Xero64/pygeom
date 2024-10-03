#%%
# Import Dependencies
from pygeom.tools.bernstein import symbolic_bernstein_polynomials
from sympy import Poly, Rational, Symbol, fraction, sqrt

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
wc = Symbol('wc', real=True)
wd = Symbol('wd', real=True)
dxadt = Symbol('dxadt', real=True)
dyadt = Symbol('dyadt', real=True)
dxbdt = Symbol('dxbdt', real=True)
dybdt = Symbol('dybdt', real=True)

#%%
# Compute the Numerator and Denominator of the Rational Bernstein Polynomial
n = 3
polys = symbolic_bernstein_polynomials(n, t)
xs = [xa, xc, xd, xb]
ys = [ya, yc, yd, yb]
ws = [1, wc, wd, 1]

numerx = 0
numery = 0
denom = 0
for i, (x, y, w, poly) in enumerate(zip(xs, ys, ws, polys)):
    numerx += w*x*poly
    numery += w*y*poly
    denom += w*poly

npolyx = numerx/denom
npolyy = numery/denom

print(f'npolyx = {npolyx}\n')
print(f'npolyy = {npolyy}\n')

dnpolyxdt = npolyx.diff(t).expand().simplify()
dnpolyydt = npolyy.diff(t).expand().simplify()

print(f'dnpolyxdt = {dnpolyxdt}\n')
print(f'dnpolyydt = {dnpolyydt}\n')

eqndxa = dxadt - dnpolyxdt.subs(t, 0).expand()
eqndya = dyadt - dnpolyydt.subs(t, 0).expand()
eqndxb = dxbdt - dnpolyxdt.subs(t, 1).expand()
eqndyb = dybdt - dnpolyydt.subs(t, 1).expand()

print(f'eqndxa = {eqndxa}\n')
print(f'eqndya = {eqndya}\n')
print(f'eqndxb = {eqndxb}\n')
print(f'eqndyb = {eqndyb}\n')

eqndxb2 = eqndxb.subs(wd, wc)
eqndyb2 = eqndyb.subs(wd, wc)

print(f'eqndxb2 = {eqndxb2}\n')
print(f'eqndyb2 = {eqndyb2}\n')

# print(f'numerx = {numerx}\n')
# print(f'numery = {numery}\n')

# print(f'denomx = {denomx}\n')
# print(f'denomy = {denomy}\n')

numerx = Poly(numerx, t)
numery = Poly(numery, t)
denom = Poly(denom, t)

print(f'numerx = {numerx}\n')
print(f'numery = {numery}\n')

print(f'denom = {denom}\n')

print(f'numerx.coeffs() = {numerx.coeffs()}\n')
print(f'numery.coeffs() = {numery.coeffs()}\n')

print(f'denom.coeffs() = {denom.coeffs()}\n')

d2npolyxdt2 = dnpolyxdt.diff(t).expand()
d2npolyydt2 = dnpolyydt.diff(t).expand()

print(f'd2npolyxdt2 = {d2npolyxdt2}\n')
print(f'd2npolyydt2 = {d2npolyydt2}\n')

kappa = (dnpolyxdt*d2npolyydt2 - dnpolyydt*d2npolyxdt2)/(dnpolyxdt**2 + dnpolyydt**2)**Rational(3, 2)

print(f'kappa = {kappa}\n')

numerkappa, denomkappa = fraction(kappa)

print(f'numerkappa = {numerkappa}\n')
print(f'denomkappa = {denomkappa}\n')

#%%
# Compute the Numerator and Denominator of the Rational Bernstein Polynomial
n = 2
polys = symbolic_bernstein_polynomials(n, t)
xs = [xa, xc, xb]
ys = [ya, yc, yb]
ws = [1, 1/sqrt(2), 1]

numerx = 0
numery = 0
denom = 0
for i, (x, y, w, poly) in enumerate(zip(xs, ys, ws, polys)):
    numerx += w*x*poly
    numery += w*y*poly
    denom += w*poly

npolyx = numerx/denom
npolyy = numery/denom

print(f'npolyx = {npolyx}\n')
print(f'npolyy = {npolyy}\n')

dnpolyxdt = npolyx.diff(t).expand()
dnpolyydt = npolyy.diff(t).expand()

print(f'dnpolyxdt = {dnpolyxdt}\n')
print(f'dnpolyydt = {dnpolyydt}\n')

eqndxa = dxadt - dnpolyxdt.subs(t, 0).expand()
eqndya = dyadt - dnpolyydt.subs(t, 0).expand()
eqndxb = dxbdt - dnpolyxdt.subs(t, 1).expand()
eqndyb = dybdt - dnpolyydt.subs(t, 1).expand()

print(f'eqndxa = {eqndxa}\n')
print(f'eqndya = {eqndya}\n')
print(f'eqndxb = {eqndxb}\n')
print(f'eqndyb = {eqndyb}\n')

eqndxb2 = eqndxb.subs(wd, wc)
eqndyb2 = eqndyb.subs(wd, wc)

print(f'eqndxb2 = {eqndxb2}\n')
print(f'eqndyb2 = {eqndyb2}\n')

# print(f'numerx = {numerx}\n')
# print(f'numery = {numery}\n')

# print(f'denomx = {denomx}\n')
# print(f'denomy = {denomy}\n')

numerx = Poly(numerx, t)
numery = Poly(numery, t)
denom = Poly(denom, t)

print(f'numerx = {numerx}\n')
print(f'numery = {numery}\n')

print(f'denom = {denom}\n')

print(f'numerx.coeffs() = {numerx.coeffs()}\n')
print(f'numery.coeffs() = {numery.coeffs()}\n')

print(f'denom.coeffs() = {denom.coeffs()}\n')

d2npolyxdt2 = dnpolyxdt.diff(t).expand()
d2npolyydt2 = dnpolyydt.diff(t).expand()

print(f'd2npolyxdt2 = {d2npolyxdt2}\n')
print(f'd2npolyydt2 = {d2npolyydt2}\n')

kappa = (dnpolyxdt*d2npolyydt2 - dnpolyydt*d2npolyxdt2)/(dnpolyxdt**2 + dnpolyydt**2)**Rational(3, 2)

print(f'kappa = {kappa}\n')

numerkappa, denomkappa = fraction(kappa)

print(f'numerkappa = {numerkappa}\n')
print(f'denomkappa = {denomkappa}\n')

numerkappa = numerkappa.expand()
denomkappa = denomkappa.expand()

print(f'numerkappa = {numerkappa}\n')
print(f'denomkappa = {denomkappa}\n')

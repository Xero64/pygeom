#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, divide, zeros
from numpy.linalg import solve
from pygeom.interp1d.quadraticinterp import (QuadraticCentreInterp,
                                             QuadraticCentreInterpSolver)
from scipy.interpolate import interp1d

#%%
# Create Arrays
x = asarray([1.0, 2.2, 3.0, 4.1, 5.0])
yc = asarray([-1.24, -0.05, -1.97, -2.7])

#%%
# Create Quadratic Interpolator
qis = QuadraticCentreInterpSolver(x)

dymat = qis.zmatop@yc

dya = dymat[0]
dyb = dymat[1]

qci = QuadraticCentreInterp(x, yc, dya, dyb)

y = qci.y

spl = interp1d(x, qci.y, kind='quadratic')

print(f'qci.emat = \n{qci.emat}\n')
print(f'qci.fmat = \n{qci.fmat}\n')
print(f'qci.gmat = \n{qci.gmat}\n')
print(f'qci.y = \n{qci.y}\n')
print(f'qci.dydx = \n{qci.dydx}\n')

Gaa = qci.gmat[0, -2]
Gba = qci.gmat[0, -1]
Gca = qci.gmat[0, :-2]

Gab = qci.gmat[-1, -2]
Gbb = qci.gmat[-1, -1]
Gcb = qci.gmat[-1, :-2]

amat = zeros((2, 2))
bmat = zeros((2, qci.num-1))

amat[0, 0] = Gaa - Gab
amat[0, 1] = Gba - Gbb
bmat[0, :] = Gca - Gcb

amat[1, 0] = 1.0
amat[1, 1] = 1.0

zmat = -solve(amat, bmat)

print(f'amat = \n{amat}\n')
print(f'bmat = \n{bmat}\n')
print(f'zmat = \n{zmat}\n')

#%%
# Plots
num = 400
xv = asarray([min(x)+i/num*(max(x)-min(x)) for i in range(0, num+1)])
yvc = qci.quadratic_interpolation_array(xv)
yvs = spl(xv)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, y, label='Data Points')
_ = ax.scatter(qci.xc, qci.yc, label='Centre Points')
_ = ax.plot(xv, yvc, label='Quadratic Interpolator')
_ = ax.plot(xv, yvs, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Derivative Plot
xm = (xv[1:]+xv[:-1])/2
dyvc = qci.quadratic_first_derivative_array(xv)
dyvs = divide(yvs[1:]-yvs[:-1], xv[1:]-xv[:-1])

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.plot(xv, dyvc, label='Quadratic Interpolator')
_ = ax.plot(xm, dyvs, '--', label='SciPy - Quadratic')
_ = ax.legend()

# #%%
# # Integral Plot
# iyvc = qci.quadratic_interpolation_integral_array(xv)

# fig = figure(figsize=(10, 8))
# ax = fig.gca()
# ax.grid(True)
# _ = ax.plot(xv, iyvc, label='Quadratic Interpolator')
# _ = ax.legend()

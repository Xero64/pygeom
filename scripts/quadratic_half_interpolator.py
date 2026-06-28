#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, divide, linspace
from pygeom.tools.solvers import quadratic_pspline_fit_solver
from scipy.interpolate import interp1d

#%%
# Create Arrays
x = asarray([1.0, 2.2, 3.0, 4.1, 5.0, 6.0])
y = asarray([-4.0, 0.0, -0.8, -3.9, -2.6, -7.0])

#%%
# Create Quadratic Interpolator
spl = interp1d(x, y, kind='quadratic')

farr, garr, harr = quadratic_pspline_fit_solver(x)

xm  = (x[1:-2] + x[2:-1]) / 2.0
ym = farr @ y
dy = harr @ y
d2y = garr @ y

#%%
# Plots
num = 400
xv = linspace(x.min(), x.max(), num + 1)
yv = spl(xv)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, y, label='Data Points')
_ = ax.scatter(xm, ym, label='Quadratic Interpolator')
_ = ax.plot(xv, yv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Derivative Plot
x1 = (xv[1:] + xv[:-1]) / 2.0
dyv = divide(yv[1:] - yv[:-1], xv[1:] - xv[:-1])

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, dy, label='Quadratic Interpolator')
_ = ax.plot(x1, dyv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Derivative Plot
x2 = xv[1:-1]
d2yv = divide(dyv[1:] - dyv[:-1], x1[1:] - x1[:-1])

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, d2y, label='Quadratic Interpolator')
_ = ax.plot(x2, d2yv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, divide, full, linspace, set_printoptions
from pygeom.geom1d.quadraticspline1d import QuadraticSpline1DSolver
from scipy.integrate import cumulative_simpson
from scipy.interpolate import interp1d

set_printoptions(precision=8, suppress=True)

#%%
# Create Arrays
x = asarray([1.0, 2.2, 3.0, 4.1, 5.0, 6.0])
y = asarray([-4.0, 0.0, -0.8, -3.9, -2.6, -7.0])

#%%
# Create Quadratic Interpolator
spl = interp1d(x, y, kind='quadratic')

qss = QuadraticSpline1DSolver(x)

xm  = (x[1:-2] + x[2:-1]) / 2.0
ym = qss.marr @ y
dy = qss.harr @ y
d2y = qss.garr @ y
Iy = qss.iarr @ y

xc = asarray([1.8, 2.8, 3.7, 5.8])

qs = qss.to_quadratic_spline_1d(y)

#%%
# Plots
num = 4000
xv = linspace(x.min(), x.max(), num + 1)
yv = spl(xv)
rarrc = qss.evaluate_points_array_at_t(xc)
yc = rarrc @ y
yf = qs.evaluate_points_at_t(xc)

yi = -3.0
xi = qs.find_intercepts(yi)
yi = full(xi.shape, yi)

xp = qs.evaluate_t(5)
yp = qs.evaluate_points_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, y, label='Data Points')
_ = ax.scatter(xm, ym, label='Quadratic Interpolator')
_ = ax.scatter(xc, yc, label='Check Points')
_ = ax.scatter(xc, yf, label='Check Points - Final')
_ = ax.scatter(xi, yi, label='Intercepts')
_ = ax.scatter(xp, yp, marker='x', label='Evaluate Points')
_ = ax.plot(xv, yv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Derivative Plot
x1 = (xv[1:] + xv[:-1]) / 2.0
dyv = divide(yv[1:] - yv[:-1], xv[1:] - xv[:-1])
drarrc = qss.evaluate_first_derivatives_array_at_t(xc)
dyc = drarrc @ y
dyf = qs.evaluate_first_derivatives_at_t(xc)
dyp = qs.evaluate_first_derivatives_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, dy, label='Quadratic Interpolator')
_ = ax.scatter(xc, dyc, label='Check Derivative Points')
_ = ax.scatter(xc, dyf, label='Check Derivative Points - Final')
_ = ax.scatter(xp, dyp, marker='x', label='Evaluate Derivative Points')
_ = ax.plot(x1, dyv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Derivative Plot
x2 = xv[1:-1]
d2yv = divide(dyv[1:] - dyv[:-1], x1[1:] - x1[:-1])
d2yf = qs.evaluate_second_derivatives_at_t(xc)
d2yp = qs.evaluate_second_derivatives_at_t(xp)
xm  = qs.sall[::2]
d2ym = qs.evaluate_second_derivatives_at_t(xm)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, d2y, label='Quadratic Interpolator')
_ = ax.scatter(xc, d2yf, label='Check Second Derivative Points - Final')
_ = ax.scatter(xp, d2yp, marker='x', label='Evaluate Second Derivative Points')
_ = ax.scatter(xm, d2ym, label='Quadratic Interpolator - Midpoints')
_ = ax.plot(x2, d2yv, '--', label='SciPy - Quadratic')
_ = ax.legend()

#%%
# Integrals
int_yv = cumulative_simpson(yv, x=xv, initial=0.0)
Iarrc = qss.evaluate_integral_array_at_t(xc)
Iyc = Iarrc @ y
Iyf = qs.evaluate_integrals_at_t(xc)
Iyp = qs.evaluate_integrals_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, Iy, label='Quadratic Interpolator')
_ = ax.scatter(xc, Iyc, label='Check Integral Points')
_ = ax.scatter(xc, Iyf, label='Check Integral Points - Final')
_ = ax.scatter(xp, Iyp, marker='x', label='Evaluate Integral Points')
_ = ax.plot(xv, int_yv, '--', label='SciPy - Quadratic')
_ = ax.legend()

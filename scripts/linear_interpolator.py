#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, divide, full, linspace, set_printoptions
from pygeom.geom1d.linearspline1d import LinearSpline1DSolver
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

set_printoptions(precision=8, suppress=True)

#%%
# Create Arrays
x = asarray([1.0, 2.2, 3.0, 4.1, 5.0, 6.0])
y = asarray([-4.0, 0.0, -0.8, -3.9, -2.6, -7.0])

xc = asarray([1.8, 2.8, 3.7, 5.8])

#%%
# Create Quadratic Interpolator
spl = interp1d(x, y, kind='linear')

lss = LinearSpline1DSolver(x)

ls = lss.to_linear_spline_1d(y)

#%%
# Plots
num = 4000
xv = linspace(x.min(), x.max(), num + 1)
yv = spl(xv)
rarrc = lss.evaluate_points_array_at_t(xc)
yc = rarrc @ y
yf = ls.evaluate_points_at_t(xc)

yi = -3.0
xi = ls.find_intercepts(yi)
yi = full(xi.shape, yi)

xp = ls.evaluate_t(5)
yp = ls.evaluate_points_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, y, label='Data Points')
_ = ax.scatter(xc, yc, label='Check Points')
_ = ax.scatter(xc, yf, label='Check Points - Final')
_ = ax.scatter(xi, yi, label='Intercepts')
_ = ax.scatter(xp, yp, marker='x', label='Evaluate Points')
_ = ax.plot(xv, yv, '--', label='SciPy - Linear')
_ = ax.legend()

#%%
# First Derivative Plots
x1 = (xv[1:] + xv[:-1]) / 2.0
dyv = divide(yv[1:] - yv[:-1], xv[1:] - xv[:-1])
# drarrc = lss.evaluate_first_derivatives_array_at_t(xc)
# dyc = drarrc @ y
dyf = ls.evaluate_first_derivatives_at_t(xc)
dyp = ls.evaluate_first_derivatives_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
# _ = ax.scatter(x, dy, label='Linear Interpolator')
# _ = ax.scatter(xc, dyc, label='Check Derivative Points')
_ = ax.scatter(xc, dyf, label='Check Derivative Points - Final')
_ = ax.scatter(xp, dyp, marker='x', label='Evaluate Derivative Points')
_ = ax.plot(x1, dyv, '--', label='SciPy - Quadratic')
_ = ax.legend()

# #%%
# # Second Derivative Plots
# x2 = xv[1:-1]
# d2yv = divide(dyv[1:] - dyv[:-1], x1[1:] - x1[:-1])
# d2yf = ls.evaluate_second_derivatives_at_t(xc)
# d2yp = ls.evaluate_second_derivatives_at_t(xp)
# xm  = ls.sall[::2]
# d2ym = ls.evaluate_second_derivatives_at_t(xm)

# fig = figure(figsize=(10, 8))
# ax = fig.gca()
# ax.grid(True)
# _ = ax.scatter(x, d2y, label='Quadratic Interpolator')
# _ = ax.scatter(xc, d2yf, label='Check Second Derivative Points - Final')
# _ = ax.scatter(xp, d2yp, marker='x', label='Evaluate Second Derivative Points')
# _ = ax.scatter(xm, d2ym, label='Quadratic Interpolator - Midpoints')
# _ = ax.plot(x2, d2yv, '--', label='SciPy - Quadratic')
# _ = ax.legend()

#%%
# Integral Plots
int_yv = cumulative_trapezoid(yv, x=xv, initial=0.0)
# Iarrc = lss.evaluate_integral_array_at_t(xc)
# Iyc = Iarrc @ y
Iyf = ls.evaluate_integrals_at_t(xc)
Iyp = ls.evaluate_integrals_at_t(xp)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
# _ = ax.scatter(x, Iy, label='Quadratic Interpolator')
# _ = ax.scatter(xc, Iyc, label='Check Integral Points')
_ = ax.scatter(xc, Iyf, label='Check Integral Points - Final')
_ = ax.scatter(xp, Iyp, marker='x', label='Evaluate Integral Points')
_ = ax.plot(xv, int_yv, '--', label='SciPy - Linear')
_ = ax.legend()

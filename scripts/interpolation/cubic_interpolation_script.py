#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import asarray, linspace, concatenate, set_printoptions
from pygeom.tools.solvers import cubic_pspline_fit_solver
from scipy.interpolate import CubicSpline

set_printoptions(suppress=True)

#%%
# Create Input Arrays
x = asarray([1.0, 2.2, 3.0, 4.1, 5.0])
y = asarray([5.0, 2.6, 3.5, 1.2, 4.3])

# bc_type = 'clamped'
bc_type = ((1, 2.0), (1, -3.0))

if bc_type == 'periodic':
    y[-1] = y[0]

if bc_type == 'periodic' or bc_type == 'quadratic':
    rmat = cubic_pspline_fit_solver(x, bc_type=bc_type)
    d2y = rmat@y
    bc_type = ((2, d2y[0]), (2, d2y[-1]))

spl = CubicSpline(x, y, bc_type=bc_type)

if isinstance(bc_type, tuple):
    bc_type = (bc_type[0][0], bc_type[1][0])
    rval = concatenate((y, [spl(x[0], bc_type[0])], [spl(x[-1], bc_type[1])]))
else:
    rval = y

rmat = cubic_pspline_fit_solver(x, bc_type=bc_type)

d2y = rmat@rval

print(f'y = {y}\n')
print(f'd2y = {d2y}\n')

#%%
# Plots
num = 400
xv = linspace(x.min(), x.max(), num)
yvs = spl(xv)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, y, label='Data Points')
_ = ax.plot(xv, yvs, '--', label='SciPy - Cubic')
_ = ax.legend()

dyvs = spl(xv, 1)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.plot(xv, dyvs, '--', label='SciPy - Cubic')
_ = ax.legend()

d2yvs = spl(xv, 2)

fig = figure(figsize=(10, 8))
ax = fig.gca()
ax.grid(True)
_ = ax.scatter(x, d2y, label='Cubic Interpolator - Cubic')
_ = ax.plot(xv, d2yvs, '--', label='SciPy - Cubic')
_ = ax.legend()

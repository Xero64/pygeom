#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import (arange, asarray, concatenate, cos, diag, hstack, linspace,
                   pi, sin, vstack, zeros)
from numpy.linalg import lstsq, norm
from pygeom.geom2d import BSplineCurve2D, Vector2D
from pygeom.tools.solvers import solve_clsq

#%%
# Define the control points
num = 13
radius = 2.0
degree = 3

numctl = 13

t = linspace(0.0, 1.0, num)
th = 2*pi*t

x_tgt = radius*cos(th)
y_tgt = radius*sin(th)

ctlpnts_linear = Vector2D.from_iter_xy(x_tgt[0::3], y_tgt[0::3])
bspline_linear = BSplineCurve2D(ctlpnts_linear, degree=1)
t_ctl = linspace(0.0, 1.0, numctl)

x_ctl = bspline_linear.evaluate_points_at_t(t_ctl).x
y_ctl = bspline_linear.evaluate_points_at_t(t_ctl).y
ctlpnts = Vector2D.from_iter_xy(x_ctl, y_ctl)

bspline = BSplineCurve2D(ctlpnts, degree=degree)
print(bspline)

#%%
# Generate the curve
nspl = 100
pnts = bspline.evaluate_points(nspl)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
ax.scatter(x_tgt, y_tgt, color='red', label='Target Points')
ax.plot(pnts.x, pnts.y, color='blue', label='BSpline Curve')
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y, color='green',
           label='Control Points')
_ = ax.legend()

#%%
# Fit the curve to the target points
ind_c = set(range(0, num, 3))
ind_f = set(range(0, num)) - ind_c

num_f = len(ind_f)

ind_c = list(ind_c)
print(f'ind_c = {ind_c}\n')

ind_f = list(ind_f)
print(f'ind_f = {ind_f}\n')

t_c = t[ind_c]
print(f't_c = {t_c}\n')

t_f = t[ind_f]
print(f't_f = {t_f}\n')

ind_f = asarray(ind_f)
print(f'ind_f = {ind_f}\n')

ind_c = asarray(ind_c)
print(f'ind_c = {ind_c}\n')

ind_f_xy = concatenate((ind_f, ind_f + num))
print(f'ind_f_xy = {ind_f_xy}\n')

ind_c_xy = concatenate((ind_c, ind_c + num))
print(f'ind_c_xy = {ind_c_xy}\n')

max_iter = 10
count = 0

while True:

    print(f'Iteration {count}\n')

    x, y = bspline.evaluate_points_at_t(t_f).to_xy()
    dx = x - x_tgt[ind_f]
    dy = y - y_tgt[ind_f]

    f = concatenate((dx, dy))
    print(f'f = {f}\n')

    norm_f = norm(f)
    print(f'norm_f = {norm_f}\n')

    if norm_f < 1e-6:
        print('Converged')
        break

    Nt = bspline.basis_functions(t_f).transpose()[:, ind_f]

    dxdX = Nt
    # print(f'dxdX = \n{dxdX}\n')

    dydX = zeros(dxdX.shape)
    # print(f'dydX = \n{dydX}\n')

    dydY = Nt
    # print(f'dydY = \n{dydY}\n')

    dxdY = zeros(dydY.shape)
    # print(f'dxdY = \n{dxdY}\n')

    dxdt, dydt = bspline.evaluate_first_derivatives_at_t(t_f).to_xy()
    dxdt = diag(dxdt)
    dydt = diag(dydt)
    print(f'dxdt = {dxdt}\n')
    print(f'dydt = {dydt}\n')

    # dxdt_f = zeros((num, num_f))
    # dxdt_f[ind_f, :] = diag(dxdt)
    # dydt_f = zeros((num, num_f))
    # dydt_f[ind_f, :] = diag(dydt)

    dfdv = vstack((hstack((dxdX, dxdY, dxdt)),
                   hstack((dydX, dydY, dydt))))
    # print(f'dfdv = \n{dfdv}\n')

    # Amat = dfdv[ind_f_xy, :]
    # Bmat = f[ind_f_xy].reshape(-1, 1)
    # Cmat = dfdv[ind_c_xy, :]
    # Dmat = f[ind_c_xy].reshape(-1, 1)
    # Cmat[:, 2*numctl:] = 0.0

    # print(f'Amat = \n{Amat}\n')
    # print(f'Bmat = \n{Bmat}\n')
    # print(f'Cmat = \n{Cmat}\n')
    # print(f'Dmat = \n{Dmat}\n')

    # dv, dl = solve_clsq(Amat, Bmat, Cmat, Dmat)
    # print(f'dv = {dv}\n')
    # print(f'dl = {dl}\n')

    dv, residuals, rank_dfdv, s_vals = lstsq(dfdv, f, rcond=None)
    print(f'dv = {dv}\n')

    norm_dv = norm(dv)
    print(f'norm_dv = {norm_dv}\n')

    if norm_dv < 1e-6:
        print('Converged')
        break

    dX = zeros(numctl)
    dY = zeros(numctl)
    dX[ind_f] = dv[0:num_f]
    dY[ind_f] = dv[num_f:2*num_f]

    bspline.ctlpnts.x -= dX
    bspline.ctlpnts.y -= dY
    bspline.reset()

    t[ind_f] -= dv[2*num_f:]

    print(f'X = {bspline.ctlpnts.x}\n')
    print(f'Y = {bspline.ctlpnts.y}\n')
    print(f't = {t}\n')

    count += 1
    if count > max_iter:
        print('Max Iterations Reached')
        break

#%%
# Generate the curve
nspl = 100
pnts = bspline.evaluate_points(nspl)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
ax.scatter(x_tgt, y_tgt, color='red', label='Target Points')
ax.plot(pnts.x, pnts.y, color='blue', label='BSpline Curve')
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y, color='green',
           label='Control Points')
_ = ax.legend()

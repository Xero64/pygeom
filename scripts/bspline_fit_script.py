#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import (concatenate, cos, diag, hstack, linspace, ones, pi, sin,
                   vstack, zeros)
from numpy.linalg import lstsq, norm
from pygeom.geom2d import BSplineCurve2D, Vector2D
from pygeom.tools.solvers import solve_clsq

#%%
# Define the control points
exponent = 0
penalty = 10.0**exponent

num = 12
radius = 4.0
degree = 9

numtgt = num - 2
numvar = degree - 1

th = linspace(0, pi/2, num)[1:-1]
x_tgt = radius*cos(th)
y_tgt = radius*sin(th)
t = linspace(0.0, 1.0, num)
t_e = t[[0, -1]]
t = t[1:-1]

dx_tgt, dy_tgt = Vector2D.from_iter_xy([0.0, -1.0], [1.0, 0.0]).to_xy()

s = ones(t_e.shape)

t_ctl = linspace(0.0, 1.0, degree + 1)
x_ctl = radius*(1.0 - t_ctl)
y_ctl = radius*t_ctl
ctlpnts = Vector2D.from_iter_xy(x_ctl, y_ctl)

bspline = BSplineCurve2D(ctlpnts)
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
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y,
           color='green', label='Control Points')
_ = ax.legend()

#%%
# Fit the curve to the target points
max_iter = 20
count = 0

while True:

    print(f'Iteration {count}\n')

    pnts = bspline.evaluate_points_at_t(t)
    # print(f'pnts = \n{pnts}\n')

    tgts = bspline.evaluate_first_derivatives_at_t(t_e)
    # print(f'tgts = \n{tgts}\n')

    Dx = pnts.x - x_tgt
    Dy = pnts.y - y_tgt
    Du = tgts.x - s*dx_tgt
    Dv = tgts.y - s*dy_tgt

    fxy = concatenate((Dx, Dy))
    print(f'fxy = {fxy}\n')
    fuv = concatenate((Du, Dv))
    print(f'fuv = {fuv}\n')

    # f = concatenate((fxy, fuv))

    norm_f = norm(fxy) + norm(fuv)
    print(f'norm_f = {norm_f}\n')

    if norm_f < 1e-6:
        print('Converged')
        break

    Nt = bspline.basis_functions(t).transpose()[:, 1:-1]
    # print(f'Nt = \n{Nt}\n')

    dNt = bspline.basis_first_derivatives(t_e).transpose()[:, 1:-1]
    # print(f'dNt = \n{dNt}\n')

    dDxdX = Nt
    dDydX = zeros(dDxdX.shape)
    dDudX = dNt
    dDvdX = zeros(dDudX.shape)

    dDydY = Nt
    dDxdY = zeros(dDydY.shape)
    dDvdY = dNt
    dDudY = zeros(dDvdY.shape)

    dDxds = zeros((Dx.size, s.size))
    dDyds = zeros((Dy.size, s.size))
    dDuds = -diag(dx_tgt)/penalty
    dDvds = -diag(dy_tgt)/penalty

    dfxydXYs = vstack((hstack((dDxdX, dDxdY, dDxds)),
                       hstack((dDydX, dDydY, dDyds))))
    dfuvdXYs = vstack((hstack((dDudX, dDudY, dDuds)),
                       hstack((dDvdX, dDvdY, dDvds))))

    dvXYs, dlXYs = solve_clsq(dfxydXYs, fxy, dfuvdXYs, fuv)

    print(f'dvXYs = {dvXYs}\n')

    bspline.ctlpnts.x[1:-1] -= dvXYs[0:numvar]
    bspline.ctlpnts.y[1:-1] -= dvXYs[numvar:2*numvar]
    bspline.reset()

    s -= dvXYs[2*numvar:]

    pnts = bspline.evaluate_points_at_t(t)
    # print(f'pnts = \n{pnts}\n')

    Dx = pnts.x - x_tgt
    Dy = pnts.y - y_tgt

    fxy = concatenate((Dx, Dy))
    print(f'fxy = {fxy}\n')

    dDxdt, dDydt = bspline.evaluate_first_derivatives_at_t(t).to_xy()
    dDxdt = diag(dDxdt)
    dDydt = diag(dDydt)
    dfxydt = vstack((dDxdt, dDydt))

    dvt, _, _, _ = lstsq(dfxydt, fxy, rcond=None)
    print(f'dvt = {dvt}\n')

    t -= dvt

    norm_dv = norm(dvXYs) + norm(dvt)
    print(f'norm_dv = {norm_dv}\n')

    if norm_dv < 1e-12:
        print('Converged')
        break

    print(f'X = {bspline.ctlpnts.x}\n')
    print(f'Y = {bspline.ctlpnts.y}\n')
    print(f't = {t}\n')
    print(f't[1:] - t[:-1] = {t[1:] - t[:-1]}\n')
    print(f's = {s}\n')

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

#%%
# Import Dependencies
from numpy import linspace, pi, cos, sin, zeros, concatenate, vstack, hstack, diag
from numpy.linalg import lstsq, norm
from pygeom.geom2d import BSplineCurve2D, Vector2D
from matplotlib.pyplot import figure

#%%
# Define the control points
num = 4
radius = 2.0
degree = 3

numtgt = num - 2
numvar = degree - 1

th = linspace(0, pi/2, num)[1:-1]
x_tgt = radius*cos(th)
y_tgt = radius*sin(th)
t = linspace(0.0, 1.0, num)[1:-1]

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
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y, color='green',
           label='Control Points')
_ = ax.legend()

#%%
# Fit the curve to the target points
max_iter = 10
count = 0

while True:

    print(f'Iteration {count}\n')

    pnts = bspline.evaluate_points_at_t(t)

    dx = pnts.x - x_tgt
    dy = pnts.y - y_tgt

    f = concatenate((dx, dy))
    print(f'f = {f}\n')

    norm_f = norm(f)
    print(f'norm_f = {norm_f}\n')

    if norm_f < 1e-6:
        print('Converged')
        break

    Nt = bspline.basis_functions(t).transpose()[:, 1:-1]
    # print(f'Nt = \n{Nt}\n')

    dxdX = Nt
    # print(f'dxdX = \n{dxdX}\n')

    dydX = zeros(dxdX.shape)
    # print(f'dydX = \n{dydX}\n')

    dydY = Nt
    # print(f'dydY = \n{dydY}\n')

    dxdY = zeros(dydY.shape)
    # print(f'dxdY = \n{dxdY}\n')

    dxdt, dydt = bspline.evaluate_first_derivatives_at_t(t).to_xy()
    # print(f'dxdt = {dxdt}\n')
    # print(f'dydt = {dydt}\n')

    dfdv = vstack((hstack((dxdX, dxdY, diag(dxdt))),
                   hstack((dydX, dydY, diag(dydt)))))
    print(f'dfdv = \n{dfdv}\n')

    dv, residuals, rank_dfdv, s_vals = lstsq(dfdv, f, rcond=None)
    print(f'dv = {dv}\n')

    norm_dv = norm(dv)
    print(f'norm_dv = {norm_dv}\n')

    if norm_dv < 1e-6:
        print('Converged')
        break

    bspline.ctlpnts.x[1:-1] -= dv[0:numvar]
    bspline.ctlpnts.y[1:-1] -= dv[numvar:2*numvar]
    bspline.reset()

    t -= dv[2*numvar:]

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

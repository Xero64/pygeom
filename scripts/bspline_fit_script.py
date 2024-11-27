#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import (concatenate, cos, diag, hstack, linspace, ones, pi, sin,
                   vstack, zeros, arange, asarray)
from numpy.linalg import lstsq, norm
from pygeom.geom2d import BSplineCurve2D, Vector2D
from pygeom.tools.solvers import solve_clsq

#%%
# Set Target Points
num = 11
radius = 4.0

t = linspace(0.0, 1.0, num)
th = t*pi/2
x_tgt = radius*cos(th)
y_tgt = radius*sin(th)

ind_t_f = arange(1, num - 1)

ind_p = set(range(0, num))
ind_p_c = set([0, num - 1])
ind_p_f = ind_p - ind_p_c

ind_p_c = asarray(list(ind_p_c), dtype=int)
ind_p_f = asarray(list(ind_p_f), dtype=int)

ind_d = set(range(0, num))
ind_d_c = set([0, num - 1])
ind_d_f = ind_d - ind_d_c

ind_d_c = asarray(list(ind_d_c), dtype=int)
ind_d_f = asarray(list(ind_d_f), dtype=int)

u_c, v_c = Vector2D.from_iter_xy([0.0, -1.0], [1.0, 0.0]).to_xy()

x_c = x_tgt[ind_p_c]
y_c = y_tgt[ind_p_c]
x_f = x_tgt[ind_p_f]
y_f = y_tgt[ind_p_f]

s = ones(ind_d_c.size)

#%%
# Create Initial Spline
degree = 5
x_ctl = zeros(degree + 1)
y_ctl = zeros(degree + 1)
ctlpnts = Vector2D.from_iter_xy(x_ctl, y_ctl)

bspline = BSplineCurve2D(ctlpnts)
print(bspline)

#%%
# Plot Initial Spline and Target Points
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
ax.set_xlim(-1.0, radius + 1.0)
ax.set_ylim(-1.0, radius + 1.0)
_ = ax.legend()

#%%
# Fit the curve to the target points
max_iter = 20
count = 0

while True:

    print(f'Iteration {count}\n')

    t_p_c = t[ind_p_c]

    pnts_c = bspline.evaluate_points_at_t(t_p_c)

    Dx_c = pnts_c.x - x_c
    Dy_c = pnts_c.y - y_c

    t_p_f = t[ind_p_f]

    pnts_f = bspline.evaluate_points_at_t(t_p_f)

    Dx_f = pnts_f.x - x_f
    Dy_f = pnts_f.y - y_f

    t_d_c = t[ind_d_c]

    tgts_c = bspline.evaluate_first_derivatives_at_t(t_d_c)

    Du_c = tgts_c.x - s*u_c
    Dv_c = tgts_c.y - s*v_c

    f_f = concatenate((Dx_f, Dy_f))
    print(f'f_f = {f_f}\n')

    f_c = concatenate((Dx_c, Dy_c, Du_c, Dv_c))
    print(f'f_c = {f_c}\n')

    norm_f = norm(f_f) + norm(f_c)
    print(f'norm_f = {norm_f}\n')

    if norm_f < 1e-6:
        print('Converged')
        break

    t_f = t[ind_t_f]

    Nt = bspline.basis_functions(t).transpose()
    # print(f'Nt = \n{Nt}\n')

    dNt = bspline.basis_first_derivatives(t).transpose()
    # print(f'dNt = \n{dNt}\n')

    dDxdX_f = Nt[ind_p_f, ...]
    dDydX_f = zeros(dDxdX_f.shape)
    dDydY_f = Nt[ind_p_f, ...]
    dDxdY_f = zeros(dDydY_f.shape)
    dDxds_f = zeros((Dx_f.size, s.size))
    dDyds_f = zeros((Dy_f.size, s.size))

    dfdXYs_f = vstack((hstack((dDxdX_f, dDxdY_f, dDxds_f)),
                       hstack((dDydX_f, dDydY_f, dDyds_f))))

    dDxdX_c = Nt[ind_p_c, ...]
    dDydX_c = zeros((Dy_c.size, ctlpnts.size))
    dDxdY_c = zeros((Dx_c.size, ctlpnts.size))
    dDydY_c = Nt[ind_p_c, ...]
    dDxds_c = zeros((Dx_c.size, s.size))
    dDyds_c = zeros((Dy_c.size, s.size))

    dDudX_c = dNt[ind_d_c, ...]
    dDvdX_c = zeros((Dv_c.size, ctlpnts.size))
    dDudY_c = zeros((Du_c.size, ctlpnts.size))
    dDvdY_c = dNt[ind_d_c, ...]
    dDuds_c = -diag(u_c)
    dDvds_c = -diag(v_c)

    dfdXYs_c = vstack((hstack((dDxdX_c, dDxdY_c, dDxds_c)),
                       hstack((dDydX_c, dDydY_c, dDyds_c)),
                       hstack((dDudX_c, dDudY_c, dDuds_c)),
                       hstack((dDvdX_c, dDvdY_c, dDvds_c))))

    dvXYs, dlXYs = solve_clsq(dfdXYs_f, f_f, dfdXYs_c, f_c)

    print(f'dvXYs = {dvXYs}\n')

    bspline.ctlpnts.x -= dvXYs[0:ctlpnts.size]
    bspline.ctlpnts.y -= dvXYs[ctlpnts.size:2*ctlpnts.size]
    bspline.reset()

    s -= dvXYs[2*ctlpnts.size:]

    t_f = t[ind_t_f]

    pnts = bspline.evaluate_points_at_t(t_f)
    # print(f'pnts = \n{pnts}\n')

    Dx = pnts.x - x_tgt[ind_t_f]
    Dy = pnts.y - y_tgt[ind_t_f]

    fxy = concatenate((Dx, Dy))
    print(f'fxy = {fxy}\n')

    dDxdt, dDydt = bspline.evaluate_first_derivatives_at_t(t_f).to_xy()
    dDxdt = diag(dDxdt)
    dDydt = diag(dDydt)
    dfxydt = vstack((dDxdt, dDydt))

    dvt, _, _, _ = lstsq(dfxydt, fxy, rcond=None)
    print(f'dvt = {dvt}\n')

    t[ind_t_f] -= dvt

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
# Plot Final Spline and Target Points
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
ax.set_xlim(-1.0, radius + 1.0)
ax.set_ylim(-1.0, radius + 1.0)
_ = ax.legend()

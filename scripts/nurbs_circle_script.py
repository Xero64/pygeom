#%%
# Import Dependencies
from numpy import cos, float64, linspace, ones, pi, set_printoptions, sqrt
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep

from pygeom.geom2d import BSplineCurve2D, CubicSpline2D, NurbsCurve2D, Vector2D
from pygeom.tools.basis import knots_from_spacing
from pygeom.tools.mpl import (plot_curvature, plot_curve,
                              plot_first_derivatives, plot_points,
                              plot_second_derivatives)
from pygeom.tools.solvers import (cubic_bspline_correction,
                                  cubic_bspline_from_pspline)

set_printoptions(suppress=True)

#%%
# Define the control points and weights
num = 90
r = 2.0

# K = 4.0/3.0*(sqrt(2.0) - 1.0)
ang = pi/2
K = 4.0/3.0/(1.0/cos(ang/2) + 1.0)
Kchk = 4.0/3.0/(sqrt(2.0/(1.0 + cos(ang))) + 1.0)
print(f'K = {K}')
# print(f'Kchk = {Kchk}')

ctlpts = Vector2D.zeros(13, dtype=float64)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, K*r)
ctlpts[2] = Vector2D(K*r, r)
ctlpts[3] = Vector2D(0.0, r)
ctlpts[4] = Vector2D(-K*r, r)
ctlpts[5] = Vector2D(-r, K*r)
ctlpts[6] = Vector2D(-r, 0.0)
ctlpts[7] = Vector2D(-r, -K*r)
ctlpts[8] = Vector2D(-K*r, -r)
ctlpts[9] = Vector2D(0.0, -r)
ctlpts[10] = Vector2D(K*r, -r)
ctlpts[11] = Vector2D(r, -K*r)
ctlpts[12] = Vector2D(r, 0.0)

bsplinecurve = BSplineCurve2D(ctlpts, degree=3)

print(bsplinecurve)

splpnts = ctlpts[::3].stack_xy()

print('\n')
print(f'splpnts = \n{splpnts}\n')

tck, splt = splprep(splpnts.T, s=0, per=1)

splt: NDArray = splt

t_spl = linspace(splt.min(), splt.max(), 4*num)
x_spl, y_spl = splev(t_spl, tck, der=0)
dx_spl, dy_spl = splev(t_spl, tck, der=1)
d2x_spl, d2y_spl = splev(t_spl, tck, der=2)
k_spl = (dx_spl*d2y_spl - dy_spl*d2x_spl)/(dx_spl**2 + dy_spl**2)**1.5

ax = None
ax = plot_curve(bsplinecurve, ax=ax, label='BSpline', ls='-.')
ax.plot(x_spl, y_spl, label='Scipy Curve')
_ = ax.legend()

ax = None
ax = plot_points(bsplinecurve, ax=ax, label='BSpline', ls='-.')
ax.plot(t_spl, x_spl, label='Scipy X')
ax.plot(t_spl, y_spl, label='Scipy Y')
_ = ax.legend()

ax = None
ax = plot_first_derivatives(bsplinecurve, ax=ax, label='BSpline', ls='-.')
ax.plot(t_spl, dx_spl, label='Scipy dXdt')
ax.plot(t_spl, dy_spl, label='Scipy dYdt')
_ = ax.legend()

ax = None
ax = plot_second_derivatives(bsplinecurve, ax=ax, label='BSpline', ls='-.')
ax.plot(t_spl, d2x_spl, label='Scipy d2Xdt2')
ax.plot(t_spl, d2y_spl, label='Scipy d2Ydt2')
_ = ax.legend()

ax = None
ax = plot_curvature(bsplinecurve, ax=ax, label='BSpline Curvature', ls='-.')
ax.plot(t_spl, k_spl, label='Scipy Curvature')
_ = ax.legend()

#%%
# BSpline Cubic Fit
bctype = 'periodic'

pnts = ctlpts[::3]

print(f'pnts = \n{pnts}\n')

cubicspline = CubicSpline2D(pnts, bctype=bctype)

S = cubicspline.s[-1]

bknots = knots_from_spacing(cubicspline.s, degree=3)
nknots = knots_from_spacing(cubicspline.s, degree=2)

rmat = cubic_bspline_from_pspline(cubicspline.s, bctype=bctype)

ctlpnts = pnts.rmatmul(rmat)

ctlpnts_corrected = cubic_bspline_correction(ctlpnts)

print(f'ctlpnts_corrected = \n{ctlpnts_corrected}\n')

bsplinecurve = BSplineCurve2D(ctlpnts_corrected, knots=bknots, degree=3)
# bsplinecurve = BSplineCurve2D(ctlpnts_corrected, degree=3)

print(bsplinecurve)

ctlpnts = Vector2D.zeros(9)
ctlpnts[0] = Vector2D(r, 0.0)
ctlpnts[1] = Vector2D(r, r)
ctlpnts[2] = Vector2D(0.0, r)
ctlpnts[3] = Vector2D(-r, r)
ctlpnts[4] = Vector2D(-r, 0.0)
ctlpnts[5] = Vector2D(-r, -r)
ctlpnts[6] = Vector2D(0.0, -r)
ctlpnts[7] = Vector2D(r, -r)
ctlpnts[8] = Vector2D(r, 0.0)

weights = ones(9)
weights[1::2] = 1/sqrt(2)

nurbscurve = NurbsCurve2D(ctlpnts, weights=weights, knots=nknots, degree=2)

print(nurbscurve)

ax = None
ax = plot_curve(bsplinecurve, ax=ax, label='BSpline Curve', ls='-.')
# ax.plot(x_spl, y_spl, label='Scipy Curve')
ax = plot_curve(nurbscurve, ax=ax, label='NURBS Curve', ls='-.')
# ax = plot_curve(cubicspline, ax=ax, label='Cubic Spline Curve', ls='-.')
ax.scatter(bsplinecurve.ctlpnts.x, bsplinecurve.ctlpnts.y, color='r', label='BSpline Control Points')
# ax.scatter(nurbscurve.ctlpnts.x, nurbscurve.ctlpnts.y, color='g', label='NURBS Control Points')
_ = ax.legend()

ax = None
ax = plot_points(bsplinecurve, ax=ax, label='BSpline Curve', ls='-.')
# ax.plot(t_spl*S, x_spl, label='Scipy X')
# ax.plot(t_spl*S, y_spl, label='Scipy Y')
ax = plot_points(nurbscurve, ax=ax, label='NURBS Curve', ls='-.')
# ax = plot_points(cubicspline, ax=ax, label='Cubic Spline Curve', ls='-.')
_ = ax.legend()

ax = None
ax = plot_first_derivatives(bsplinecurve, ax=ax, label='BSpline Curve', ls='-.')
# ax.plot(t_spl*S, dx_spl/S, label='Scipy dXdt')
# ax.plot(t_spl*S, dy_spl/S, label='Scipy dYdt')
ax = plot_first_derivatives(nurbscurve, ax=ax, label='NURBS Curve', ls='-.')
# ax = plot_first_derivatives(cubicspline, ax=ax, label='Cubic Spline Curve', ls='-.')
_ = ax.legend()

ax = None
ax = plot_second_derivatives(bsplinecurve, ax=ax, label='BSpline Curve', ls='-.')
# ax.plot(t_spl*S, d2x_spl/S**2, label='Scipy d2Xdt2')
# ax.plot(t_spl*S, d2y_spl/S**2, label='Scipy d2Ydt2')
ax = plot_second_derivatives(nurbscurve, ax=ax, label='NURBS Curve', ls='-.')
# ax = plot_second_derivatives(cubicspline, ax=ax, label='Cubic Spline Curve', ls='-.')
_ = ax.legend()

ax = None
ax = plot_curvature(bsplinecurve, ax=ax, label='BSpline Curve', ls='-.')
# ax.plot(t_spl*S, k_spl, label='Scipy Curvature')
ax = plot_curvature(nurbscurve, ax=ax, label='NURBS Curve', ls='-.')
# ax = plot_curvature(cubicspline, ax=ax, label='Cubic Spline Curve', ls='-.')
_ = ax.legend()

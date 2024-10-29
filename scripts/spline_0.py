#%%
# Import Dependencies
from numpy import asarray

from pygeom.geom2d import BSplineCurve2D, CubicSpline2D, Vector2D
from pygeom.geom3d import Vector
from pygeom.geom3d.spline import Spline
from pygeom.tools.basis import knots_from_spacing
from pygeom.tools.mpl import (plot_curvature, plot_curve,
                              plot_first_derivatives, plot_points,
                              plot_second_derivatives)
from pygeom.tools.solvers import cubic_bspline_from_pspline

#%%
# Create Spline 2D
x = asarray([1.0, -1.0, 0.0, 0.0, 1.0])
y = asarray([0.0, 0.0, 1.0, -1.0, 0.0])
z = asarray([0.0, 0.0, 0.0, 0.0, 0.0])

pnts = Vector2D(x, y)

bctype = 'periodic'

cspline = CubicSpline2D(pnts, bctype=bctype)

rmat = cubic_bspline_from_pspline(cspline.s, bctype=bctype)

ctlpnts = rmat@pnts

knots = knots_from_spacing(cspline.s, 3)

bspline = BSplineCurve2D(ctlpnts, knots=knots, degree=3)

t = cspline.evaluate_t(num=100)

S = cspline.s[-1]

axs = None
axs = plot_curve(cspline, ax=axs, label='Cubic Spline')
axs = plot_curve(bspline, ax=axs, linestyle='--', label='BSpline')
_ = axs.legend()

axp = None
axp = plot_points(cspline, t=t, ax=axp)
axp = plot_points(bspline, t=t, ax=axp, linestyle='--')

axg = None
axg = plot_first_derivatives(cspline, t=t, ax=axg, label='Cubic Spline')
axg = plot_first_derivatives(bspline, t=t, ax=axg, linestyle='--', label='BSpline')
_ = axg.legend()

axn = None
axn = plot_second_derivatives(cspline, t=t, ax=axn, label='Cubic Spline')
axn = plot_second_derivatives(bspline, t=t, ax=axn, linestyle='--', label='BSpline')
_ = axn.legend()

axk = None
axk = plot_curvature(cspline, t=t, ax=axk, label='Cubic Spline')
axk = plot_curvature(bspline, t=t, ax=axk, linestyle='--', label='BSpline')
_ = axk.legend()

#%%
# Create Spline 3D
x = asarray([1.0, -1.0, 0.0, 0.0])
y = asarray([0.0, 0.0, 1.0, -1.0])
z = asarray([0.0, 0.0, 0.0, 0.0])

pnts = Vector(x, y, z)

spline = Spline(pnts, closed=True)

axs = spline.plot_spline(num=100, plane='xy')

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

print(f'spline.k = {spline.k}')

#%%
# Straight Edge Spline
spline.reset()

spline.pnls[1].set_straight_edge()

axs = spline.plot_spline(num=100, plane='xy')

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

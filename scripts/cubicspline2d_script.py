#%%
# Import Dependencies
from math import sqrt

from pygeom.geom2d import CubicSpline2D, Vector2D
from pygeom.tools.mpl import (plot_curvature, plot_curve,
                              plot_first_derivatives, plot_points,
                              plot_second_derivatives)

#%%
# Create CubicSpline2D 1
pnts = Vector2D.zeros(2)
pnts[0] = Vector2D(0.0, 0.0)
pnts[1] = Vector2D(1.0, 0.0)

tana = Vector2D(1.0, 1.0).to_unit()
tanb = Vector2D(1.0, -1.0).to_unit()

nrma = Vector2D(1.0, -1.0).to_unit()
nrmb = Vector2D(-1.0, -1.0).to_unit()

abm = 1.0
adb = tana.dot(tanb)

K = 4*sqrt(abm + adb)/(3*(sqrt(2)*abm**2 + sqrt(abm + adb)))
print(f'K = {K}\n')

# tena = 0.8
# tenb = 0.4

tena = 1.0
tenb = 1.0

bctype = ((1, tana*tena*K*2), (1, tanb*tenb*K*2))
# bctype = ((2, nrma*tena), (2, nrmb*tenb))

cs1 = CubicSpline2D(pnts, bctype=bctype)

ax = None
ax = plot_curve(cs1, ax=ax, label='Cubic Spline 1', ls='-.')
_ = ax.legend()

ax = None
ax = plot_points(cs1, ax=ax, label='Cubic Spline 1', ls='-.')
_ = ax.legend()

ax = None
ax = plot_first_derivatives(cs1, ax=ax, label='Cubic Spline 1', ls='-.')
_ = ax.legend()

ax = None
ax = plot_second_derivatives(cs1, ax=ax, label='Cubic Spline 1', ls='-.')
_ = ax.legend()

ax = None
ax = plot_curvature(cs1, ax=ax, label='Cubic Spline 1', ls='-.')
_ = ax.legend()

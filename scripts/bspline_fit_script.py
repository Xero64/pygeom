#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import cos, linspace, pi, sin, zeros
from pygeom.geom2d import BSplineCurve2D, Vector2D
from pygeom.tools.fits import bspline2d_lstsq_fit

#%%
# Set Target Points
num = 11
radius = 4.0

t = linspace(0.0, 1.0, num)
th = t*pi
pnts_target = Vector2D.from_iter_xy(cos(th), sin(th))*radius

tgts_dict = {0: Vector2D(0.0, 1.0), num - 1: Vector2D(0.0, 1.0)}

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
ax.plot(pnts.x, pnts.y, color='blue', label='BSpline Curve')
ax.scatter(pnts_target.x, pnts_target.y,
           color='red', label='Target Points')
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y,
           color='green', label='Control Points')
_ = ax.legend()

#%%
# Fit the curve to the target points
bspline = bspline2d_lstsq_fit(bspline, pnts_target, tgts_dict=tgts_dict,
                              display=True)

#%%
# Plot Final Spline and Target Points
nspl = 100
pnts = bspline.evaluate_points(nspl)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.set_aspect('equal')
ax.plot(pnts.x, pnts.y, color='blue', label='BSpline Curve')
ax.scatter(pnts_target.x, pnts_target.y,
           color='red', label='Target Points')
ax.scatter(bspline.ctlpnts.x, bspline.ctlpnts.y,
           color='green', label='Control Points')
_ = ax.legend()

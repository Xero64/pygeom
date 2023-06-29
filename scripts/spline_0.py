#%%
# Import Dependencies
from pygeom.geom2d.point2d import Point2D
from pygeom.geom2d.cubicspline2d import CubicSpline2D

#%%
# Create Spline
pnts = [Point2D(1.0, 0.0),
        Point2D(-1.0, 0.0),
        Point2D(0.0, 1.0),
        Point2D(0.0, -1.0)]

spline = CubicSpline2D(pnts, clsd=True)

axs = spline.plot_spline(num=100)

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

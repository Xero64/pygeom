#%%
# Import Dependencies
from pygeom.geom2d.point2d import Point2D
from pygeom.geom2d.spline2d import Spline2D

#%%
# Create Spline and Plot

pnts = [Point2D(1.0, 0.0),
        Point2D(-1.0, 0.0),
        Point2D(0.0, 1.0),
        Point2D(0.0, -1.0)]

spline = Spline2D(pnts, closed=True)

axs = spline.plot_spline(num=100)
axs.set_aspect('equal')

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

axq = spline.quiver_tangent(ax=axs, color='g')

axq = spline.quiver_normal(ax=axs, color='r')
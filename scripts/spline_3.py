#%%
# Import Dependencies
from pygeom.geom2d import Vector2D
from pygeom.geom2d.spline2d import Spline2D

#%%
# Create Spline and Plot
points = Vector2D.zeros(4)
points[0] = Vector2D(1.0, 0.0)
points[1] = Vector2D(-1.0, 0.0)
points[2] = Vector2D(0.0, 1.0)
points[3] = Vector2D(0.0, -1.0)

spline = Spline2D(points, closed=True)

axs = spline.plot_spline(num=100)
axs.set_aspect('equal')

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

axq = spline.quiver_tangent(ax=axs, color='g')

axq = spline.quiver_normal(ax=axs, color='r')

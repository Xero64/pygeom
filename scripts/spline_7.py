#%%
# Import Dependencies
from math import pi, cos, sin
from pygeom.geom2d.spline2d import Spline2D, Vector2D
from pygeom.geom2d import point2d_from_lists

#%%
# Create Spline
num = 4
x = [1.0, 0.0, -1.0, 0.0]
y = [0.0, 1.0, 0.0, -1.0]

pnts = point2d_from_lists(x, y)

thA = pi/2
tanA = Vector2D(cos(thA), sin(thA))*2.0
thB = pi/2 - 2*pi/num
tanB = Vector2D(cos(thB), sin(thB))*0.5

spline1 = Spline2D(pnts, closed=False, tanA=tanA, tanB=tanB)

axs = spline1.plot_spline(num=100)

axg = spline1.plot_gradient(num=100)

axc = spline1.plot_curvature()

axk = spline1.plot_inverse_radius(num=100)

#%%
# Create Spline
num = 4
x = [1.0, 0.0, -1.0, 0.0]
y = [0.0, 1.0, 0.0, -1.0]

pnts = point2d_from_lists(x, y)

thA = pi/2
tanA = Vector2D(cos(thA), sin(thA))*2.0
thB = pi/2 - 2*pi/num
tanB = Vector2D(cos(thB), sin(thB))*0.5

spline2 = Spline2D(pnts, closed=True)

axs = spline2.plot_spline(num=100)

axg = spline2.plot_gradient(num=100)

axc = spline2.plot_curvature()

axk = spline2.plot_inverse_radius(num=100)

#%%
# Import Dependencies
from numpy import asarray, cos, pi, sin
from pygeom.geom3d import Vector
from pygeom.geom3d.spline import Spline, Vector

#%%
# Create Spline
num = 4
x = asarray([1.0, 0.0, -1.0, 0.0])
y = asarray([0.0, 1.0, 0.0, -1.0])
z = asarray([0.0, 0.0, 0.0, 0.0])

pnts = Vector(x, y, z)

thA = pi/2
tanA = Vector(cos(thA), sin(thA), 0.0)*2.0
thB = pi/2 - 2*pi/num
tanB = Vector(cos(thB), sin(thB), 0.0)*0.5

spline1 = Spline(pnts, closed=False, tanA=tanA, tanB=tanB)

axs = spline1.plot_spline(num=100, plane='xy')

axg = spline1.plot_gradient(num=100)

axc = spline1.plot_curvature()

axk = spline1.plot_inverse_radius(num=100)

#%%
# Create Spline
thA = pi/2
tanA = Vector(cos(thA), sin(thA), 0.0)*2.0
thB = pi/2 - 2*pi/num
tanB = Vector(cos(thB), sin(thB), 0.0)*0.5

spline2 = Spline(pnts, closed=True)

axs = spline2.plot_spline(num=100, plane='xy')

axg = spline2.plot_gradient(num=100)

axc = spline2.plot_curvature()

axk = spline2.plot_inverse_radius(num=100)

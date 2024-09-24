#%%
# Import Dependencies
from numpy import asarray, cos, pi, sin
from pygeom.geom3d import point_from_lists
from pygeom.geom3d.spline import Spline, Vector

#%%
# Create Spline
x = asarray([1.0, 0.0, -1.0])
y = asarray([0.0, 1.0, 0.0])
z = asarray([0.0, 0.0, 0.0])

pnts = point_from_lists(x, y, z)

thA = pi/2
tanA = Vector(cos(thA), sin(thA), 0.0)
thB = -pi/2
tanB = Vector(cos(thB), sin(thB), 0.0)

thA = pi
nrmA = Vector(cos(thA), sin(thA), 0.0)*1.5
thB = 0.0
nrmB = Vector(cos(thB), sin(thB), 0.0)*1.5

spline1 = Spline(pnts, closed=False, tanA=tanA, tanB=tanB)

spline2 = Spline(pnts, closed=False, nrmA=nrmA, nrmB=nrmB)

axs = spline1.plot_spline(num=100, plane='xy')

axs = spline2.plot_spline(ax=axs, num=100, plane='xy')

axg = spline1.plot_gradient(num=100)

axg = spline2.plot_gradient(ax=axg, num=100)

axc = spline1.plot_curvature()

axc = spline2.plot_curvature(ax=axc)

axk = spline1.plot_inverse_radius(num=100)

axk = spline2.plot_inverse_radius(ax=axk, num=100)

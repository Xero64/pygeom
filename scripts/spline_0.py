#%%
# Import Dependencies
from pygeom.geom2d import point2d_from_lists
from pygeom.geom2d.spline2d import Spline2D
from pygeom.geom3d.spline import Spline
from pygeom.geom3d import point_from_lists

#%%
# Create Spline 2D
x = [1.0, -1.0, 0.0, 0.0]
y = [0.0, 0.0, 1.0, -1.0]
z = [0.0, 0.0, 0.0, 0.0]

pnts = point2d_from_lists(x, y)

spline = Spline2D(pnts, closed=True)

axs = spline.plot_spline(num=100)

axg = spline.plot_gradient(num=100)

axc = spline.plot_curvature()

axk = spline.plot_inverse_radius(num=100)

print(f'spline.k = {spline.k}')

#%%
# Create Spline 3D
pnts = point_from_lists(x, y, z)

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

#%%
# Straight Edge Spline
from numpy import zeros
from vedo import settings
from vedo.shapes import Spline, Points

settings.default_backend = 'k3d'

numpnt = len(pnts)
newpnts = zeros((numpnt, 3))
for i in range(numpnt):
    newpnts[i, 0] = pnts[i].x
    newpnts[i, 1] = pnts[i].y
    newpnts[i, 2] = pnts[i].z

newpnts = Points(newpnts)

spln = Spline(newpnts, closed=True)

spln.show(interactive=True)

#%%
# Import Dependencies
from numpy import float64, sqrt, zeros

from pygeom.geom2d import NurbsSurface2D, Vector2D
from pygeom.tools.k3d import (Plot, k3d_nurbs_control_points,
                              k3d_nurbs_control_polygon, k3d_surface,
                              k3d_surface_normals, k3d_surface_tangents)

#%%
# Create a Nurbs Circle Surface
ro = 4.0
ri = 0.0

ctlpnts = Vector2D.zeros((9, 2))

ctlpnts[0, 0] = Vector2D(ro, 0.0)
ctlpnts[1, 0] = Vector2D(ro, ro)
ctlpnts[2, 0] = Vector2D(0.0, ro)
ctlpnts[3, 0] = Vector2D(-ro, ro)
ctlpnts[4, 0] = Vector2D(-ro, 0.0)
ctlpnts[5, 0] = Vector2D(-ro, -ro)
ctlpnts[6, 0] = Vector2D(0.0, -ro)
ctlpnts[7, 0] = Vector2D(ro, -ro)
ctlpnts[8, 0] = Vector2D(ro, 0.0)
ctlpnts[0, 1] = Vector2D(ri, 0.0)
ctlpnts[1, 1] = Vector2D(ri, ri)
ctlpnts[2, 1] = Vector2D(0.0, ri)
ctlpnts[3, 1] = Vector2D(-ri, ri)
ctlpnts[4, 1] = Vector2D(-ri, 0.0)
ctlpnts[5, 1] = Vector2D(-ri, -ri)
ctlpnts[6, 1] = Vector2D(0.0, -ri)
ctlpnts[7, 1] = Vector2D(ri, -ri)
ctlpnts[8, 1] = Vector2D(ri, 0.0)

weights = zeros((9, 2), dtype=float64)
weights[0, 0] = 1.0
weights[1, 0] = 1.0/sqrt(2.0)
weights[2, 0] = 1.0
weights[3, 0] = 1.0/sqrt(2.0)
weights[4, 0] = 1.0
weights[5, 0] = 1.0/sqrt(2.0)
weights[6, 0] = 1.0
weights[7, 0] = 1.0/sqrt(2.0)
weights[8, 0] = 1.0
weights[0, 1] = 1.0
weights[1, 1] = 1.0/sqrt(2.0)
weights[2, 1] = 1.0
weights[3, 1] = 1.0/sqrt(2.0)
weights[4, 1] = 1.0
weights[5, 1] = 1.0/sqrt(2.0)
weights[6, 1] = 1.0
weights[7, 1] = 1.0/sqrt(2.0)
weights[8, 1] = 1.0

nurbssurface = NurbsSurface2D(ctlpnts, weights=weights, udegree=2, vdegree=1)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dmesh = k3d_surface(nurbssurface, unum=12, vnum=12)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrml = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtu, k3dtgtv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrml
plot += k3dtgtu
plot += k3dtgtv
plot.display()

#%%
# Create a Nurbs Trapezoid Surface
ctlpnts = Vector2D.zeros((2, 2))

ctlpnts[0, 0] = Vector2D(-2.0, -1.5)
ctlpnts[1, 0] = Vector2D(1.8, -1.7)
ctlpnts[0, 1] = Vector2D(-1.9, 2.0)
ctlpnts[1, 1] = Vector2D(2.1, 1.6)

weights = zeros((2, 2), dtype=float64)
weights[0, 0] = 1.0
weights[1, 0] = 1.0
weights[0, 1] = 1.0
weights[1, 1] = 1.0

nurbssurface = NurbsSurface2D(ctlpnts, weights=weights, udegree=1, vdegree=1)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dmesh = k3d_surface(nurbssurface, unum=36, vnum=36)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrml = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtu, k3dtgtv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrml
plot += k3dtgtu
plot += k3dtgtv
plot.display()

#%%
# Import Dependencies
from numpy import ones, sqrt
from pygeom.geom3d import NurbsSurface, Vector
from pygeom.tools.k3d import (Plot, k3d_nurbs_control_points,
                              k3d_nurbs_control_polygon, k3d_surface,
                              k3d_surface_normals, k3d_surface_tangents)

#%%
# Create a Nurbs Cylinder Surface
numu = 20
numv = 20

radius = 4.0
height = 8.0

ctlpnts = Vector.zeros((9, 2))

ctlpnts[0, 0] = Vector(radius, 0.0, -height/2)
ctlpnts[1, 0] = Vector(radius, radius, -height/2)
ctlpnts[2, 0] = Vector(0.0, radius, -height/2)
ctlpnts[3, 0] = Vector(-radius, radius, -height/2)
ctlpnts[4, 0] = Vector(-radius, 0.0, -height/2)
ctlpnts[5, 0] = Vector(-radius, -radius, -height/2)
ctlpnts[6, 0] = Vector(0.0, -radius, -height/2)
ctlpnts[7, 0] = Vector(radius, -radius, -height/2)
ctlpnts[8, 0] = Vector(radius, 0.0, -height/2)
ctlpnts[0, 1] = Vector(radius, 0.0, height/2)
ctlpnts[1, 1] = Vector(radius, radius, height/2)
ctlpnts[2, 1] = Vector(0.0, radius, height/2)
ctlpnts[3, 1] = Vector(-radius, radius, height/2)
ctlpnts[4, 1] = Vector(-radius, 0.0, height/2)
ctlpnts[5, 1] = Vector(-radius, -radius, height/2)
ctlpnts[6, 1] = Vector(0.0, -radius, height/2)
ctlpnts[7, 1] = Vector(radius, -radius, height/2)
ctlpnts[8, 1] = Vector(radius, 0.0, height/2)

weights = ones((9, 2))
weights[1::2, :] = 1.0/sqrt(2.0)

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=1)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dmesh = k3d_surface(nurbssurface, unum=36, vnum=36)
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrms = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

k3dpnts.visible = False
k3dpoly.visible = False
k3dnrms.visible = False
k3dtgtsu.visible = False
k3dtgtsv.visible = False

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Nurbs Cone Surface
numu = 20
numv = 20

ro = 4.0
ri = 0.0
height = 8.0

ctlpnts = Vector.zeros((9, 2))

ctlpnts[0, 0] = Vector(ro, 0.0, 0.0)
ctlpnts[1, 0] = Vector(ro, ro, 0.0)
ctlpnts[2, 0] = Vector(0.0, ro, 0.0)
ctlpnts[3, 0] = Vector(-ro, ro, 0.0)
ctlpnts[4, 0] = Vector(-ro, 0.0, 0.0)
ctlpnts[5, 0] = Vector(-ro, -ro, 0.0)
ctlpnts[6, 0] = Vector(0.0, -ro, 0.0)
ctlpnts[7, 0] = Vector(ro, -ro, 0.0)
ctlpnts[8, 0] = Vector(ro, 0.0, 0.0)
ctlpnts[0, 1] = Vector(ri, 0.0, height)
ctlpnts[1, 1] = Vector(ri, ri, height)
ctlpnts[2, 1] = Vector(0.0, ri, height)
ctlpnts[3, 1] = Vector(-ri, ri, height)
ctlpnts[4, 1] = Vector(-ri, 0.0, height)
ctlpnts[5, 1] = Vector(-ri, -ri, height)
ctlpnts[6, 1] = Vector(0.0, -ri, height)
ctlpnts[7, 1] = Vector(ri, -ri, height)
ctlpnts[8, 1] = Vector(ri, 0.0, height)

weights = ones((9, 2))
weights[1::2, :] = 1.0/sqrt(2.0)

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=1)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dmesh = k3d_surface(nurbssurface, unum=36, vnum=36)
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrms = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

k3dpnts.visible = False
k3dpoly.visible = False
k3dnrms.visible = False
k3dtgtsu.visible = False
k3dtgtsv.visible = False

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Nurbs Torus Surface
ro = 4.0
ri = 2.0
rm = (ro + ri)/2
ra = (ro - ri)/2

ctlpnts = Vector.zeros((9, 9))
ctlpnts[0, 0] = Vector(ro, 0.0, 0.0)
ctlpnts[1, 0] = Vector(ro, ro, 0.0)
ctlpnts[2, 0] = Vector(0.0, ro, 0.0)
ctlpnts[3, 0] = Vector(-ro, ro, 0.0)
ctlpnts[4, 0] = Vector(-ro, 0.0, 0.0)
ctlpnts[5, 0] = Vector(-ro, -ro, 0.0)
ctlpnts[6, 0] = Vector(0.0, -ro, 0.0)
ctlpnts[7, 0] = Vector(ro, -ro, 0.0)
ctlpnts[8, 0] = Vector(ro, 0.0, 0.0)
ctlpnts[0, 1] = Vector(ro, 0.0, ra)
ctlpnts[1, 1] = Vector(ro, ro, ra)
ctlpnts[2, 1] = Vector(0.0, ro, ra)
ctlpnts[3, 1] = Vector(-ro, ro, ra)
ctlpnts[4, 1] = Vector(-ro, 0.0, ra)
ctlpnts[5, 1] = Vector(-ro, -ro, ra)
ctlpnts[6, 1] = Vector(0.0, -ro, ra)
ctlpnts[7, 1] = Vector(ro, -ro, ra)
ctlpnts[8, 1] = Vector(ro, 0.0, ra)
ctlpnts[0, 2] = Vector(rm, 0.0, ra)
ctlpnts[1, 2] = Vector(rm, rm, ra)
ctlpnts[2, 2] = Vector(0.0, rm, ra)
ctlpnts[3, 2] = Vector(-rm, rm, ra)
ctlpnts[4, 2] = Vector(-rm, 0.0, ra)
ctlpnts[5, 2] = Vector(-rm, -rm, ra)
ctlpnts[6, 2] = Vector(0.0, -rm, ra)
ctlpnts[7, 2] = Vector(rm, -rm, ra)
ctlpnts[8, 2] = Vector(rm, 0.0, ra)
ctlpnts[0, 3] = Vector(ri, 0.0, ra)
ctlpnts[1, 3] = Vector(ri, ri, ra)
ctlpnts[2, 3] = Vector(0.0, ri, ra)
ctlpnts[3, 3] = Vector(-ri, ri, ra)
ctlpnts[4, 3] = Vector(-ri, 0.0, ra)
ctlpnts[5, 3] = Vector(-ri, -ri, ra)
ctlpnts[6, 3] = Vector(0.0, -ri, ra)
ctlpnts[7, 3] = Vector(ri, -ri, ra)
ctlpnts[8, 3] = Vector(ri, 0.0, ra)
ctlpnts[0, 4] = Vector(ri, 0.0, 0.0)
ctlpnts[1, 4] = Vector(ri, ri, 0.0)
ctlpnts[2, 4] = Vector(0.0, ri, 0.0)
ctlpnts[3, 4] = Vector(-ri, ri, 0.0)
ctlpnts[4, 4] = Vector(-ri, 0.0, 0.0)
ctlpnts[5, 4] = Vector(-ri, -ri, 0.0)
ctlpnts[6, 4] = Vector(0.0, -ri, 0.0)
ctlpnts[7, 4] = Vector(ri, -ri, 0.0)
ctlpnts[8, 4] = Vector(ri, 0.0, 0.0)
ctlpnts[0, 5] = Vector(ri, 0.0, -ra)
ctlpnts[1, 5] = Vector(ri, ri, -ra)
ctlpnts[2, 5] = Vector(0.0, ri, -ra)
ctlpnts[3, 5] = Vector(-ri, ri, -ra)
ctlpnts[4, 5] = Vector(-ri, 0.0, -ra)
ctlpnts[5, 5] = Vector(-ri, -ri, -ra)
ctlpnts[6, 5] = Vector(0.0, -ri, -ra)
ctlpnts[7, 5] = Vector(ri, -ri, -ra)
ctlpnts[8, 5] = Vector(ri, 0.0, -ra)
ctlpnts[0, 6] = Vector(rm, 0.0, -ra)
ctlpnts[1, 6] = Vector(rm, rm, -ra)
ctlpnts[2, 6] = Vector(0.0, rm, -ra)
ctlpnts[3, 6] = Vector(-rm, rm, -ra)
ctlpnts[4, 6] = Vector(-rm, 0.0, -ra)
ctlpnts[5, 6] = Vector(-rm, -rm, -ra)
ctlpnts[6, 6] = Vector(0.0, -rm, -ra)
ctlpnts[7, 6] = Vector(rm, -rm, -ra)
ctlpnts[8, 6] = Vector(rm, 0.0, -ra)
ctlpnts[0, 7] = Vector(ro, 0.0, -ra)
ctlpnts[1, 7] = Vector(ro, ro, -ra)
ctlpnts[2, 7] = Vector(0.0, ro, -ra)
ctlpnts[3, 7] = Vector(-ro, ro, -ra)
ctlpnts[4, 7] = Vector(-ro, 0.0, -ra)
ctlpnts[5, 7] = Vector(-ro, -ro, -ra)
ctlpnts[6, 7] = Vector(0.0, -ro, -ra)
ctlpnts[7, 7] = Vector(ro, -ro, -ra)
ctlpnts[8, 7] = Vector(ro, 0.0, -ra)
ctlpnts[0, 8] = Vector(ro, 0.0, 0.0)
ctlpnts[1, 8] = Vector(ro, ro, 0.0)
ctlpnts[2, 8] = Vector(0.0, ro, 0.0)
ctlpnts[3, 8] = Vector(-ro, ro, 0.0)
ctlpnts[4, 8] = Vector(-ro, 0.0, 0.0)
ctlpnts[5, 8] = Vector(-ro, -ro, 0.0)
ctlpnts[6, 8] = Vector(0.0, -ro, 0.0)
ctlpnts[7, 8] = Vector(ro, -ro, 0.0)
ctlpnts[8, 8] = Vector(ro, 0.0, 0.0)

w = 1.0/sqrt(2.0)

wu = ones((9, 1))
wu[1::2, 0] = w
wv = ones((1, 9))
wv[0, 1::2] = w
weights = wu@wv

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=2)

print(nurbssurface)

pnts = nurbssurface.evaluate_points(numu, numv)
tgtsu, tgtsv = nurbssurface.evaluate_tangents(numu, numv)
nrms = tgtsu.cross(tgtsv)

#%%
# Plot the NURBS Surface using K3D
k3dmesh = k3d_surface(nurbssurface, unum=36, vnum=36)
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrms = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

k3dpnts.visible = False
k3dpoly.visible = False
k3dnrms.visible = False
k3dtgtsu.visible = False
k3dtgtsv.visible = False

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Nurbs Sphere Surface
ri = 4.0

ctlpnts = Vector.zeros((9, 5))

ctlpnts[0, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[1, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[2, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[3, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[4, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[5, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[6, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[7, 0] = Vector(0.0, 0.0, -radius)
ctlpnts[8, 0] = Vector(0.0, 0.0, -radius)

ctlpnts[0, 1] = Vector(radius, 0.0, -radius)
ctlpnts[1, 1] = Vector(radius, radius, -radius)
ctlpnts[2, 1] = Vector(0.0, radius, -radius)
ctlpnts[3, 1] = Vector(-radius, radius, -radius)
ctlpnts[4, 1] = Vector(-radius, 0.0, -radius)
ctlpnts[5, 1] = Vector(-radius, -radius, -radius)
ctlpnts[6, 1] = Vector(0.0, -radius, -radius)
ctlpnts[7, 1] = Vector(radius, -radius, -radius)
ctlpnts[8, 1] = Vector(radius, 0.0, -radius)

ctlpnts[0, 2] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 2] = Vector(radius, radius, 0.0)
ctlpnts[2, 2] = Vector(0.0, radius, 0.0)
ctlpnts[3, 2] = Vector(-radius, radius, 0.0)
ctlpnts[4, 2] = Vector(-radius, 0.0, 0.0)
ctlpnts[5, 2] = Vector(-radius, -radius, 0.0)
ctlpnts[6, 2] = Vector(0.0, -radius, 0.0)
ctlpnts[7, 2] = Vector(radius, -radius, 0.0)
ctlpnts[8, 2] = Vector(radius, 0.0, 0.0)

ctlpnts[0, 3] = Vector(radius, 0.0, radius)
ctlpnts[1, 3] = Vector(radius, radius, radius)
ctlpnts[2, 3] = Vector(0.0, radius, radius)
ctlpnts[3, 3] = Vector(-radius, radius, radius)
ctlpnts[4, 3] = Vector(-radius, 0.0, radius)
ctlpnts[5, 3] = Vector(-radius, -radius, radius)
ctlpnts[6, 3] = Vector(0.0, -radius, radius)
ctlpnts[7, 3] = Vector(radius, -radius, radius)
ctlpnts[8, 3] = Vector(radius, 0.0, radius)

ctlpnts[0, 4] = Vector(0.0, 0.0, radius)
ctlpnts[1, 4] = Vector(0.0, 0.0, radius)
ctlpnts[2, 4] = Vector(0.0, 0.0, radius)
ctlpnts[3, 4] = Vector(0.0, 0.0, radius)
ctlpnts[4, 4] = Vector(0.0, 0.0, radius)
ctlpnts[5, 4] = Vector(0.0, 0.0, radius)
ctlpnts[6, 4] = Vector(0.0, 0.0, radius)
ctlpnts[7, 4] = Vector(0.0, 0.0, radius)
ctlpnts[8, 4] = Vector(0.0, 0.0, radius)

w = 1.0/sqrt(2.0)

wu = ones((9, 1))
wu[1::2, 0] = w
wv = ones((1, 5))
wv[0, 1::2] = w
weights = wu@wv

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=2)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dmesh = k3d_surface(nurbssurface, unum=36, vnum=36)
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrms = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

k3dpnts.visible = False
k3dpoly.visible = False
k3dnrms.visible = False
k3dtgtsu.visible = False
k3dtgtsv.visible = False

plot = Plot()
plot += k3dpnts
plot += k3dmesh
plot += k3dpoly
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

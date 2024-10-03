#%%
# Import Dependencies
from numpy import set_printoptions, sqrt, zeros
from pygeom.geom3d import NurbsCurve, NurbsSurface, Vector
from pygeom.tools.k3d import (Plot, k3d_nurbs_control_points,
                              k3d_nurbs_control_polygon, k3d_surface,
                              k3d_surface_normals, k3d_surface_tangents)
from pygeom.tools.solvers import cubic_bspline_fit_solver

set_printoptions(suppress=True)

#%%
# Nurbs Circles
ctlpnts = Vector.zeros(9)
ctlpnts[0] = Vector(1.0, 0.0, 0.0)
ctlpnts[1] = Vector(1.0, 1.0, 0.0)
ctlpnts[2] = Vector(0.0, 1.0, 0.0)
ctlpnts[3] = Vector(-1.0, 1.0, 0.0)
ctlpnts[4] = Vector(-1.0, 0.0, 0.0)
ctlpnts[5] = Vector(-1.0, -1.0, 0.0)
ctlpnts[6] = Vector(0.0, -1.0, 0.0)
ctlpnts[7] = Vector(1.0, -1.0, 0.0)
ctlpnts[8] = Vector(1.0, 0.0, 0.0)

weights = zeros(9)
weights[:] = 1.0
weights[1::2] = 1.0/sqrt(2.0)

radius1 = 2.0
rpos1 = Vector(-0.25, 0.0, 0.0)

ctlpnts1 = ctlpnts*radius1 + rpos1

nurbscircle1 = NurbsCurve(ctlpnts1, weights=weights)

radius2 = 4.0
rpos2 = Vector(0.0, 0.25, 2.0)

ctlpnts2 = ctlpnts*radius2 + rpos2

nurbscircle2 = NurbsCurve(ctlpnts2, weights=weights)

radius3 = 3.0
rpos3 = Vector(0.25, 0.0, 4.0)

ctlpnts3 = ctlpnts*radius3 + rpos3

nurbscircle3 = NurbsCurve(ctlpnts3, weights=weights)

radius4 = 4.0
rpos4 = Vector(0.0, -0.25, 6.0)

ctlpnts4 = ctlpnts*radius4 + rpos4

nurbscircle4 = NurbsCurve(ctlpnts4, weights=weights)

#%%
# Create Nurbs Surface
pntsS = Vector.zeros((9, 4))

pntsS[:, 0] = ctlpnts1
pntsS[:, 1] = ctlpnts2
pntsS[:, 2] = ctlpnts3
pntsS[:, 3] = ctlpnts4

print(f'pntsS.shape = {pntsS.shape}')

rmat = cubic_bspline_fit_solver(4, bc_type='quadratic')

print(f'rmat.shape = {rmat.shape}')

ctlpntsS = pntsS@rmat.transpose()

print(f'ctlpntsS.shape = {ctlpntsS.shape}')

wgtsS = weights.reshape((-1, 1)).repeat(10, axis=1)

print(f'wgtsS.shape = {wgtsS.shape}')

nurbssurface = NurbsSurface(ctlpntsS, weights=wgtsS, udegree=2, vdegree=3)

print(nurbssurface)

#%%
# Plot the NURBS Surface using K3D
k3dmesh = k3d_surface(nurbssurface, unum=12, vnum=12)
k3dpnts = k3d_nurbs_control_points(nurbssurface, scale=0.2)
k3dpoly = k3d_nurbs_control_polygon(nurbssurface)
k3dnrml = k3d_surface_normals(nurbssurface, unum=12, vnum=12, scale=0.2)
k3dtgtu, k3dtgtv = k3d_surface_tangents(nurbssurface, unum=12, vnum=12, scale=0.2)

k3dpnts.visible = False
k3dpoly.visible = False
k3dnrml.visible = False
k3dtgtu.visible = False
k3dtgtv.visible = False

plot = Plot()
plot += k3dmesh
plot += k3dpnts
plot += k3dpoly
plot += k3dnrml
plot += k3dtgtu
plot += k3dtgtv
plot.display()

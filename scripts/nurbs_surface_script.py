#%%
# Import Dependencies
from typing import TYPE_CHECKING, Union

from k3d import Plot as k3dPlot
from k3d import mesh as k3dmesh
from k3d import points as k3dpoints
from numpy import arange, asarray, float64, sqrt, zeros
from pygeom.array3d import NurbsSurface, zero_arrayvector
from pygeom.geom3d import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]

#%%
# Create a Nurbs Torus Surface
numu = 20
numv = 20

ro = 4.0
ri = 2.0
rm = (ro + ri)/2
ra = (ro - ri)/2

ctlpnts = zero_arrayvector((9, 9))
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

weights = zeros((9, 9), dtype=float64)
weights[0, 0] = 1.0
weights[1, 0] = 1.0/sqrt(2.0)
weights[2, 0] = 1.0
weights[3, 0] = 1.0/sqrt(2.0)
weights[4, 0] = 1.0
weights[5, 0] = 1.0/sqrt(2.0)
weights[6, 0] = 1.0
weights[7, 0] = 1.0/sqrt(2.0)
weights[8, 0] = 1.0
weights[0, 1] = 1.0/sqrt(2.0)
weights[1, 1] = 0.5
weights[2, 1] = 1.0/sqrt(2.0)
weights[3, 1] = 0.5
weights[4, 1] = 1.0/sqrt(2.0)
weights[5, 1] = 0.5
weights[6, 1] = 1.0/sqrt(2.0)
weights[7, 1] = 0.5
weights[8, 1] = 1.0/sqrt(2.0)
weights[0, 2] = 1.0
weights[1, 2] = 1.0/sqrt(2.0)
weights[2, 2] = 1.0
weights[3, 2] = 1.0/sqrt(2.0)
weights[4, 2] = 1.0
weights[5, 2] = 1.0/sqrt(2.0)
weights[6, 2] = 1.0
weights[7, 2] = 1.0/sqrt(2.0)
weights[8, 2] = 1.0
weights[0, 3] = 1.0/sqrt(2.0)
weights[1, 3] = 0.5
weights[2, 3] = 1.0/sqrt(2.0)
weights[3, 3] = 0.5
weights[4, 3] = 1.0/sqrt(2.0)
weights[5, 3] = 0.5
weights[6, 3] = 1.0/sqrt(2.0)
weights[7, 3] = 0.5
weights[8, 3] = 1.0/sqrt(2.0)
weights[0, 4] = 1.0
weights[1, 4] = 1.0/sqrt(2.0)
weights[2, 4] = 1.0
weights[3, 4] = 1.0/sqrt(2.0)
weights[4, 4] = 1.0
weights[5, 4] = 1.0/sqrt(2.0)
weights[6, 4] = 1.0
weights[7, 4] = 1.0/sqrt(2.0)
weights[8, 4] = 1.0
weights[0, 5] = 1.0/sqrt(2.0)
weights[1, 5] = 0.5
weights[2, 5] = 1.0/sqrt(2.0)
weights[3, 5] = 0.5
weights[4, 5] = 1.0/sqrt(2.0)
weights[5, 5] = 0.5
weights[6, 5] = 1.0/sqrt(2.0)
weights[7, 5] = 0.5
weights[8, 5] = 1.0/sqrt(2.0)
weights[0, 6] = 1.0
weights[1, 6] = 1.0/sqrt(2.0)
weights[2, 6] = 1.0
weights[3, 6] = 1.0/sqrt(2.0)
weights[4, 6] = 1.0
weights[5, 6] = 1.0/sqrt(2.0)
weights[6, 6] = 1.0
weights[7, 6] = 1.0/sqrt(2.0)
weights[8, 6] = 1.0
weights[0, 7] = 1.0/sqrt(2.0)
weights[1, 7] = 0.5
weights[2, 7] = 1.0/sqrt(2.0)
weights[3, 7] = 0.5
weights[4, 7] = 1.0/sqrt(2.0)
weights[5, 7] = 0.5
weights[6, 7] = 1.0/sqrt(2.0)
weights[7, 7] = 0.5
weights[8, 7] = 1.0/sqrt(2.0)
weights[0, 8] = 1.0
weights[1, 8] = 1.0/sqrt(2.0)
weights[2, 8] = 1.0
weights[3, 8] = 1.0/sqrt(2.0)
weights[4, 8] = 1.0
weights[5, 8] = 1.0/sqrt(2.0)
weights[6, 8] = 1.0
weights[7, 8] = 1.0/sqrt(2.0)
weights[8, 8] = 1.0

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=2)

print(f'Control Points:\n{nurbssurface.ctlpnts}\n')
print(f'Weights:\n{nurbssurface.weights}\n')
print(f'Knots:\n{nurbssurface.uknots}\n{nurbssurface.vknots}\n')
print(f'Degree:\n{nurbssurface.udegree}\n{nurbssurface.vdegree}\n')

pnts = nurbssurface.evaluate_points(numu, numv)
tgtsu, tgtsv = nurbssurface.evaluate_tangents(numu, numv)
nrms = tgtsu.cross(tgtsv)

#%%
# Plot the NURBS Surface using K3D
u, v = nurbssurface.evaluate_uv(numu, numv)
num = u.size*v.size
ind = arange(num, dtype=int).reshape(u.size, v.size)

faces = []
for i in range(u.size-1):
    for j in range(v.size-1):
        faces.append([ind[i, j], ind[i+1, j], ind[i+1, j+1]])
        faces.append([ind[i, j], ind[i+1, j+1], ind[i, j+1]])

faces = asarray(faces, dtype=int)

pnts = pnts.reshape(num)
nrms = nrms.reshape(num)

pntsxyz = pnts.stack_xyz()
nrmsxyz = nrms.stack_xyz()

plot = k3dPlot()
mesh = k3dmesh(pntsxyz.astype('float32'), faces.astype('uint32'),
               nrmsxyz.astype('float32'), color=0xffd500, flat_shading=False)
ctlptsxyz = ctlpnts.stack_xyz()
plot += k3dpoints(ctlptsxyz.astype('float32'), point_size=0.1, color=0xff0000)
plot += mesh
plot.display()

#%%
# Create a Nurbs Sphere Surface
numu = 20
numv = 20

radius = 4.0

ctlpnts = zero_arrayvector((5, 9))

ctlpnts[0, 0] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 0] = Vector(radius, radius, 0.0)
ctlpnts[2, 0] = Vector(0.0, radius, 0.0)
ctlpnts[3, 0] = Vector(-radius, radius, 0.0)
ctlpnts[4, 0] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 1] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 1] = Vector(radius, radius, radius)
ctlpnts[2, 1] = Vector(0.0, radius, radius)
ctlpnts[3, 1] = Vector(-radius, radius, radius)
ctlpnts[4, 1] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 2] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 2] = Vector(radius, 0.0, radius)
ctlpnts[2, 2] = Vector(0.0, 0.0, radius)
ctlpnts[3, 2] = Vector(-radius, 0.0, radius)
ctlpnts[4, 2] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 3] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 3] = Vector(radius, -radius, radius)
ctlpnts[2, 3] = Vector(0.0, -radius, radius)
ctlpnts[3, 3] = Vector(-radius, -radius, radius)
ctlpnts[4, 3] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 4] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 4] = Vector(radius, -radius, 0.0)
ctlpnts[2, 4] = Vector(0.0, -radius, 0.0)
ctlpnts[3, 4] = Vector(-radius, -radius, 0.0)
ctlpnts[4, 4] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 5] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 5] = Vector(radius, -radius, -radius)
ctlpnts[2, 5] = Vector(0.0, -radius, -radius)
ctlpnts[3, 5] = Vector(-radius, -radius, -radius)
ctlpnts[4, 5] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 6] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 6] = Vector(radius, 0.0, -radius)
ctlpnts[2, 6] = Vector(0.0, 0.0, -radius)
ctlpnts[3, 6] = Vector(-radius, 0.0, -radius)
ctlpnts[4, 6] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 7] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 7] = Vector(radius, radius, -radius)
ctlpnts[2, 7] = Vector(0.0, radius, -radius)
ctlpnts[3, 7] = Vector(-radius, radius, -radius)
ctlpnts[4, 7] = Vector(-radius, 0.0, 0.0)

ctlpnts[0, 8] = Vector(radius, 0.0, 0.0)
ctlpnts[1, 8] = Vector(radius, radius, 0.0)
ctlpnts[2, 8] = Vector(0.0, radius, 0.0)
ctlpnts[3, 8] = Vector(-radius, radius, 0.0)
ctlpnts[4, 8] = Vector(-radius, 0.0, 0.0)

weights = zeros((5, 9), dtype=float64)

weights[0, 0] = 1.0
weights[1, 0] = 1.0/sqrt(2.0)
weights[2, 0] = 1.0
weights[3, 0] = 1.0/sqrt(2.0)
weights[4, 0] = 1.0

weights[0, 1] = 1.0/sqrt(2.0)
weights[1, 1] = 0.5
weights[2, 1] = 1.0/sqrt(2.0)
weights[3, 1] = 0.5
weights[4, 1] = 1.0/sqrt(2.0)

weights[0, 2] = 1.0
weights[1, 2] = 1.0/sqrt(2.0)
weights[2, 2] = 1.0
weights[3, 2] = 1.0/sqrt(2.0)
weights[4, 2] = 1.0

weights[0, 3] = 1.0/sqrt(2.0)
weights[1, 3] = 0.5
weights[2, 3] = 1.0/sqrt(2.0)
weights[3, 3] = 0.5
weights[4, 3] = 1.0/sqrt(2.0)

weights[0, 4] = 1.0
weights[1, 4] = 1.0/sqrt(2.0)
weights[2, 4] = 1.0
weights[3, 4] = 1.0/sqrt(2.0)
weights[4, 4] = 1.0

weights[0, 5] = 1.0/sqrt(2.0)
weights[1, 5] = 0.5
weights[2, 5] = 1.0/sqrt(2.0)
weights[3, 5] = 0.5
weights[4, 5] = 1.0/sqrt(2.0)

weights[0, 6] = 1.0
weights[1, 6] = 1.0/sqrt(2.0)
weights[2, 6] = 1.0
weights[3, 6] = 1.0/sqrt(2.0)
weights[4, 6] = 1.0

weights[0, 7] = 1.0/sqrt(2.0)
weights[1, 7] = 0.5
weights[2, 7] = 1.0/sqrt(2.0)
weights[3, 7] = 0.5
weights[4, 7] = 1.0/sqrt(2.0)

weights[0, 8] = 1.0
weights[1, 8] = 1.0/sqrt(2.0)
weights[2, 8] = 1.0
weights[3, 8] = 1.0/sqrt(2.0)
weights[4, 8] = 1.0

nurbssurface = NurbsSurface(ctlpnts, weights=weights, udegree=2, vdegree=2)

print(f'Control Points:\n{nurbssurface.ctlpnts}\n')
print(f'Weights:\n{nurbssurface.weights}\n')
print(f'Knots:\n{nurbssurface.uknots}\n{nurbssurface.vknots}\n')
print(f'Degree:\n{nurbssurface.udegree}\n{nurbssurface.vdegree}\n')

pnts = nurbssurface.evaluate_points(numu, numv)
tgtsu, tgtsv = nurbssurface.evaluate_tangents(numu, numv)
nrms = tgtsu.cross(tgtsv)

#%%
# Plot the NURBS Surface using K3D
u, v = nurbssurface.evaluate_uv(numu, numv)
num = u.size*v.size
ind = arange(num, dtype=int).reshape(u.size, v.size)

faces = []
for i in range(u.size-1):
    for j in range(v.size-1):
        faces.append([ind[i, j], ind[i+1, j], ind[i+1, j+1]])
        faces.append([ind[i, j], ind[i+1, j+1], ind[i, j+1]])

faces = asarray(faces, dtype=int)

pnts = pnts.reshape(num)
nrms = nrms.reshape(num)

pntsxyz = pnts.stack_xyz()
nrmsxyz = nrms.stack_xyz()

plot = k3dPlot()
mesh = k3dmesh(pntsxyz.astype('float32'), faces.astype('uint32'),
               nrmsxyz.astype('float32'), color=0xffd500, flat_shading=False)
ctlptsxyz = ctlpnts.flatten().stack_xyz()
weightsps = weights.flatten()
plot += k3dpoints(ctlptsxyz.astype('float32'),
                  point_sizes=weightsps.astype('float32'), color=0xff0000)
plot += mesh
plot.display()
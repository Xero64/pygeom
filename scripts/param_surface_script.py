#%%
# Import Dependencies
from numpy import cos, float64, full, pi, sin, zeros
from numpy.typing import NDArray
from pygeom.array3d import ArrayVector, ParamSurface
from pygeom.geom3d import Vector
from pygeom.tools.k3d import (Plot, k3d_surface, k3d_surface_normals,
                              k3d_surface_tangents)

#%%
# Create a Parametric Cylinder Surface
radius = 4.0
height = 8.0

def ruv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = radius*cos(2*pi*u)
    y = radius*sin(2*pi*u)
    z = height*v
    return ArrayVector(x, y, z)

def drdu(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -2*pi*radius*sin(2*pi*u)
    y = 2*pi*radius*cos(2*pi*u)
    z = zeros(v.shape, dtype=v.dtype)
    return ArrayVector(x, y, z)

def drdv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = full(v.shape, height, dtype=v.dtype)
    return ArrayVector(x, y, z)

paramsurface = ParamSurface(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=36, vnum=36)
k3dnrms = k3d_surface_normals(paramsurface, unum=36, vnum=36, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=36, vnum=36, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Parametric Cone Surface
radius = 4.0
height = 8.0

def ruv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = radius*(1 - v)*cos(2*pi*u)
    y = radius*(1 - v)*sin(2*pi*u)
    z = height*v
    return ArrayVector(x, y, z)

def drdv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -radius*cos(2*pi*u)
    y = -radius*sin(2*pi*u)
    z = full(v.shape, height, dtype=v.dtype)
    return ArrayVector(x, y, z)

def drdu(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -2*pi*radius*(1 - v)*sin(2*pi*u)
    y = 2*pi*radius*(1 - v)*cos(2*pi*u)
    z = zeros(v.shape, dtype=v.dtype)
    
    drdu = ArrayVector(x, y, z)

    chk_v1 = v == 1.0
    u_v1 = u[chk_v1]
    v_v1 = v[chk_v1]
    drdv_v1 = drdv(u_v1, v_v1)
    temp_v1 = ArrayVector(-drdv_v1.x, -drdv_v1.y, drdv_v1.z)
    drdu_v1 = drdv_v1.cross(temp_v1)
    drdu[chk_v1] = drdu_v1

    return drdu

paramsurface = ParamSurface(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=36, vnum=36)
k3dnrms = k3d_surface_normals(paramsurface, unum=36, vnum=36, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=36, vnum=36, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Parametric Torus Surface
ro = 4.0
ri = 2.0
rm = (ro + ri)/2
ra = (ro - ri)/2

def ruv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = (ra*cos(2*pi*v) + rm)*cos(2*pi*u)
    y = (ra*cos(2*pi*v) + rm)*sin(2*pi*u)
    z = ra*sin(2*pi*v)
    return ArrayVector(x, y, z)

def drdu(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -2*pi*(ra*cos(2*pi*v) + rm)*sin(2*pi*u)
    y = 2*pi*(ra*cos(2*pi*v) + rm)*cos(2*pi*u)
    z = zeros(u.shape, dtype=u.dtype)
    return ArrayVector(x, y, z)

def drdv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -2*pi*ra*sin(2*pi*v)*cos(2*pi*u)
    y = -2*pi*ra*sin(2*pi*u)*sin(2*pi*v)
    z = 2*pi*ra*cos(2*pi*v)
    return ArrayVector(x, y, z)

paramsurface = ParamSurface(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=36, vnum=36)
k3dnrms = k3d_surface_normals(paramsurface, unum=36, vnum=36, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=36, vnum=36, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Parametric Sphere Surface
radius = 4.0

def ruv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = radius*cos(2*pi*u)*sin(pi*v)
    y = radius*sin(2*pi*u)*sin(pi*v)
    z = -radius*cos(pi*v)
    return ArrayVector(x, y, z)

def drdv(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = pi*radius*cos(2*pi*u)*cos(pi*v)
    y = pi*radius*sin(2*pi*u)*cos(pi*v)
    z = pi*radius*sin(pi*v)
    return ArrayVector(x, y, z)

def drdu(u: NDArray[float64], v: NDArray[float64]) -> ArrayVector:
    x = -2*pi*radius*sin(2*pi*u)*sin(pi*v)
    y = 2*pi*radius*cos(2*pi*u)*sin(pi*v)
    z = zeros(v.shape, dtype=u.dtype)
    drdu = ArrayVector(x, y, z)

    chk_v0 = v == 0.0
    u_v0 = u[chk_v0]
    v_v0 = v[chk_v0]
    nrml_v0 = Vector(0.0, 0.0, -1.0)
    drdv_v0 = drdv(u_v0, v_v0)
    drdu_v0 = drdv_v0.cross(nrml_v0)

    chk_v1 = v == 1.0
    u_v1 = u[chk_v1]
    v_v1 = v[chk_v1]
    nrml_v1 = Vector(0.0, 0.0, 1.0)
    drdv_v1 = drdv(u_v1, v_v1)
    drdu_v1 = drdv_v1.cross(nrml_v1)

    drdu[chk_v0] = drdu_v0
    drdu[chk_v1] = drdu_v1

    return drdu

paramsurface = ParamSurface(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=36, vnum=36)
k3dnrms = k3d_surface_normals(paramsurface, unum=36, vnum=36, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=36, vnum=36, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

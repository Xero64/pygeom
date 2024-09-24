#%%
# Import Dependencies
from numpy import cos, pi, sin
from numpy.typing import NDArray
from pygeom.geom2d import ParamSurface2D, Vector2D
from pygeom.tools.k3d import (Plot, k3d_surface, k3d_surface_normals,
                              k3d_surface_tangents)

#%%
# Create a Parametric Circle Surface
ro = 4.0
ri = 0.0

def ruv(u: NDArray, v: NDArray) -> Vector2D:
    radius = ro*(1.0 - v) + ri*v
    x = radius*cos(2*pi*u)
    y = radius*sin(2*pi*u)
    return Vector2D(x, y)

def drdv(u: NDArray, v: NDArray) -> Vector2D:
    radius_v = ri - ro
    x = radius_v*cos(2*pi*u)
    y = radius_v*sin(2*pi*u)
    return Vector2D(x, y)

def drdu(u: NDArray, v: NDArray) -> Vector2D:
    radius = ro*(1.0 - v) + ri*v
    x = -2*pi*radius*sin(2*pi*u)
    y = 2*pi*radius*cos(2*pi*u)
    vec = Vector2D(x, y)

    chk_r0 = radius == 0.0
    u_r0 = u[chk_r0]
    v_r0 = v[chk_r0]
    drdv_r0 = drdv(u_r0, v_r0)
    drdu_r0 = drdv_r0.rotate(-pi/2)
    vec[chk_r0] = drdu_r0

    return Vector2D(x, y)

paramsurface = ParamSurface2D(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=72, vnum=12)
k3dnrms = k3d_surface_normals(paramsurface, unum=72, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=72, vnum=12, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

#%%
# Create a Parametric Trapezoid Surface
pnta = Vector2D(-2.0, -1.5)
pntb = Vector2D(1.8, -1.7)
pntc = Vector2D(-1.9, 2.0)
pntd = Vector2D(2.1, 1.6)

def ruv(u: NDArray, v: NDArray) -> Vector2D:
    pnts = (pnta*(1.0 - u) + pntb*u)*(1.0 - v) + (pntc*(1.0 - u) + pntd*u)*v
    return pnts

def drdu(u: NDArray, v: NDArray) -> Vector2D:
    vecs = (pntb - pnta)*(1.0 - v) + (pntd - pntc)*v
    return vecs

def drdv(u: NDArray, v: NDArray) -> Vector2D:
    vecs = (pntc*(1.0 - u) + pntd*u) - (pnta*(1.0 - u) + pntb*u)
    return vecs

paramsurface = ParamSurface2D(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=12, vnum=12)
k3dnrms = k3d_surface_normals(paramsurface, unum=12, vnum=12, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=12, vnum=12, scale=0.2)

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

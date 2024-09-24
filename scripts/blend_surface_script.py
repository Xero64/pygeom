#%%
# Import Dependencies
from numpy import full, zeros
from numpy.typing import NDArray
from pygeom.geom3d import ParamCurve, ParamSurface, Vector
from pygeom.tools.k3d import (Plot, k3d_surface, k3d_surface_normals,
                              k3d_surface_tangents)

#%%
# Create Parametric End Curves
xle1 = 0.0
yle1 = 0.0
zle1 = 0.0
chrd1 = 1.0

def ru1(u: NDArray) -> Vector:
    x = xle1 + chrd1*u
    y = full(u.shape, yle1, dtype=u.dtype)
    z = full(u.shape, yle1, dtype=u.dtype)
    return Vector(x, y, z)

def drdu1(u: NDArray) -> Vector:
    x = full(u.shape, chrd1, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

def d2rdu2(u: NDArray) -> Vector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

paramcurve1 = ParamCurve(ru1, drdu1, d2rdu2)

xle2 = 0.1
yle2 = 5.0
zle2 = 0.1
chrd2 = 0.6

def ru2(u: NDArray) -> Vector:
    x = xle2 + chrd2*u
    y = full(u.shape, yle2, dtype=u.dtype)
    z = full(u.shape, zle2, dtype=u.dtype)
    return Vector(x, y, z)

def drdu2(u: NDArray) -> Vector:
    x = full(u.shape, chrd2, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

def d2rdu2(u: NDArray) -> Vector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

paramcurve2 = ParamCurve(ru2, drdu2, d2rdu2)

def ruv(u: NDArray, v: NDArray) -> Vector:
    ru1 = paramcurve1.evaluate_points_at_t(u)
    ru2 = paramcurve2.evaluate_points_at_t(u)
    ruv = ru1*(1 - v) + ru2*v
    return ruv

def drdu(u: NDArray, v: NDArray) -> Vector:
    drdu1 = paramcurve1.evaluate_first_derivatives_at_t(u)
    drdu2 = paramcurve2.evaluate_first_derivatives_at_t(u)
    drdu = drdu1*(1 - v) + drdu2*v
    return drdu

def drdv(u: NDArray, v: NDArray) -> Vector:
    ru1 = paramcurve1.evaluate_points_at_t(u)
    ru2 = paramcurve2.evaluate_points_at_t(u)
    drdv = ru2 - ru1
    return drdv

paramsurface = ParamSurface(ruv, drdu, drdv)

#%%
# Plot the Parametric Surface using K3D
k3dmesh = k3d_surface(paramsurface, unum=36, vnum=36)
k3dnrms = k3d_surface_normals(paramsurface, unum=36, vnum=36, scale=0.2)
k3dtgtsu, k3dtgtsv = k3d_surface_tangents(paramsurface, unum=36, vnum=36, scale=0.2)

k3dnrms.visible = False
k3dtgtsu.visible = False
k3dtgtsv.visible = False

plot = Plot()
plot += k3dmesh
plot += k3dnrms
plot += k3dtgtsu
plot += k3dtgtsv
plot.display()

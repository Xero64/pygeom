#%%
# Import Dependencies
from numpy import full, zeros
from numpy.typing import NDArray

from pygeom.geom3d import ParamCurve, ParamSurface, Vector
from pygeom.tools.k3d import (Plot, k3d_surface, k3d_surface_normals,
                              k3d_surface_tangents)

#%%
# Create Parametric End Curves
xle_0 = 0.1
yle_0 = 5.0
zle_0 = 0.1
chrd_0 = 0.6

def ru_0(u: NDArray) -> Vector:
    x = xle_0 + chrd_0*u
    y = full(u.shape, yle_0, dtype=u.dtype)
    z = full(u.shape, zle_0, dtype=u.dtype)
    return Vector(x, y, z)

def drdu_0(u: NDArray) -> Vector:
    x = full(u.shape, chrd_0, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

def d2rdu2_0(u: NDArray) -> Vector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

paramcurve_0 = ParamCurve(ru_0, drdu_0, d2rdu2_0)

xle_1 = 0.0
yle_1 = 0.0
zle_1 = 0.0
chrd_1 = 1.0

def ru_1(u: NDArray) -> Vector:
    x = xle_1 + chrd_1*u
    y = full(u.shape, yle_1, dtype=u.dtype)
    z = full(u.shape, yle_1, dtype=u.dtype)
    return Vector(x, y, z)

def drdu_1(u: NDArray) -> Vector:
    x = full(u.shape, chrd_1, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

def d2rdu2_1(u: NDArray) -> Vector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

paramcurve_1 = ParamCurve(ru_1, drdu_1, d2rdu2_1)

xle_2 = 0.1
yle_2 = 5.0
zle_2 = 0.1
chrd_2 = 0.6

def ru_2(u: NDArray) -> Vector:
    x = xle_2 + chrd_2*u
    y = full(u.shape, yle_2, dtype=u.dtype)
    z = full(u.shape, zle_2, dtype=u.dtype)
    return Vector(x, y, z)

def drdu_2(u: NDArray) -> Vector:
    x = full(u.shape, chrd_2, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

def d2rdu2_2(u: NDArray) -> Vector:
    x = zeros(u.shape, dtype=u.dtype)
    y = zeros(u.shape, dtype=u.dtype)
    z = zeros(u.shape, dtype=u.dtype)
    return Vector(x, y, z)

paramcurve_2 = ParamCurve(ru_2, drdu_2, d2rdu2_2)

paramcurves = zeros(3, dtype=ParamCurve)
paramcurves[0] = paramcurve_0
paramcurves[1] = paramcurve_1
paramcurves[2] = paramcurve_2

def ruv(u: NDArray, v: NDArray) -> Vector:
    ru1 = paramcurve_1.evaluate_points_at_t(u)
    ru2 = paramcurve_2.evaluate_points_at_t(u)
    ruv = ru1*(1 - v) + ru2*v
    return ruv

def drdu(u: NDArray, v: NDArray) -> Vector:
    drdu1 = paramcurve_1.evaluate_first_derivatives_at_t(u)
    drdu2 = paramcurve_2.evaluate_first_derivatives_at_t(u)
    drdu = drdu1*(1 - v) + drdu2*v
    return drdu

def drdv(u: NDArray, v: NDArray) -> Vector:
    ru1 = paramcurve_1.evaluate_points_at_t(u)
    ru2 = paramcurve_2.evaluate_points_at_t(u)
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

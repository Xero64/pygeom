#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import cos, float64, full, pi, sin, zeros
from numpy.typing import NDArray
from pygeom.array3d import ArrayVector, ParamCurve
from pygeom.tools.k3d import Plot, k3d_curve

#%%
# Parametric Helix
num = 200

radius = 2.0
TWOPI = 2*pi
TWOPI2 = TWOPI**2

height = 4.0

def ru(u: NDArray[float64]) -> ArrayVector:
    x = radius*cos(TWOPI*u)
    y = radius*sin(TWOPI*u)
    z = height*u
    return ArrayVector(x, y, z)

def drdu(u: NDArray[float64]) -> ArrayVector:
    x = -TWOPI*radius*sin(TWOPI*u)
    y = TWOPI*radius*cos(TWOPI*u)
    z = full(u.shape, height, dtype=u.dtype)
    return ArrayVector(x, y, z)

def d2rdu2(u: NDArray[float64]) -> ArrayVector:
    x = -TWOPI2*radius*cos(TWOPI*u)
    y = -TWOPI2*radius*sin(TWOPI*u)
    z = zeros(u.shape, dtype=u.dtype)
    return ArrayVector(x, y, z)

paramcurve = ParamCurve(ru, drdu, d2rdu2)

u = paramcurve.evaluate_u(num)
hpnts = paramcurve.evaluate_points(num)
hvecs = paramcurve.evaluate_first_derivatives(num)
hcurs = paramcurve.evaluate_second_derivatives(num)

hkappa = hvecs.cross(hcurs)/hvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hpnts.x, label='Helix X')
ax.plot(u, hpnts.y, label='Helix Y')
ax.plot(u, hpnts.z, label='Helix Z')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hvecs.x, label='Helix dXdu')
ax.plot(u, hvecs.y, label='Helix dYdu')
ax.plot(u, hvecs.z, label='Helix dZdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hcurs.x, label='Helix d2Xdu2')
ax.plot(u, hcurs.y, label='Helix d2Ydu2')
ax.plot(u, hcurs.z, label='Helix d2Zdu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hkappa.x, label='Helix Curvature X')
ax.plot(u, hkappa.y, label='Helix Curvature Y')
ax.plot(u, hkappa.z, label='Helix Curvature Z')
_ = ax.legend()

#%%
# k3d Plot
plot = Plot()
plot += k3d_curve(paramcurve, unum=num, color=0xff0000)
plot.display()

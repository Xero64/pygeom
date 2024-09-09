#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import arctan2, cos, full, hstack, pi, sin, zeros
from pygeom.geom2d import Vector2D
from pygeom.geom3d import NurbsCurve, Vector, zero_vector
from pygeom.tools.k3d import Plot, k3d_curve, k3d_nurbs_control_points, line

#%%
# Full Helix
num = 20
r = 2.0
l = 8.0

ctlpts = zero_vector(9)
ctlpts[0] = Vector(r, 0.0, 0*l/8)
ctlpts[1] = Vector(r, r, 1*l/8)
ctlpts[2] = Vector(0.0, r, 2*l/8)
ctlpts[3] = Vector(-r, r, 3*l/8)
ctlpts[4] = Vector(-r, 0.0, 4*l/8)
ctlpts[5] = Vector(-r, -r, 5*l/8)
ctlpts[6] = Vector(0.0, -r, 6*l/8)
ctlpts[7] = Vector(r, -r, 7*l/8)
ctlpts[8] = Vector(r, 0.0, 8*l/8)

weights = full(9, 1.0)
weights[1::2] = 1.0/2.0**0.5

nurbscurve = NurbsCurve(ctlpts, degree=2, weights=weights)
print(nurbscurve)

u = nurbscurve.evaluate_u(num)

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(Nu.shape[0]):
    ax.plot(u, Nu[i, :], label=f'N_{i}^{nurbscurve.degree}')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
for i in range(dNu.shape[0]):
    ax.plot(u, dNu[i, :], label=f'dN_{i}^{nurbscurve.degree}')
_ = ax.legend()

npnts = nurbscurve.evaluate_points(num)
nvecs = nurbscurve.evaluate_first_derivatives(num)
ncurs = nurbscurve.evaluate_second_derivatives(num)
nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

th = arctan2(npnts.y, npnts.x)
th[th < 0] += 2*pi
th[-1] = 2*pi

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

wpNu = Vector2D(wpNu.x, wpNu.y)
wpdNu = Vector2D(wpdNu.x, wpdNu.y)
wpd2Nu = Vector2D(wpd2Nu.x, wpd2Nu.y)

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)
z = l*th/2/pi

drdth = 0.0
dxdth = -r*sin(th)
dydth = r*cos(th)
dzdu = full(th.shape, l)

d2xdth2 = -r*cos(th)
d2ydth2 = -r*sin(th)
d2zdu2 = zeros(th.shape)

hpnts = Vector(x, y, z)
hvecs = Vector(dxdth*dthdu, dydth*dthdu, dzdu)
hcurs = Vector(d2xdth2*dthdu**2, d2ydth2*dthdu**2, d2zdu2) + Vector(dxdth*d2thdu2, dydth*d2thdu2, dzdu)

hkappa = hvecs.cross(hcurs)/hvecs.return_magnitude()**3

dpnts = hpnts - npnts
dvecs = hvecs - nvecs
dcurs = hcurs - ncurs
dkappa = hkappa - nkappa

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Helix Angle')
ax.plot(u, dthdu, label='Helix Angle Derivative')
ax.plot(u, d2thdu2, label='Helix Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(hpnts.x, hpnts.y, label='Helix Curve')
ax.scatter(hpnts.x, hpnts.y, label='Helix Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hpnts.x, label='Helix X')
ax.plot(u, hpnts.y, label='Helix Y')
ax.plot(u, hpnts.z, label='Helix Z')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
ax.plot(u, npnts.z, '-.', label='NURBS Z')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dpnts.x, label='Difference X')
ax.plot(u, dpnts.y, label='Difference Y')
ax.plot(u, dpnts.z, label='Difference Z')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hvecs.x, label='Helix dXdu')
ax.plot(u, hvecs.y, label='Helix dYdu')
ax.plot(u, hvecs.z, label='Helix dZdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
ax.plot(u, nvecs.z, '-.', label='NURBS dZdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dvecs.x, label='Difference dXdu')
ax.plot(u, dvecs.y, label='Difference dYdu')
ax.plot(u, dvecs.z, label='Difference dZdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hcurs.x, label='Helix d2Xdu2')
ax.plot(u, hcurs.y, label='Helix d2Ydu2')
ax.plot(u, hcurs.z, label='Helix d2Zdu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
ax.plot(u, ncurs.z, '-.', label='NURBS d2Zdu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dcurs.x, label='Difference d2Xdu2')
ax.plot(u, dcurs.y, label='Difference d2Ydu2')
ax.plot(u, dcurs.z, label='Difference d2Zdu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, hkappa.x, label='Helix Curvature X')
ax.plot(u, hkappa.y, label='Helix Curvature Y')
ax.plot(u, hkappa.z, label='Helix Curvature Z')
ax.plot(u, nkappa.x, '-.', label='NURBS Curvative X')
ax.plot(u, nkappa.y, '-.', label='NURBS Curvature Y')
ax.plot(u, nkappa.z, '-.', label='NURBS Curvature Z')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, dkappa.x, label='Difference Curvature X')
ax.plot(u, dkappa.y, label='Difference Curvature Y')
ax.plot(u, dkappa.z, label='Difference Curvature Z')
_ = ax.legend()

#%%
# k3d Plot
plot = Plot()
plot += k3d_curve(nurbscurve)
plot += k3d_nurbs_control_points(nurbscurve, scale=0.2)
plot += line(hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1))).astype('float32'),
             color=0xff0000)
plot.display()

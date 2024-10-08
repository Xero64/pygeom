#%%
# Import Dependencies
from matplotlib.pyplot import figure
from numpy import cos, ones, pi, sin, sqrt

from pygeom.geom2d import NurbsCurve2D, Vector2D
from pygeom.tools.k3d import Plot, k3d_curve, k3d_nurbs_control_points

#%%
# Quarter Ellipse
num = 20
a = 2.0
b = 1.0

ctlpts = Vector2D.zeros(3)
ctlpts[0] = Vector2D(a, 0.0)
ctlpts[1] = Vector2D(a, b)
ctlpts[2] = Vector2D(0.0, b)

w = sin(pi/4)

weights = ones(3)
weights[1::2] = w

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

u = nurbscurve.evaluate_t(num)

npnts = nurbscurve.evaluate_points_at_t(u)
nvecs = nurbscurve.evaluate_first_derivatives_at_t(u)
ncurs = nurbscurve.evaluate_second_derivatives_at_t(u)

nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

th = wpNu.return_angle()

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

r = a*b/sqrt(a**2*sin(th)**2 + b**2*cos(th)**2)
x = r*cos(th)
y = r*sin(th)

drdth = a*b*(b**2 - a**2)*sin(th)*cos(th)/(a**2*sin(th)**2 + b**2*cos(th)**2)**(3/2)
dxdth = drdth*cos(th) - r*sin(th)
dydth = drdth*sin(th) + r*cos(th)

d2rdth2 = a*b*(b**2 - a**2)*(3*(b**2 - a**2)*sin(th)**2*cos(th)**2 - (a**2*sin(th)**2 + b**2*cos(th)**2)*sin(th)**2 + (a**2*sin(th)**2 + b**2*cos(th)**2)*cos(th)**2)/(a**2*sin(th)**2 + b**2*cos(th)**2)**(5/2)
d2xdth2 = d2rdth2*cos(th) - 2*drdth*sin(th) - r*cos(th)
d2ydth2 = d2rdth2*sin(th) + 2*drdth*cos(th) - r*sin(th)

epnts = Vector2D(x, y)
evecs = Vector2D(dxdth, dydth)*dthdu
ecurs = Vector2D(d2xdth2, d2ydth2)*dthdu**2 + Vector2D(dxdth, dydth)*d2thdu2

ekappa = evecs.cross(ecurs)/evecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Ellipse Angle')
ax.plot(u, dthdu, label='Ellipse Angle Derivative')
ax.plot(u, d2thdu2, label='Ellipse Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(epnts.x, epnts.y, label='Ellipse Curve')
ax.scatter(epnts.x, epnts.y, label='Ellipse Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, epnts.x, label='Ellipse X')
ax.plot(u, epnts.y, label='Ellipse Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, evecs.x, label='Ellipse dXdu')
ax.plot(u, evecs.y, label='Ellipse dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ecurs.x, label='Ellipse d2Xdu2')
ax.plot(u, ecurs.y, label='Ellipse d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ekappa, label='Ellipse Curvature')
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# Quarter Circle
num = 20
r = 1.0

ctlpts = Vector2D.zeros(3)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)

w = sin(pi/4)

weights = ones(3)
weights[1::2] = w

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

u = nurbscurve.evaluate_t(num)
npnts = nurbscurve.evaluate_points(num)
nvecs = nurbscurve.evaluate_first_derivatives(num)
ncurs = nurbscurve.evaluate_second_derivatives(num)

nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

th = wpNu.return_angle()

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)

drdth = 0
dxdth = -r*sin(th)
dydth = r*cos(th)

d2xdth2 = -r*cos(th)
d2ydth2 = -r*sin(th)

cpnts = Vector2D(x, y)
cvecs = Vector2D(dxdth, dydth)*dthdu
ccurs = Vector2D(d2xdth2, d2ydth2)*dthdu**2 + Vector2D(dxdth, dydth)*d2thdu2

ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Circle Angle')
ax.plot(u, dthdu, label='Circle Angle Derivative')
ax.plot(u, d2thdu2, label='Circle Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cpnts.x, label='Circle X')
ax.plot(u, cpnts.y, label='Circle Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cvecs.x, label='Circle dXdu')
ax.plot(u, cvecs.y, label='Circle dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ccurs.x, label='Circle d2Xdu2')
ax.plot(u, ccurs.y, label='Circle d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ckappa, label='Circle Curvature')
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# Circle Arc
num = 20
r = 1.0
ang = pi/3
a = pi/2 - ang

ctlpts = Vector2D.zeros(3)
ctlpts[0] = Vector2D(r*cos(ang), -r*sin(ang))
ctlpts[1] = Vector2D(r/cos(ang), 0.0)
ctlpts[2] = Vector2D(r*cos(ang), r*sin(ang))

vec1 = ctlpts[1] - ctlpts[0]
vec2 = ctlpts[2] - ctlpts[1]

mag1 = vec1.return_magnitude()
mag2 = vec2.return_magnitude()

print(f'mag1 = {mag1}\n')
print(f'mag2 = {mag2}\n')

unit1 = vec1.to_unit()
unit2 = vec2.to_unit()

w = sin(a)

weights = ones(3)
weights[1::2] = w

nurbscurve = NurbsCurve2D(ctlpts, weights=weights)

print(nurbscurve)

u = nurbscurve.evaluate_t(num)
npnts = nurbscurve.evaluate_points(num)
nvecs = nurbscurve.evaluate_first_derivatives(num)
ncurs = nurbscurve.evaluate_second_derivatives(num)

nkappa = nvecs.cross(ncurs)/nvecs.return_magnitude()**3

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

th = wpNu.return_angle()

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)

drdth = 0
dxdth = -r*sin(th)
dydth = r*cos(th)

d2xdth2 = -r*cos(th)
d2ydth2 = -r*sin(th)

cpnts = Vector2D(x, y)
cvecs = Vector2D(dxdth, dydth)*dthdu
ccurs = Vector2D(d2xdth2, d2ydth2)*dthdu**2 + Vector2D(dxdth, dydth)*d2thdu2

ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Circle Angle')
ax.plot(u, dthdu, label='Circle Angle Derivative')
ax.plot(u, d2thdu2, label='Circle Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cpnts.x, label='Circle X')
ax.plot(u, cpnts.y, label='Circle Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cvecs.x, label='Circle dXdu')
ax.plot(u, cvecs.y, label='Circle dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ccurs.x, label='Circle d2Xdu2')
ax.plot(u, ccurs.y, label='Circle d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ckappa, label='Circle Curvature')
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# Full Circle
num = 20
r = 2.0

ctlpts = Vector2D.zeros(9)
ctlpts[0] = Vector2D(r, 0.0)
ctlpts[1] = Vector2D(r, r)
ctlpts[2] = Vector2D(0.0, r)
ctlpts[3] = Vector2D(-r, r)
ctlpts[4] = Vector2D(-r, 0.0)
ctlpts[5] = Vector2D(-r, -r)
ctlpts[6] = Vector2D(0.0, -r)
ctlpts[7] = Vector2D(r, -r)
ctlpts[8] = Vector2D(r, 0.0)

w = sin(pi/4)

weights = ones(9)
weights[1::2] = w

nurbscurve = NurbsCurve2D(ctlpts, weights=weights, degree=2)
print(f'nurbscurve.degree = {nurbscurve.degree}')
print(f'nurbscurve.knots = {nurbscurve.knots}')

u = nurbscurve.evaluate_t(num)

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

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

th = wpNu.return_angle()

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)

drdth = 0
dxdth = -r*sin(th)
dydth = r*cos(th)

d2xdth2 = -r*cos(th)
d2ydth2 = -r*sin(th)

cpnts = Vector2D(x, y)
cvecs = Vector2D(dxdth, dydth)*dthdu
ccurs = Vector2D(d2xdth2, d2ydth2)*dthdu**2 + Vector2D(dxdth, dydth)*d2thdu2

ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Circle Angle')
ax.plot(u, dthdu, label='Circle Angle Derivative')
ax.plot(u, d2thdu2, label='Circle Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cpnts.x, label='Circle X')
ax.plot(u, cpnts.y, label='Circle Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cvecs.x, label='Circle dXdu')
ax.plot(u, cvecs.y, label='Circle dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ccurs.x, label='Circle d2Xdu2')
ax.plot(u, ccurs.y, label='Circle d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ckappa, label='Circle Curvature')
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# Full Circle from 3 Points
num = 20
r = 2.0

ctlpts = Vector2D.zeros(7)
ctlpts[0] = Vector2D(0.0, -r)
ctlpts[1] = Vector2D(2*r*cos(pi/6), -r)
ctlpts[2] = Vector2D(r*cos(pi/6), r*sin(pi/6))
ctlpts[3] = Vector2D(0.0, r/sin(pi/6))
ctlpts[4] = Vector2D(-r*cos(pi/6), r*sin(pi/6))
ctlpts[5] = Vector2D(-2*r*cos(pi/6), -r)
ctlpts[6] = Vector2D(0.0, -r)

w = sin(pi/6)

weights = ones(7)
weights[1::2] = w

nurbscurve = NurbsCurve2D(ctlpts, weights=weights, degree=2)

print(f'nurbscurve.degree = {nurbscurve.degree}')
print(f'nurbscurve.knots = {nurbscurve.knots}')

u = nurbscurve.evaluate_t(num)

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

Nu = nurbscurve.basis_functions(u)
dNu = nurbscurve.basis_first_derivatives(u)
d2Nu = nurbscurve.basis_second_derivatives(u)

wpNu = nurbscurve.wpoints@Nu
wpdNu = nurbscurve.wpoints@dNu
wpd2Nu = nurbscurve.wpoints@d2Nu

th = wpNu.return_angle()

magwpNu2 = wpNu.return_magnitude()**2

dthdu = wpNu.cross(wpdNu)/magwpNu2
d2thdu2 = wpNu.cross(wpd2Nu)/magwpNu2 - wpNu.cross(wpdNu)*wpNu.dot(wpdNu)*2/magwpNu2**2

x = r*cos(th)
y = r*sin(th)

drdth = 0
dxdth = -r*sin(th)
dydth = r*cos(th)

d2xdth2 = -r*cos(th)
d2ydth2 = -r*sin(th)

cpnts = Vector2D(x, y)
cvecs = Vector2D(dxdth, dydth)*dthdu
ccurs = Vector2D(d2xdth2, d2ydth2)*dthdu**2 + Vector2D(dxdth, dydth)*d2thdu2

ckappa = cvecs.cross(ccurs)/cvecs.return_magnitude()**3

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, th, label='Circle Angle')
ax.plot(u, dthdu, label='Circle Angle Derivative')
ax.plot(u, d2thdu2, label='Circle Angle Second Derivative')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(cpnts.x, cpnts.y, label='Circle Curve')
ax.scatter(cpnts.x, cpnts.y, label='Circle Points')
ax.plot(npnts.x, npnts.y, '-.', label='NURBS Curve')
ax.scatter(npnts.x, npnts.y, label='NURBS Points')
ax.set_aspect('equal')
ax.scatter(ctlpts.x, ctlpts.y, color='r', label='Control Points')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cpnts.x, label='Circle X')
ax.plot(u, cpnts.y, label='Circle Y')
ax.plot(u, npnts.x, '-.', label='NURBS X')
ax.plot(u, npnts.y, '-.', label='NURBS Y')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, cvecs.x, label='Circle dXdu')
ax.plot(u, cvecs.y, label='Circle dYdu')
ax.plot(u, nvecs.x, '-.', label='NURBS dXdu')
ax.plot(u, nvecs.y, '-.', label='NURBS dYdu')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ccurs.x, label='Circle d2Xdu2')
ax.plot(u, ccurs.y, label='Circle d2Ydu2')
ax.plot(u, ncurs.x, '-.', label='NURBS d2Xdu2')
ax.plot(u, ncurs.y, '-.', label='NURBS d2Ydu2')
_ = ax.legend()

fig = figure(figsize=(12, 8))
ax = fig.gca()
ax.grid(True)
ax.plot(u, ckappa, label='Circle Curvature')
ax.plot(u, nkappa, '-.', label='NURBS Curvative')
_ = ax.legend()

#%%
# k3d Plot
plot = Plot()
plot += k3d_curve(nurbscurve)
plot += k3d_nurbs_control_points(nurbscurve, scale=0.2)
plot.display()

#%%
# Import Dependencies
from math import pi, cos, sin
from pygeom.geom3d import Vector, point_from_lists
from pygeom.geom3d.cubicspline import CubicSpline

#%%
# Create Spline 1
num = 36
R = 2.0
th = [i/num*2*pi for i in range(num)]
x = [0.0 for _ in th]
y = [R*cos(thi) for thi in th]
z = [R*sin(thi) for thi in th]

pnts = point_from_lists(x, y, z)

TestSpline = CubicSpline(pnts)

ax = TestSpline.scatter(label=True)
ax = TestSpline.plot_spline(50, ax=ax)

#%%
# Create Spline 2
thA = pi/2
tanA = Vector(0.0, cos(thA), sin(thA))
thB = pi/2-2*pi/num
tanB = Vector(0.0, cos(thB), sin(thB))

TestSpline2 = CubicSpline(pnts, tanA=tanA, tanB=tanB)

ax = TestSpline2.plot_spline(50, ax=ax, color='green')

ax = TestSpline2.plot_gradient()
ax = TestSpline2.plot_curvature()

ax = TestSpline2.scatter(label=True)
ax = TestSpline2.plot_spline(50, ax=ax)

#%%
# Create Spline 3
TestSpline3 = CubicSpline(pnts, clsd=True)

ax = TestSpline3.plot_spline(50, ax=ax, color='red')

ax = TestSpline3.plot_gradient()
ax = TestSpline3.plot_curvature()

ax = TestSpline3.scatter(label=True)
ax = TestSpline3.plot_spline(50, ax=ax)
